"""Data pipeline for the v2 protocol.

* Face crops were produced by the authors' detector.py (commit 387a422) with the README flags
  (-m at_least, --det_thres 0.1, -s 1.3, --target_size none), 8 frames per video, through the
  output-identical fast reader in fast_frames.py.
* Augmentations and CLIP preprocessing are the OFFICIAL ones (src.dataset.augmentations,
  CLIPProcessor), applied in the same order as the official dataset: augment(PIL) -> preprocess.
* Training batches come from PairedStratifiedBatchSampler, which is shared by EVERY arm, so all
  arms see byte-identical batches for a given (seed, epoch). Augmentation randomness is seeded
  per (seed, epoch, sample-occurrence), so it is identical across arms too.
"""
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .paths import V2_DIR, WORK_DIR

DRIVE_V2 = V2_DIR
LOCAL_CROPS = WORK_DIR / "crops"
FF_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# ----------------------------------------------------------------------------- index / crops
def load_index(v2: Path = DRIVE_V2, name: str = "crops_index_v2.csv") -> pd.DataFrame:
    return pd.read_csv(v2 / "manifest" / name, dtype={"frame_ids": str})


def ensure_crops(groups, v2: Path = DRIVE_V2, local: Path = LOCAL_CROPS, idx: pd.DataFrame | None = None):
    """Extract per-group archives '<split>__<method>.tar' from Drive unless already complete locally."""
    local.mkdir(parents=True, exist_ok=True)
    idx = load_index(v2) if idx is None else idx
    for g in groups:
        split, method = g.split("__")
        rels = idx[(idx.split == split) & (idx.method == method)].rel
        complete = all(len(list((local / r.replace(".mp4", "")).glob("*.png"))) >= 8 for r in rels)
        if complete:
            continue
        with tarfile.open(v2 / "crops" / f"{g}.tar") as tf:
            tf.extractall(local)


def frame_table(idx: pd.DataFrame, split: str, methods=None) -> pd.DataFrame:
    """One row per face crop: path (relative to crops root), label, method, video, target id, frame id."""
    sub = idx[idx.split.isin([split] if isinstance(split, str) else list(split))]
    if methods is not None:
        sub = sub[sub.method.isin(methods)]
    rows = []
    for r in sub.itertuples():
        vdir = r.rel.replace(".mp4", "")
        target = Path(r.rel).stem.split("_")[0]
        for f in str(r.frame_ids).split():
            fid = int(f)
            rows.append(dict(path=f"{vdir}/frame_{fid:04d}.png", label=int(r.label), method=r.method,
                             video=vdir, target=target, frame=fid))
    return pd.DataFrame(rows).reset_index(drop=True)


# ----------------------------------------------------------------------------- transforms
def official_augmentations():
    """The paper's training augmentations: default C.Augmentations() (flip .5, affine 10deg/0.1/[.9,1.1],
    blur p=.1 k=7 s=[.1,2], color jitter .1/.1, JPEG q in [40,100])."""
    from src import config as C
    from src.dataset.augmentations import init_augmentations
    return init_augmentations(C.Augmentations())


def make_preprocess(clip_processor):
    """Identical to CLIPEncoder.preprocess (custom_preprocessing is None in the paper config)."""
    def preprocess(image):
        return clip_processor(images=image, return_tensors="pt")["pixel_values"][0]
    return preprocess


# ----------------------------------------------------------------------------- dataset
class CropDataset(Dataset):
    """Index convention: an index may encode an occurrence counter, idx = i + occ * N, so that a real crop
    reused by several pairs in one epoch receives a different (but reproducible) augmentation each time."""

    def __init__(self, table: pd.DataFrame, preprocess, augment=None, root: Path = LOCAL_CROPS, seed: int = 0):
        self.paths = table.path.tolist()
        self.labels = table.label.astype(int).tolist()
        self.N = len(self.paths)
        self.pre, self.aug, self.root, self.seed, self.epoch = preprocess, augment, Path(root), seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        i = idx % self.N
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        if self.aug is not None:
            s = (self.seed * 1_000_003 + self.epoch * 100_003 + idx) % (2**31 - 1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(s)
                img = self.aug(img)
        return self.pre(img), self.labels[i], i


# ----------------------------------------------------------------------------- paired sampler
class PairedStratifiedBatchSampler(Sampler):
    """Each batch holds P pairs, P/len(methods) per manipulation method (the V-REx environments).

    Pair = (fake crop of video '<t>_<s>' at frame k, real crop of video '<t>' at the same frame k,
    or the nearest saved frame of that real video if k was not saved).
    Batch layout: [real_1..real_P, fake_1..fake_P]; pair i = rows (i, P+i); method of pair i = i // (P/len(methods)).
    """

    def __init__(self, table: pd.DataFrame, pairs_per_batch: int = 16, methods=FF_METHODS, seed: int = 0):
        assert pairs_per_batch % len(methods) == 0, "pairs_per_batch must be divisible by number of methods"
        self.P, self.methods, self.seed, self.epoch = pairs_per_batch, list(methods), seed, 0
        self.k = pairs_per_batch // len(methods)
        self.N = len(table)
        real = table[table.label == 0]
        real_by_target: dict[str, dict[int, int]] = {}
        for i, tgt, fr in zip(real.index, real.target, real.frame):
            real_by_target.setdefault(tgt, {})[int(fr)] = int(i)
        self.fake_by_method, self.partner, self.exact = {}, {}, {}
        for m in self.methods:
            ids = table.index[(table.label == 1) & (table.method == m)].to_numpy()
            assert len(ids) > 0, f"no fakes for {m}"
            self.fake_by_method[m] = ids
            n_exact = 0
            for i in ids:
                tgt, fr = table.at[i, "target"], int(table.at[i, "frame"])
                cand = real_by_target[tgt]
                if fr in cand:
                    j = cand[fr]
                    n_exact += 1
                else:
                    j = cand[min(cand, key=lambda c: (abs(c - fr), c))]
                self.partner[int(i)] = j
            self.exact[m] = n_exact / len(ids)
        self.n_per_method = min(len(v) for v in self.fake_by_method.values())

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.n_per_method // self.k

    def __iter__(self):
        rng = np.random.RandomState(self.seed * 1000 + self.epoch)
        perms = {m: rng.permutation(self.fake_by_method[m])[: self.n_per_method] for m in self.methods}
        occ: dict[int, int] = {}
        for b in range(len(self)):
            fakes = np.concatenate([perms[m][b * self.k:(b + 1) * self.k] for m in self.methods])
            reals = [self.partner[int(i)] for i in fakes]
            batch = []
            for j in list(reals) + [int(i) for i in fakes]:
                o = occ.get(j, 0)
                occ[j] = o + 1
                batch.append(int(j + o * self.N))
            yield batch
