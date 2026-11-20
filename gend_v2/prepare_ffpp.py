"""FF++ (c23) data preparation for the v2 protocol: manifest -> download -> authors' face preprocessing.

  python -m gend_v2.prepare_ffpp                     # all stages (≈1 min download, ≈70 min preprocessing on a T4)
  python -m gend_v2.prepare_ffpp --verify_only       # only the fast-reader equivalence check (8 videos)

Requires: accepted FaceForensics++ terms (https://github.com/ondyari/FaceForensics), the official GenD repo at
commit 387a422 in GEND_OFFICIAL_DIR, onnxruntime-gpu, imageio, imageio-ffmpeg, av.

Protocol (fixed before any training):
* Official FF++ splits (train/val/test JSON from the FaceForensics repo); no source id is shared across splits.
* train: first 120 of the 360 official train pairs -> 240 originals + 240 fakes for each of DF, F2F, FS, NT.
  val  : all 70 val pairs -> 140 originals + 140 FaceShifter fakes (unseen generator, used for model selection).
  test : all 70 test pairs -> 140 originals + 140 fakes for each of DF, F2F, FS, NT, FaceShifter.
* Fake '<t>_<s>' is paired with original '<t>' (target video: same scene/background; verified on pixels).
* Faces: the authors' detector.process_video with the README flags (-m at_least -n 8 --det_thres 0.1 -s 1.3
  --target_size none), no masks for FF++. Frames are read with gend_v2.fast_frames (pixel-identical to the
  authors' reader, checked by --verify_only / stage 4 on every manipulation type; ≈4.3x faster).
Outputs (V2_DIR): manifest/manifest_v2.csv, manifest/official_*.json, crops/<split>__<method>.tar,
manifest/crops_index_v2.csv.
"""
import argparse
import builtins
import contextlib
import glob
import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .paths import GEND_ROOT, V2_DIR, WORK_DIR

FF = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
BASE = "http://kaldir.vc.in.tum.de/faceforensics/v3/"
RAW, CROPS = WORK_DIR / "raw", WORK_DIR / "crops"
N_TRAIN_PAIRS = 120


def build_manifest():
    (V2_DIR / "manifest").mkdir(parents=True, exist_ok=True)
    splits = {}
    for s in ["train", "val", "test"]:
        url = f"https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{s}.json"
        splits[s] = json.loads(urllib.request.urlopen(url, timeout=30).read())
        (V2_DIR / "manifest" / f"official_{s}.json").write_text(json.dumps(splits[s]))
    ids = {s: {str(x) for p in v for x in p} for s, v in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (ids[a] & ids[b]), f"leak {a}/{b}"
    use = {"train": splits["train"][:N_TRAIN_PAIRS], "val": splits["val"], "test": splits["test"]}
    rows = []
    for s, pairs in use.items():
        for a, b in pairs:
            for tgt, src in [(a, b), (b, a)]:
                rows.append(dict(split=s, label=0, method="real", video_id=tgt, target_id=tgt,
                                 rel=f"original_sequences/youtube/c23/videos/{tgt}.mp4"))
                methods = (FF if s in ("train", "test") else []) + (["FaceShifter"] if s in ("val", "test") else [])
                for m in methods:
                    rows.append(dict(split=s, label=1, method=m, video_id=f"{tgt}_{src}", target_id=tgt,
                                     rel=f"manipulated_sequences/{m}/c23/videos/{tgt}_{src}.mp4"))
    man = pd.DataFrame(rows)
    assert not man.duplicated(["rel"]).any()
    print("[1] manifest:", len(man), "videos\n", man.groupby(["split", "method"]).size().unstack(fill_value=0))
    return man


def verify(p: Path):
    if not p.exists() or p.stat().st_size == 0:
        return False, 0
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()
    return bool(ok and n > 0), n


def download(man):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dl = WORK_DIR / "faceforensics_download_v4.py"
    if not dl.exists():
        urllib.request.urlretrieve("https://kaldir.vc.in.tum.de/faceforensics_download_v4.py", dl)
    spec = importlib.util.spec_from_file_location("ffpp_dl", dl)
    ffpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ffpp)

    def fetch(rel):
        dst = RAW / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok, n = verify(dst)
        tries, err = 0, ""
        while not ok and tries < 3:
            tries += 1
            try:
                if dst.exists():
                    dst.unlink()
                ffpp.download_file(BASE + rel, str(dst))
            except Exception as e:
                err = str(e)[:120]
                time.sleep(2)
            ok, n = verify(dst)
        return rel, ok, n, (dst.stat().st_size if dst.exists() else 0), err

    t0, res = time.time(), {}
    with ThreadPoolExecutor(6) as ex:
        for f in as_completed([ex.submit(fetch, r) for r in man.rel]):
            rel, ok, n, size, err = f.result()
            res[rel] = (ok, n, size, err)
    man["valid"] = man.rel.map(lambda r: res[r][0])
    man["n_frames"] = man.rel.map(lambda r: res[r][1])
    man["size_mb"] = man.rel.map(lambda r: round(res[r][2] / 1e6, 3))
    man["error"] = man.rel.map(lambda r: res[r][3])
    print(f"[2] downloaded: valid {int(man.valid.sum())}/{len(man)}, {man.size_mb.sum() / 1e3:.2f} GB, {time.time() - t0:.0f}s")
    assert man.valid.all(), man[~man.valid][["rel", "error"]].head()
    man.to_csv(V2_DIR / "manifest" / "manifest_v2.csv", index=False)
    return man


def _setup_detector():
    builtins.exit = sys.exit          # detector.py ends with a bare exit(0)
    os.chdir(GEND_ROOT)
    if GEND_ROOT not in sys.path:
        sys.path.insert(0, GEND_ROOT)
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:
            print("preload_dlls failed:", e)
    import detector
    from src.retinaface import prepare_model
    from . import fast_frames as ff
    model = prepare_model(0.1)
    print("    detector providers:", model.session.get_providers())
    return detector, ff, model


def _run(detector, ff, model, rels, out_root, fast, workers=4):
    (ff.install if fast else ff.uninstall)()

    def one(rel):
        d = out_root / rel.replace(".mp4", "")
        if d.exists() and len(list(d.glob("*.png"))) >= 8:
            return
        detector.process_video(str(RAW / rel), str(d), None, model, scale=1.3, target_size=None,
                               num_frames=8, mode="at_least")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, rels))
    ff.uninstall()
    return [l for l in buf.getvalue().splitlines() if any(k in l for k in ("Warning", "No face", "Error"))]


def verify_fast_reader(man, detector, ff, model):
    pick = []
    for m in ["real"] + FF + ["FaceShifter"]:
        pick += man[(man.split == "test") & (man.method == m)].rel.head(1).tolist()
    pick += man[(man.split == "train") & (man.method == "real")].rel.head(2).tolist()
    O, F = WORK_DIR / "eq_orig", WORK_DIR / "eq_fast"
    t = time.time(); _run(detector, ff, model, pick, O, fast=False); t_o = time.time() - t
    t = time.time(); _run(detector, ff, model, pick, F, fast=True); t_f = time.time() - t
    ok_all = True
    for rel in pick:
        A = sorted(glob.glob(str(O / rel.replace(".mp4", "")) + "/*.png"))
        B = sorted(glob.glob(str(F / rel.replace(".mp4", "")) + "/*.png"))
        ok = [Path(a).name for a in A] == [Path(b).name for b in B] and len(A) > 0 and all(
            np.array_equal(cv2.imread(a, cv2.IMREAD_UNCHANGED), cv2.imread(b, cv2.IMREAD_UNCHANGED)) for a, b in zip(A, B))
        ok_all &= ok
        print(f"    {rel:70s} identical={ok}")
    print(f"[4] fast reader pixel-identical on all methods: {ok_all} | original {t_o:.0f}s vs fast {t_f:.0f}s")
    assert ok_all, "fast reader is NOT output-identical -> do not use"


def preprocess(man, detector, ff, model):
    arch = V2_DIR / "crops"
    arch.mkdir(parents=True, exist_ok=True)
    order = [("train", m) for m in ["real"] + FF] + [("val", "real"), ("val", "FaceShifter")] + \
            [("test", m) for m in ["real"] + FF + ["FaceShifter"]]
    for split, meth in order:
        rels = man[(man.split == split) & (man.method == meth)].rel.tolist()
        tar_path = arch / f"{split}__{meth}.tar"
        if tar_path.exists():
            print(f"    skip {split}/{meth}: archive exists")
            continue
        t0 = time.time()
        warn = _run(detector, ff, model, rels, CROPS, fast=True)
        local = WORK_DIR / f"{split}__{meth}.tar"
        with tarfile.open(local, "w") as tf:
            for r in rels:
                d = CROPS / r.replace(".mp4", "")
                if d.exists():
                    tf.add(d, arcname=r.replace(".mp4", ""))
        shutil.copy(local, str(tar_path) + ".part")
        os.replace(str(tar_path) + ".part", tar_path)
        print(f"    {split}/{meth}: {len(rels)} videos, {time.time() - t0:.0f}s, warnings {len(warn)}")
    rows = []
    for r in man.itertuples():
        d = CROPS / r.rel.replace(".mp4", "")
        if not d.exists():
            with tarfile.open(arch / f"{r.split}__{r.method}.tar") as tf:
                tf.extractall(CROPS)
        fids = sorted(int(p.stem.split("_")[1]) for p in d.glob("*.png"))
        rows.append(dict(rel=r.rel, split=r.split, method=r.method, label=r.label, n_crops=len(fids),
                         frame_ids=" ".join(map(str, fids))))
    idx = pd.DataFrame(rows)
    idx.to_csv(V2_DIR / "manifest" / "crops_index_v2.csv", index=False)
    print("[5] crops per video:", idx.n_crops.value_counts().sort_index().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()
    man = build_manifest()
    man = download(man)
    detector, ff, model = _setup_detector()
    verify_fast_reader(man, detector, ff, model)
    if not args.verify_only:
        preprocess(man, detector, ff, model)
    print("PREPARE_FFPP: COMPLETE")


if __name__ == "__main__":
    main()
