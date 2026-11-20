"""Scale-up data (pre-registered follow-up): the remaining 240 official FF++ train pairs (pairs 120..359).

  python -m gend_v2.prepare_full --workers 8

Identical pipeline to prepare_ffpp.py (official downloader, authors' detector.process_video with the README
flags, 8 frames/video, verified fast reader). New videos get split='train2'; archives crops/train2__<method>.tar;
index manifest/crops_index_full.csv = crops_index_v2.csv + the train2 rows. Val/test are unchanged.
"""
import argparse
import os
import shutil
import tarfile
import time

import pandas as pd

from . import prepare_ffpp as P
from .paths import V2_DIR, WORK_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import json
    tr = json.loads((V2_DIR / "manifest" / "official_train.json").read_text())
    pairs = tr[P.N_TRAIN_PAIRS:]
    rows = []
    for a, b in pairs:
        for tgt, src in [(a, b), (b, a)]:
            rows.append(dict(split="train2", label=0, method="real", video_id=tgt, target_id=tgt,
                             rel=f"original_sequences/youtube/c23/videos/{tgt}.mp4"))
            for m in P.FF:
                rows.append(dict(split="train2", label=1, method=m, video_id=f"{tgt}_{src}", target_id=tgt,
                                 rel=f"manipulated_sequences/{m}/c23/videos/{tgt}_{src}.mp4"))
    man = pd.DataFrame(rows)
    old = pd.read_csv(V2_DIR / "manifest" / "manifest_v2.csv", dtype={"video_id": str, "target_id": str})
    assert not set(man.rel) & set(old.rel), "overlap with existing manifest"
    test_val_ids = set(old[old.split != "train"].target_id)
    assert not set(man.target_id) & test_val_ids, "train2 leaks into val/test"
    print(f"[1] train2: {len(pairs)} pairs, {len(man)} videos;", man.method.value_counts().to_dict())

    # download (reuses the verified downloader)
    t0 = time.time()
    man = _download(man)
    print(f"[2] downloaded {int(man.valid.sum())}/{len(man)} in {time.time() - t0:.0f}s")
    man.to_csv(V2_DIR / "manifest" / "manifest_train2.csv", index=False)

    detector, ff, model = P._setup_detector()
    arch = V2_DIR / "crops"
    for meth in ["real"] + P.FF:
        rels = man[man.method == meth].rel.tolist()
        tar_path = arch / f"train2__{meth}.tar"
        if tar_path.exists():
            print(f"    skip train2/{meth}: archive exists")
            continue
        t = time.time()
        warn = P._run(detector, ff, model, rels, P.CROPS, fast=True, workers=args.workers)
        local = WORK_DIR / f"train2__{meth}.tar"
        with tarfile.open(local, "w") as tf:
            for r in rels:
                d = P.CROPS / r.replace(".mp4", "")
                if d.exists():
                    tf.add(d, arcname=r.replace(".mp4", ""))
        shutil.copy(local, str(tar_path) + ".part")
        os.replace(str(tar_path) + ".part", tar_path)
        print(f"[3] train2/{meth}: {len(rels)} videos, {time.time() - t:.0f}s, warnings {len(warn)}, "
              f"tar {tar_path.stat().st_size / 1e6:.0f} MB", flush=True)

    rows = []
    for r in man.itertuples():
        d = P.CROPS / r.rel.replace(".mp4", "")
        fids = sorted(int(p.stem.split("_")[1]) for p in d.glob("*.png")) if d.exists() else []
        rows.append(dict(rel=r.rel, split="train2", method=r.method, label=r.label, n_crops=len(fids),
                         frame_ids=" ".join(map(str, fids))))
    new = pd.DataFrame(rows)
    base = pd.read_csv(V2_DIR / "manifest" / "crops_index_v2.csv", dtype={"frame_ids": str})
    full = pd.concat([base, new[[c for c in base.columns if c in new.columns]]], ignore_index=True)
    full.to_csv(V2_DIR / "manifest" / "crops_index_full.csv", index=False)
    print("[4] train2 crops per video:", new.n_crops.value_counts().sort_index().to_dict())
    print("    full index:", full.groupby(["split", "method"]).size().unstack(fill_value=0).to_string())
    print("PREPARE_FULL: COMPLETE")


def _download(man):
    """Same as prepare_ffpp.download but does not overwrite manifest_v2.csv."""
    import importlib.util
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dl = WORK_DIR / "faceforensics_download_v4.py"
    if not dl.exists():
        urllib.request.urlretrieve("https://kaldir.vc.in.tum.de/faceforensics_download_v4.py", dl)
    spec = importlib.util.spec_from_file_location("ffpp_dl", dl)
    ffpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ffpp)

    def fetch(rel):
        dst = P.RAW / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok, n = P.verify(dst)
        tries = 0
        while not ok and tries < 3:
            tries += 1
            try:
                if dst.exists():
                    dst.unlink()
                ffpp.download_file(P.BASE + rel, str(dst))
            except Exception:
                time.sleep(2)
            ok, n = P.verify(dst)
        return rel, ok, n
    with ThreadPoolExecutor(6) as ex:
        res = {r: (ok, n) for r, ok, n in ex.map(fetch, man.rel)}
    man["valid"] = man.rel.map(lambda r: res[r][0])
    man["n_frames"] = man.rel.map(lambda r: res[r][1])
    assert man.valid.all(), man[~man.valid].rel.head().tolist()
    return man


if __name__ == "__main__":
    main()
