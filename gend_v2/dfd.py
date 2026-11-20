"""Build the cross-dataset test set: Google/Jigsaw DeepFakeDetection (DFD), distributed with FF++.

  python -m gend_v2.dfd --n_real 200 --n_fake 200 --seed 0

* File list from the FF++ server (misc/deepfake_detection_filenames.json); a seeded random subset
  of real (original_sequences/actors) and fake (manipulated_sequences/DeepFakeDetection) c23 videos.
* Preprocessing = authors' detector.process_video with the README flags; fakes use the official
  manipulation masks (as in the README), reals use none. 8 frames per video, as for FF++.
* Before the bulk run, the fast readers are checked to be pixel-identical to the authors' reader on
  a few DFD videos (with and without masks); the script aborts otherwise.
* Outputs: v2/crops/test__DFD_real.tar, v2/crops/test__DFD_fake.tar, v2/manifest/crops_index_dfd.csv,
  v2/manifest/dfd_selection.json. Never touches the FF++ files.
"""
import argparse
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .paths import GEND_ROOT as GEND, V2_DIR as V2, WORK_DIR
BASE = "http://kaldir.vc.in.tum.de/faceforensics/v3/"
RAW = WORK_DIR / "raw_dfd"
CROPS = WORK_DIR / "crops"
REAL_DIR = "original_sequences/actors/c23/videos"
FAKE_DIR = "manipulated_sequences/DeepFakeDetection/c23/videos"
MASK_DIR = "manipulated_sequences/DeepFakeDetection/masks/videos"


def get_downloader():
    dl = WORK_DIR / "faceforensics_download_v4.py"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not dl.exists():
        urllib.request.urlretrieve("https://kaldir.vc.in.tum.de/faceforensics_download_v4.py", dl)
    spec = importlib.util.spec_from_file_location("ffpp_dl", dl)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def verify(p: Path):
    if not p.exists() or p.stat().st_size == 0:
        return False, 0
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()
    return bool(ok and n > 0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_real", type=int, default=200)
    ap.add_argument("--n_fake", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import builtins
    builtins.exit = sys.exit
    os.chdir(GEND)
    sys.path.insert(0, GEND)
    T0 = time.time()

    # ---------------- [1] selection
    names = json.loads(urllib.request.urlopen(BASE + "misc/deepfake_detection_filenames.json", timeout=60).read())
    reals_all, fakes_all = sorted(names["actors"]), sorted(names["DeepFakesDetection"])
    rng = np.random.RandomState(args.seed)
    reals = sorted(rng.choice(reals_all, args.n_real, replace=False).tolist())
    fakes = sorted(rng.choice(fakes_all, args.n_fake, replace=False).tolist())
    sel = dict(seed=args.seed, n_real_total=len(reals_all), n_fake_total=len(fakes_all), reals=reals, fakes=fakes)
    (V2 / "manifest").mkdir(parents=True, exist_ok=True)
    (V2 / "manifest" / "dfd_selection.json").write_text(json.dumps(sel, indent=1))
    print(f"[1] DFD lists: {len(reals_all)} real, {len(fakes_all)} fake -> selected {len(reals)} + {len(fakes)}", flush=True)

    # ---------------- [2] download videos (+ masks for fakes)
    ffpp = get_downloader()
    jobs = [f"{REAL_DIR}/{n}.mp4" for n in reals] + [f"{FAKE_DIR}/{n}.mp4" for n in fakes] + \
           [f"{MASK_DIR}/{n}.mp4" for n in fakes]

    def fetch(rel):
        dst = RAW / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok, n = verify(dst)
        tries = 0
        while not ok and tries < 3:
            tries += 1
            try:
                if dst.exists():
                    dst.unlink()
                ffpp.download_file(BASE + rel, str(dst))
            except Exception:
                time.sleep(2)
            ok, n = verify(dst)
        return rel, ok, n

    t = time.time()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch, jobs))
    bad = [r for r in res if not r[1]]
    nframes = {r[0]: r[2] for r in res}
    size = sum((RAW / r[0]).stat().st_size for r in res if (RAW / r[0]).exists()) / 1e9
    print(f"[2] downloaded {len(res) - len(bad)}/{len(res)} files, {size:.2f} GB, {time.time() - t:.0f}s; failed: {bad[:5]}", flush=True)
    assert not bad, "download failures -> rerun"

    # ---------------- [3] equivalence of fast readers on DFD (2 reals no-mask, 3 fakes with masks)
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # falls back to CPU detection if CUDA libs are not loadable
            print("    preload_dlls failed:", e)
    import detector
    sys.path.insert(0, str(V2 / "code"))
    import gend_v2.fast_frames as ff
    from src.retinaface import prepare_model
    model = prepare_model(0.1)
    print("    detector providers in use:", model.session.get_providers(), flush=True)

    def run(items, root, mode, workers):
        if mode == "orig":
            ff.uninstall()
        else:
            ff.install_masks()

        def one(it):
            rel, mask = it
            d = root / rel.replace(".mp4", "")
            if d.exists() and len(list(d.glob("*.png"))) >= 8:
                return
            detector.process_video(str(RAW / rel), str(d), None if mask is None else str(RAW / mask), model,
                                   scale=1.3, target_size=None, num_frames=8, mode="at_least")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(one, items))
        ff.uninstall()
        return [l for l in buf.getvalue().splitlines() if ("Warning" in l or "No face" in l or "No suitable" in l or "Error" in l)]

    items_real = [(f"{REAL_DIR}/{n}.mp4", None) for n in reals]
    items_fake = [(f"{FAKE_DIR}/{n}.mp4", f"{MASK_DIR}/{n}.mp4") for n in fakes]

    # frame-count metadata check (the authors' reader crashes when it is missing)
    fc = {rel: ff.n_frames(str(RAW / rel)) for rel in jobs}
    no_meta = {k: [r for r in jobs if r.startswith(d) and fc[r][1]]
               for k, d in [("real", REAL_DIR), ("fake", FAKE_DIR), ("mask", MASK_DIR)]}
    print("    files WITHOUT frame-count metadata (count taken by demuxing):",
          {k: len(v) for k, v in no_meta.items()}, flush=True)
    meta_ok = lambda it: not fc[it[0]][1] and (it[1] is None or not fc[it[1]][1])
    # equivalence is only testable where the authors' reader runs at all
    eq_items = [it for it in items_real if meta_ok(it)][:2] + [it for it in items_fake if meta_ok(it)][:3]
    print(f"    equivalence test on {len(eq_items)} videos with metadata (authors' reader works there)", flush=True)
    O, F = WORK_DIR / "dfd_eq_orig", WORK_DIR / "dfd_eq_fast"
    t = time.time(); run(eq_items, O, "orig", 2); t_o = time.time() - t
    t = time.time(); run(eq_items, F, "fast", 2); t_f = time.time() - t
    all_ok = True
    for rel, _ in eq_items:
        A = sorted(glob.glob(str(O / rel.replace(".mp4", "")) + "/*.png"))
        B = sorted(glob.glob(str(F / rel.replace(".mp4", "")) + "/*.png"))
        same = [Path(a).name for a in A] == [Path(b).name for b in B] and len(A) > 0 and all(
            np.array_equal(cv2.imread(a, cv2.IMREAD_UNCHANGED), cv2.imread(b, cv2.IMREAD_UNCHANGED)) for a, b in zip(A, B))
        all_ok &= same
        print(f"    eq {Path(rel).stem[:40]:40s} orig={len(A)} fast={len(B)} identical={same}", flush=True)
    print(f"[3] DFD fast-reader equivalence: {all_ok} | 5 videos orig {t_o:.0f}s vs fast {t_f:.0f}s", flush=True)
    assert all_ok, "fast reader NOT identical on DFD -> stop, do not use"

    # ---------------- [4] bulk preprocessing (fast readers), archive to Drive
    rows, warn = [], []
    for tag, items, label in [("real", items_real, 0), ("fake", items_fake, 1)]:
        t = time.time()
        warn += run(items, CROPS, "fast", args.workers)
        local_tar = WORK_DIR / f"test__DFD_{tag}.tar"
        with tarfile.open(local_tar, "w") as tf:
            for rel, _ in items:
                d = CROPS / rel.replace(".mp4", "")
                if d.exists():
                    tf.add(d, arcname=rel.replace(".mp4", ""))
        dst = V2 / "crops" / f"test__DFD_{tag}.tar"
        shutil.copy(local_tar, str(dst) + ".part")
        os.replace(str(dst) + ".part", dst)
        for rel, mask in items:
            d = CROPS / rel.replace(".mp4", "")
            fids = sorted(int(p.stem.split("_")[1]) for p in d.glob("*.png")) if d.exists() else []
            rows.append(dict(rel=rel, split="dfd", method="real" if label == 0 else "DeepFakeDetection",
                             label=label, n_crops=len(fids), frame_ids=" ".join(map(str, fids)),
                             n_frames=nframes.get(rel, -1),
                             count_from_demux=bool(fc[rel][1] or (mask is not None and fc[mask][1]))))
        cnt = [r["n_crops"] for r in rows if r["label"] == label]
        print(f"[4] DFD {tag}: videos={len(items)} crops={sum(cnt)} zero={cnt.count(0)} min={min(cnt)} "
              f"| {time.time() - t:.0f}s | tar {dst.stat().st_size / 1e6:.0f} MB", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(V2 / "manifest" / "crops_index_dfd.csv", index=False)
    (V2 / "step9").mkdir(exist_ok=True)
    (V2 / "step9" / "warnings.log").write_text("\n".join(warn))
    print(f"[5] crops per video: {df.n_crops.value_counts().sort_index().to_dict()} | warnings: {len(warn)}")
    for w in warn[:8]:
        print("     ", w)
    print(f"    total {(time.time() - T0) / 60:.1f} min | saved crops_index_dfd.csv")
    print("DFD: COMPLETE")


if __name__ == "__main__":
    main()
