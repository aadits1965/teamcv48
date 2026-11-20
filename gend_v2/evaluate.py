"""Final evaluation of all arms (run ONCE, after training).

  python -m gend_v2.evaluate --seeds 0 --out /content/drive/MyDrive/GenD_PCR_Project/v2/eval_s0

Order of operations (pre-registered, see README):
  1. Validation (FaceShifter val split) for every trained arm (best.pt) and for the M3 candidates built
     from the BASELINE run's weight snapshots (no extra training):
        swa_all  = mean of all snapshots (every 80 steps)
        swa_late = mean of snapshots from epoch 3 on (step >= 640)
        wise_0.5 / wise_0.75 = LayerNorm params (1-a)*pretrained + a*best ; head from best
     M3 = best candidate on validation video AP (tie-break AUC).
  2. The final proposed method = arm with the best validation video AP among {pdc, vrex, patch, m3}.
     Written to selection.json BEFORE any test prediction is made.
  3. Test: FF++ c23 test (4 seen methods), FaceShifter test (unseen generator, same actors),
     DFD subset (cross-dataset), for every arm + the authors' released checkpoint (yermandy/GenD_CLIP_L_14).
  4. Video-level metrics (mean frame P(fake), as in the official code) + frame-level AUC,
     paired bootstrap (videos resampled within class) for AUC differences vs the baseline,
     mechanism diagnostics (PDC direction consistency, per-method AUC spread, M4 gamma).
"""
import argparse
import gc
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from . import data as D
from .evalutil import binary_metrics
from .model import Detector

ARMS = ["baseline", "pdc", "vrex", "patch"]


def seed_of(name, default=0):
    m = re.search(r"_s(\d+)", name)
    return int(m.group(1)) if m else default
TEST_FF = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# ----------------------------------------------------------------------------- data
def ensure_dfd(v2: Path):
    idx_path = v2 / "manifest" / "crops_index_dfd.csv"
    if not idx_path.exists():
        return None
    idx = pd.read_csv(idx_path, dtype={"frame_ids": str})
    for tag in ["real", "fake"]:
        rels = idx[idx.label == (0 if tag == "real" else 1)].rel
        if not all((D.LOCAL_CROPS / r.replace(".mp4", "")).exists() for r in rels):
            with tarfile.open(v2 / "crops" / f"test__DFD_{tag}.tar") as tf:
                tf.extractall(D.LOCAL_CROPS)
    idx["frame_ids"] = idx.frame_ids.fillna("")
    return idx


def tables(v2: Path):
    idx = D.load_index(v2)
    D.ensure_crops(["val__real", "val__FaceShifter", "test__real", "test__FaceShifter"] +
                   [f"test__{m}" for m in TEST_FF], idx=idx)
    T = {"val": D.frame_table(idx, "val"), "test": D.frame_table(idx, "test")}
    dfd = ensure_dfd(v2)
    if dfd is not None:
        T["dfd"] = D.frame_table(dfd, "dfd")
    return T


# ----------------------------------------------------------------------------- inference
WORKERS = 2


@torch.no_grad()
def run_model(feat_fn, head_fn, table, preprocess, bs=64, workers=None):
    workers = WORKERS if workers is None else workers
    ds = D.CropDataset(table, preprocess, augment=None)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    P, Z = [], []
    for x, _, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            f = feat_fn(x.cuda(non_blocking=True))
        logits = head_fn(f.float())
        P.append(logits.float().softmax(1)[:, 1].cpu())
        Z.append(F.normalize(f.float(), dim=1).half().cpu())
    t = table.copy()
    t["p_fake"] = torch.cat(P).numpy()
    return t, torch.cat(Z)


def det_fns(det):
    det.eval()
    return det.features, (lambda f: det.head(f).logits_labels)


def vids(t):
    return (t.groupby("video", sort=True)
             .agg(p=("p_fake", "mean"), y=("label", "first"), method=("method", "first"), n=("p_fake", "size"))
             .reset_index())


# ----------------------------------------------------------------------------- metrics
def auc_fast(y, p):
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap(y, p_model, p_base, reps=2000, seed=0):
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    a_m, a_b = [], []
    for _ in range(reps):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        a_m.append(auc_fast(y[i], p_model[i]))
        a_b.append(auc_fast(y[i], p_base[i]))
    a_m, a_b = np.array(a_m), np.array(a_b)
    d = a_m - a_b
    return dict(auc_ci=(float(np.percentile(a_m, 2.5)), float(np.percentile(a_m, 97.5))),
                d_auc=float(auc_fast(y, p_model) - auc_fast(y, p_base)),
                d_ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                p_le0=float((d <= 0).mean()))


def subsets(test_vid, dfd_vid):
    """Named evaluation sets (video level)."""
    real = test_vid[test_vid.y == 0]
    S = {"FF++ (4 seen)": test_vid[test_vid.method.isin(["real"] + TEST_FF)],
         "FaceShifter (unseen gen.)": pd.concat([real, test_vid[test_vid.method == "FaceShifter"]])}
    for m in TEST_FF:
        S[f"FF++/{m}"] = pd.concat([real, test_vid[test_vid.method == m]])
    if dfd_vid is not None:
        S["DFD (cross-dataset)"] = dfd_vid
    return S


# ----------------------------------------------------------------------------- PDC diagnostic
def pair_direction_consistency(test_tab, Z):
    """cos(d_i, u_seen), d_i = z(fake frame) - z(real frame of same target, nearest saved frame);
    u_seen = mean normalised d over the 4 seen FF++ methods (test split)."""
    t = test_tab.reset_index(drop=True)
    real = t[t.label == 0]
    rmap = {}
    for i, tg, fr in zip(real.index, real.target, real.frame):
        rmap.setdefault(tg, {})[int(fr)] = i
    rows = []
    for i, tg, fr, m in zip(t.index, t.target, t.frame, t.method):
        if m == "real" or tg not in rmap:
            continue
        cand = rmap[tg]
        j = cand[min(cand, key=lambda c: (abs(c - fr), c))]
        rows.append((i, j, m))
    fi = torch.tensor([r[0] for r in rows])
    ri = torch.tensor([r[1] for r in rows])
    meth = np.array([r[2] for r in rows])
    d = F.normalize(Z[fi].float() - Z[ri].float(), dim=1)
    seen = torch.from_numpy(np.isin(meth, TEST_FF))
    u = F.normalize(d[seen].mean(0), dim=0)
    cos = (d @ u).numpy()
    return {m: float(cos[meth == m].mean()) for m in TEST_FF + ["FaceShifter"]}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_official", action="store_true")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only_extra", action="store_true",
                    help="evaluate only --extra_runs (follow-up experiments); no M3, no selection, no main arms")
    ap.add_argument("--extra_runs", nargs="*", default=[],
                    help="run dirs relative to v2/ (e.g. runs_ablation/pdc_shuffled_s0); evaluated AFTER selection, "
                         "never eligible for selection; non-patch arms only")
    args = ap.parse_args()
    global WORKERS
    WORKERS = args.workers
    v2 = D.DRIVE_V2
    out = Path(args.out)
    (out / "video_scores").mkdir(parents=True, exist_ok=True)
    T = tables(v2)
    print("[0] frames:", {k: len(v) for k, v in T.items()},
          "| videos:", {k: v.video.nunique() for k, v in T.items()}, flush=True)

    val_rows, models = [], {}          # models[name] = (arm, trainable_state)
    for seed in ([] if args.only_extra else args.seeds):
        for arm in ARMS:
            ck = v2 / "runs" / f"{arm}_s{seed}" / "best.pt"
            if not ck.exists():
                print(f"   missing {ck} -> skipped")
                continue
            c = torch.load(ck, map_location="cpu", weights_only=False)
            print(f"   {arm}_s{seed}: best step {c['step']} (logged val AP {c['val']['ap']:.4f})")
            if arm == "vrex":
                # V-REx penalty is OFF for the first vrex_warmup*total_steps steps. A best checkpoint from inside
                # the warm-up never trained with the penalty (it is a re-run of the baseline). Disclosed deviation:
                # select V-REx among validated checkpoints AFTER penalty activation; keep the pre-activation
                # checkpoint as a run-to-run noise control ("vrex_s*_preact").
                rdir = ck.parent
                cfg = json.loads((rdir / "config.json").read_text())
                act = int(cfg["vrex_warmup"] * cfg["total_steps"])
                if c["step"] <= act:
                    models[f"vrex_s{seed}_preact"] = ("baseline", c["trainable"])
                    vc = pd.read_csv(rdir / "val.csv")
                    r = vc[vc.step > act].sort_values(["ap", "auc"], ascending=False).iloc[0]
                    st = torch.load(rdir / "snapshots" / f"step_{int(r.step):05d}.pt", map_location="cpu")
                    models[f"vrex_s{seed}"] = ("vrex", st)
                    print(f"   vrex_s{seed}: best step {c['step']} <= activation {act} -> using post-activation "
                          f"step {int(r.step)} (logged val AP {r.ap:.4f}); step {c['step']} kept as vrex_s{seed}_preact")
                    continue
            models[f"{arm}_s{seed}"] = (arm, c["trainable"])

    # ---------- [1] validation for arms + M3 candidates
    base_det = Detector("baseline", seed=0).cuda().eval()
    pre = D.make_preprocess(base_det.clip_processor)
    theta0 = base_det.trainable_state()
    is_ln = lambda n: not n.startswith("gend.model.")   # head params live under gend.model.*

    for seed in ([] if args.only_extra else args.seeds):
        bdir = v2 / "runs" / f"baseline_s{seed}"
        if not (bdir / "best.pt").exists():
            continue
        snaps = sorted((bdir / "snapshots").glob("step_*.pt"))
        S = {int(p.stem.split("_")[1]): torch.load(p, map_location="cpu") for p in snaps}
        best = models[f"baseline_s{seed}"][1]
        avg = lambda steps: {k: torch.stack([S[s][k] for s in steps]).mean(0) for k in best}
        cands = {"swa_all": avg(sorted(S)), "swa_late": avg([s for s in S if s >= 640])}
        for a in (0.5, 0.75):
            cands[f"wise_{a}"] = {k: ((1 - a) * theta0[k] + a * best[k]) if is_ln(k) else best[k] for k in best}
        print(f"[1] M3 candidates from {len(S)} baseline snapshots (seed {seed}):", list(cands))
        for name, st in cands.items():
            base_det.load_trainable_state(st)
            t, _ = run_model(*det_fns(base_det), T["val"], pre)
            m = binary_metrics(vids(t).y, vids(t).p)
            val_rows.append(dict(model=f"m3:{name}_s{seed}", seed=seed, **m))
            print(f"    m3:{name:9s} val AUC {m['auc']:.4f} AP {m['ap']:.4f}", flush=True)
        m3 = [r for r in val_rows if r["model"].startswith("m3:") and r["seed"] == seed]
        pick = max(m3, key=lambda r: (r["ap"], r["auc"]))
        models[f"m3_s{seed}"] = ("baseline", cands[pick["model"][3:].rsplit("_s", 1)[0]])
        val_rows.append(dict(pick, model=f"m3_s{seed}", m3_choice=pick["model"]))

    for name, (arm, st) in models.items():
        if name.startswith("m3_"):
            continue
        det = base_det if arm != "patch" else Detector("patch", seed=0).cuda().eval()
        det.load_trainable_state(st)
        t, _ = run_model(*det_fns(det), T["val"], pre)
        m = binary_metrics(vids(t).y, vids(t).p)
        val_rows.append(dict(model=name, seed=seed_of(name), **m))
        print(f"    {name:12s} val AUC {m['auc']:.4f} AP {m['ap']:.4f}", flush=True)
        if det is not base_det:
            del det
            gc.collect()
            torch.cuda.empty_cache()
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(out / "val_all.csv", index=False)

    # ---------- [2] selection (before any test prediction)
    sel = {}
    for seed in (args.seeds if len(val_df) else []):
        pool = val_df[(val_df.seed == seed) & val_df.model.isin([f"{a}_s{seed}" for a in ["pdc", "vrex", "patch", "m3"]])]
        if len(pool):
            r = pool.sort_values(["ap", "auc"], ascending=False).iloc[0]
            sel[f"seed{seed}"] = dict(selected=r.model, val_ap=float(r.ap), val_auc=float(r.auc))
    (out / "selection.json").write_text(json.dumps(sel, indent=2))
    print("[2] SELECTED ON VALIDATION (frozen before test):", sel, flush=True)

    # ---------- [2b] extra runs (sensitivity / ablation): reported only, added after selection
    extra_val = []
    for rel in args.extra_runs:
        ck = v2 / rel / "best.pt"
        if not ck.exists():
            print(f"   extra run missing, skipped: {rel}")
            continue
        c = torch.load(ck, map_location="cpu", weights_only=False)
        name = Path(rel).name
        models[name] = ("baseline", c["trainable"])
        base_det.load_trainable_state(c["trainable"])
        t, _ = run_model(*det_fns(base_det), T["val"], pre)
        m = binary_metrics(vids(t).y, vids(t).p)
        extra_val.append(dict(model=name, seed=seed_of(name), best_step=c["step"], **m))
        print(f"    extra {name:22s} best step {c['step']:4d} val AUC {m['auc']:.4f} AP {m['ap']:.4f}", flush=True)
    if extra_val:
        val_df = pd.concat([val_df, pd.DataFrame(extra_val)], ignore_index=True)
        val_df.to_csv(out / "val_all.csv", index=False)

    # ---------- [3] test predictions
    VS, diag = {}, {}

    def evaluate(name, feat_fn, head_fn, preprocess):
        t, Z = run_model(feat_fn, head_fn, T["test"], preprocess)
        tv = vids(t)
        frame_auc = {"test_ff": float(auc_fast(t[t.method.isin(["real"] + TEST_FF)].label.values,
                                               t[t.method.isin(["real"] + TEST_FF)].p_fake.values))}
        dv = None
        if "dfd" in T:
            td, _ = run_model(feat_fn, head_fn, T["dfd"], preprocess)
            dv = vids(td)
            frame_auc["dfd"] = float(auc_fast(td.label.values, td.p_fake.values))
        VS[name] = (tv, dv)
        pd.concat([tv.assign(set="test"), dv.assign(set="dfd") if dv is not None else None]).to_csv(
            out / "video_scores" / f"{name}.csv", index=False)
        diag[name] = dict(pair_direction_cos=pair_direction_consistency(T["test"], Z), frame_auc=frame_auc)
        print(f"   tested {name}", flush=True)

    for name, (arm, st) in models.items():
        det = base_det if arm != "patch" else Detector("patch", seed=0).cuda().eval()
        det.load_trainable_state(st)
        evaluate(name, *det_fns(det), pre)
        if arm == "patch":
            diag[name]["gamma"] = float(det.pool.gamma.item())
            del det
            gc.collect()
            torch.cuda.empty_cache()

    if not args.no_official:
        del base_det
        gc.collect()
        torch.cuda.empty_cache()
        from src.hf.modeling_gend import GenD as OfficialGenD
        off = OfficialGenD.from_pretrained("yermandy/GenD_CLIP_L_14").cuda().eval()
        evaluate("official_GenD_CLIP_L14", off.feature_extractor, off.model,
                 lambda im: off.feature_extractor.preprocess(im))

    # ---------- [4] metrics + paired bootstrap vs baseline of the same seed
    rows = []
    for name, (tv, dv) in VS.items():
        seed = seed_of(name, args.seeds[0])
        base = VS.get(f"baseline_s{seed}")
        S_m, S_b = subsets(tv, dv), (subsets(*base) if base else None)
        for sname, sv in S_m.items():
            m = binary_metrics(sv.y.values, sv.p.values)
            r = dict(model=name, testset=sname, **m)
            if S_b is not None and name != f"baseline_s{seed}":
                bv = S_b[sname]
                mm = sv.merge(bv[["video", "p"]], on="video", suffixes=("", "_b"))
                assert len(mm) == len(sv), "video sets differ between model and baseline"
                r.update(bootstrap(mm.y.values, mm.p.values, mm.p_b.values, reps=args.reps))
            else:
                y, p = sv.y.values, sv.p.values
                r.update(bootstrap(y, p, p, reps=args.reps))
                r.update(d_auc=0.0, d_ci=(0.0, 0.0), p_le0=np.nan)
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "results_test.csv", index=False)
    for name in diag:
        sub = res[(res.model == name) & res.testset.str.startswith("FF++/")]
        diag[name]["ffpp_per_method_auc_std"] = float(sub.auc.std())
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))

    # ---------- [5] summary
    pd.set_option("display.width", 200)
    main_sets = ["FF++ (4 seen)", "FaceShifter (unseen gen.)", "DFD (cross-dataset)"]
    print("\n[5] VIDEO-LEVEL AUC [95% CI] and dAUC vs baseline [95% CI]")
    for s in main_sets:
        sub = res[res.testset == s]
        if not len(sub):
            continue
        print(f"\n  == {s}  (n videos = {int(sub.n.iloc[0])})")
        for r in sub.itertuples():
            print(f"   {r.model:24s} AUC {r.auc:.4f} [{r.auc_ci[0]:.3f},{r.auc_ci[1]:.3f}]  AP {r.ap:.4f}  "
                  f"bAcc {r.bacc:.3f}  dAUC {r.d_auc:+.4f} [{r.d_ci[0]:+.3f},{r.d_ci[1]:+.3f}]")
    print("\n  FF++ per-method AUC:")
    print(res[res.testset.str.startswith("FF++/")].pivot(index="model", columns="testset", values="auc").round(4).to_string())
    print("\n  validation (FaceShifter val):")
    print(val_df[["model", "auc", "ap", "bacc"]].round(4).to_string(index=False))
    print("\n  diagnostics:")
    for k, v in diag.items():
        print(f"   {k:24s} pair-dir cos {json.dumps({m: round(c, 3) for m, c in v['pair_direction_cos'].items()})} "
              f"| FF++ per-method AUC std {v['ffpp_per_method_auc_std']:.4f}" + (f" | gamma {v['gamma']:.4f}" if "gamma" in v else ""))
    print("\nsaved to", out)
    print("EVALUATION: COMPLETE")


if __name__ == "__main__":
    main()
