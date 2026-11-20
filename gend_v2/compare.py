"""Pre-registered follow-up comparisons: paired bootstrap of ΔAUC between two models' video scores.

  python -m gend_v2.compare --eval_dirs results/eval_final results/eval_followup --out results/followup.md

Each comparison 'A vs B' uses the video scores written by evaluate.py (video_scores/<model>.csv), the same
test subsets and the same stratified paired bootstrap (2,000 resamples) as the main evaluation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# same functions as in evaluate.py, duplicated so this script needs no torch / GenD install
TEST_FF = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def auc_fast(y, p):
    r = stats.rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap(y, p_model, p_base, reps=2000, seed=0):
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(reps):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(auc_fast(y[i], p_model[i]) - auc_fast(y[i], p_base[i]))
    d = np.array(d)
    return dict(d_auc=float(auc_fast(y, p_model) - auc_fast(y, p_base)),
                d_ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))))


def subsets(test_vid, dfd_vid):
    real = test_vid[test_vid.y == 0]
    S = {"FF++ (4 seen)": test_vid[test_vid.method.isin(["real"] + TEST_FF)],
         "FaceShifter (unseen gen.)": pd.concat([real, test_vid[test_vid.method == "FaceShifter"]])}
    if dfd_vid is not None and len(dfd_vid):
        S["DFD (cross-dataset)"] = dfd_vid
    return S

SETS = ["FF++ (4 seen)", "FaceShifter (unseen gen.)", "DFD (cross-dataset)"]
PLANNED = ([(f"pdc_full_s{k}", f"baseline_full_s{k}") for k in (0, 1)]
           + [(f"pdcvrex_s{k}", f"baseline_s{k}") for k in (0, 1, 2)]
           + [(f"pdcvrex_s{k}", f"pdc_s{k}") for k in (0, 1, 2)]
           + [(f"baseline_full_s{k}", f"baseline_s{k}") for k in (0, 1)])


def load(dirs, name):
    for d in dirs:
        f = Path(d) / "video_scores" / f"{name}.csv"
        if f.exists():
            v = pd.read_csv(f)
            return v[v.set == "test"].drop(columns="set"), v[v.set == "dfd"].drop(columns="set")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    rows = []
    for A, B in PLANNED:
        va, vb = load(a.eval_dirs, A), load(a.eval_dirs, B)
        if va is None or vb is None:
            print(f"skip {A} vs {B} (missing scores)")
            continue
        SA, SB = subsets(*va), subsets(*vb)
        for s in SETS:
            m = SA[s].merge(SB[s][["video", "p"]], on="video", suffixes=("", "_b"))
            assert len(m) == len(SA[s]) == len(SB[s]), f"video sets differ for {A}/{B}/{s}"
            b = bootstrap(m.y.values, m.p.values, m.p_b.values, reps=a.reps)
            rows.append(dict(A=A, B=B, testset=s, auc_A=auc_fast(m.y.values, m.p.values),
                             auc_B=auc_fast(m.y.values, m.p_b.values), d_auc=b["d_auc"],
                             d_lo=b["d_ci"][0], d_hi=b["d_ci"][1]))
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    L = ["# Pre-registered follow-up comparisons (paired bootstrap, 95% CI)\n",
         "| A | B | test set | AUC A | AUC B | ΔAUC [95% CI] |", "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        L.append(f"| {r.A} | {r.B} | {r.testset} | {r.auc_A:.3f} | {r.auc_B:.3f} | {r.d_auc:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}] |")
    L.append("\n## Seed-level summary (mean Δ, seeds with CI>0 / CI<0, one-sample t-test over seeds)\n")
    L += ["| A family vs B family | test set | mean Δ | +/− | p |", "|---|---|---|---|---|"]
    if len(df):
        df["fam"] = df.A.str.replace(r"_s\d+$", "", regex=True) + " vs " + df.B.str.replace(r"_s\d+$", "", regex=True)
        for (fam, s), g in df.groupby(["fam", "testset"], sort=False):
            p = stats.ttest_1samp(g.d_auc, 0).pvalue if len(g) > 1 else float("nan")
            L.append(f"| {fam} | {s} | {g.d_auc.mean():+.3f} | +{int((g.d_lo > 0).sum())}/−{int((g.d_hi < 0).sum())} "
                     f"(n={len(g)}) | {p:.3f} |")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
