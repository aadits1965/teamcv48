# Team CV48 — GenD + Pair-Difference Consistency + Risk Equalisation

DLCV project assessment (IIT Guwahati). We reproduce **GenD** at reduced scale with the authors' own code and
pre-processing, implement four candidate extensions under one controlled protocol, select one on validation, and test a
combination as an exploratory, pre-planned follow-up. Report: [`report/TeamCV48_26_DLCV_Project_Report.pdf`](report/TeamCV48_26_DLCV_Project_Report.pdf).

| Member | Roll No. |
|---|---|
| Aadit Siroya | 230103126 |
| Kshitiz Kumar Rai | 230103122 |
| Pankaj | 230108035 |
| Mohit Nagraj | 230104062 |
| Anubhav Jindal | 230101013 |

## Result in one table (video-level AUC, mean ± sd over 3 seeds)

| Method | FF++ (4 seen manipulations) | FaceShifter (generator unseen in training; used for validation) | DFD (unseen dataset; fully held out) |
|---|---|---|---|
| GenD reproduction (baseline) | 0.866 ± 0.018 | 0.861 ± 0.009 | 0.961 ± 0.004 |
| + PDC (selected on validation) | 0.891 ± 0.017 | **0.885 ± 0.005** | 0.959 ± 0.002 |
| **+ PDC + V-REx (proposed; exploratory follow-up)** | **0.922 ± 0.003** | 0.883 ± 0.011 | **0.971 ± 0.003** |
| Δ proposed − baseline (seeds with 95 % CI > 0; t-test over seeds) | **+0.055 (3/3, p = 0.041)** | +0.022 (2/3, p = 0.18) | +0.010 (2/3, p = 0.12) |

Same 104,450 trainable parameters and identical inference for all three rows (the changes are training losses only).
The FaceShifter and DFD gains are not significant across seeds; the combination was designed after the main study's
test results were seen (see §2), so these numbers are exploratory.
All numbers are produced by `gend_v2/summarize.py` and `gend_v2/compare.py` from the files in `results/`.

---

## 1. Original paper

**GenD — "Deepfake Detection that Generalizes Across Benchmarks"**, Yermakov, Čech, Matas, Fritz (WACV 2026),
[arXiv:2508.06248](https://arxiv.org/abs/2508.06248) (**v4**). Official code:
[yermandy/GenD](https://github.com/yermandy/GenD), pinned to **commit `387a422`**.

GenD fine-tunes only the LayerNorm parameters (≈0.03 %) of a frozen CLIP ViT-L/14, L2-normalises the CLS feature onto
a hypersphere and trains a linear head with `L_GenD = CE + 0.1·L_align + 0.5·L_unif`, on real/fake data paired at the
source-video level (so a fake and its real share background and scene; for face swaps the identity still differs).

**What GenD does not constrain (verified in the code):** the *direction* in which a fake's embedding differs from a
real one. `L_align` pulls same-label embeddings together and `L_unif` spreads all embeddings (both act on positions);
pairing only shapes the training set (`load_pairs` in `DeepfakeDataset` is unused, no loss sees which real frame a fake
was made from).

## 2. Our contribution

### Proposed method: GenD + PDC + V-REx

With `z` the L2-normalised CLS feature, a batch of `P` pairs (fake `f_i` made from the source video of real `r_i`),
`d_i = z_{f_i} − z_{r_i}` and `d̂_i = d_i/‖d_i‖`:

```
L = CE + 0.1·L_align + 0.5·L_unif                       (GenD, official code)
  + λ_p · (1/P) Σ_i (1 − <d̂_i, u>)                      (PDC, λ_p = 0.5)
  + λ_r(t) · Var_e[ R_e ]                                (V-REx, λ_r = 10 after a 20 % warm-up)
u ← normalise(β u + (1−β) normalise(mean_i d̂_i)),  stop-gradient, β = 0.9
R_e = cross-entropy over the pairs of manipulation e ∈ {Deepfakes, Face2Face, FaceSwap, NeuralTextures}
```

* **Pair-Difference Consistency (PDC)** — our addition; not part of GenD, and we did not find it in the works we checked
  (we cannot claim it is new to the literature as a whole). It asks all fake−real displacements, across all manipulation
  types, to share one direction on the hypersphere. For a linear head the pair margin is
  `‖w‖·‖d_i‖·cos∠(d_i, w)`. When displacements of different manipulations point in widely different directions, one
  head direction gives inconsistent margins across methods; aligning them to a shared `u` makes consistently positive
  margins easier to obtain (a sufficient regulariser, not a necessary condition for separability). An unseen generator
  benefits to the extent its displacement also points along `u`.
  Since `E_{i,j}[z_{f_i} − z_{r_j}] = μ_f − μ_r = E_i[d_i]`, the mean displacement is the same for matched and shuffled
  pairs; `u` averages *normalised* displacements, so this holds for `u` only approximately (pairing mainly changes the
  variance of `d`) — consistent with our shuffled-pair control.
* **Risk equalisation (V-REx**, Krueger et al., ICML 2021, PMLR 139:5815–5826) with the four FF++ manipulations as
  environments: penalises the variance of per-manipulation cross-entropy risk, encouraging consistent performance across
  manipulation types; complementary to PDC's shared direction.
* PDC's gradient reaches only the LayerNorm parameters (through `z`); V-REx also reaches the head.

| Component | Code |
|---|---|
| PDC loss | `gend_v2/losses.py::PDC` |
| V-REx penalty | `gend_v2/losses.py::vrex_penalty` |
| Combined objective (`--arm pdcvrex`) | `gend_v2/train.py` (branch `args.arm == "pdcvrex"`) |
| Official GenD model / loss (unchanged) | `gend_v2/model.py` (imports `src.model.GenD.GenD`), `gend_v2/losses.py::GenDLoss` |

### How we got there (selection was fixed before any test result)

1. **Main study:** four candidates, each changing one thing relative to the baseline, all with identical data, batches,
   and budget, 3 seeds each:

   | ID | Candidate | Code |
   |---|---|---|
   | M1 | PDC (above) | `losses.py::PDC` |
   | M2 | V-REx (above) | `losses.py::vrex_penalty` |
   | M3 | LayerNorm weight averaging (SWA / WiSE over baseline snapshots, variant chosen on validation) | `evaluate.py` |
   | M4 | Attention-pooled patch tokens, `h = CLS + γ·Σ_j softmax(q·p_j/√D)·p_j` (γ initialised to 0) | `model.py::AttnPatchPool` |

   The method with the best **validation** video AP was selected and written to `results/eval_final/selection.json`
   before any test prediction: **M1 PDC at all three seeds**.
2. **Exploratory follow-ups, planned in writing before they were run** ([`PREREGISTRATION.md`](PREREGISTRATION.md)):
   F1 scale-up (3× training data, baseline vs PDC, 2 seeds) and F2 the PDC + V-REx combination (3 seeds).
   **Caveat:** F2 was motivated by the main study's *test* results (PDC did not help on DFD, V-REx did) and is evaluated
   on the same test sets, so it is not an independent confirmation; a fresh held-out set would be needed for that.

## 3. Experimental protocol

| Item | Setting |
|---|---|
| Train (main) | FF++ c23, first 120 of 360 official train pairs: 240 originals + 240 fakes × {DF, F2F, FS, NT}; 8 frames/video (9,600 crops) |
| Train (scale-up F1) | all 360 official train pairs (28,800 crops) |
| Validation | FF++ official val split: 140 originals + 140 **FaceShifter** fakes (generator unseen in training) — checkpoint and method selection only. Because the same generator is used for selection, the FaceShifter *test* is not fully independent of selection; DFD is the only fully held-out test. |
| Test 1 — in-domain | FF++ official test split: 140 originals + 4 × 140 fakes (700 videos) |
| Test 2 — unseen generator | FF++ test split: 140 originals + 140 FaceShifter fakes (280 videos; same actors as FF++) |
| Test 3 — unseen dataset | Google/Jigsaw DeepFakeDetection (DFD) c23: seeded random 200 real + 200 fake videos |
| Faces | authors' `detector.py` with README flags (`-m at_least -n 8 --det_thres 0.1 -s 1.3 --target_size none`); DFD fakes use the official masks |
| Model | CLIP ViT-L/14 (224 px), trainable LayerNorms + linear head on L2-normalised features (104,450 parameters; M4 +1,025) |
| Optimiser | AdamW lr 3e-4, wd 0 (paper values), cosine to 1e-6, fp16 autocast, 24 pairs (48 images)/batch, 6 epochs (1,920 steps main; 5,760 scale-up) |
| Batches | `PairedStratifiedBatchSampler`, identical across arms (fingerprints: seed 0 `425b31dcfc25`, 1 `a5b33d8ba1a7`, 2 `df0b84af2080`; scale-up `ea1dd02ef4c3`, `1941e211b691`) |
| Checkpoint | best validation video AP (every 160 steps); V-REx and PDC+V-REx only after the penalty is active (step > 384) |
| Metric | video level (mean frame P(fake), as in the official code): AUC, AP, accuracy, balanced accuracy, F1 |
| Statistics | 3 seeds (2 for scale-up and shuffled control); paired bootstrap over videos (2,000 resamples, stratified) for ΔAUC per seed; one-sample t-test over seed deltas |

### Deviations from the paper (deliberate, for free-tier GPUs)

* ≈1/3 of FF++ training videos (main study), 8 instead of 32 frames/video, 6 epochs, batch 48, fp16 instead of bf16 —
  **a scaled-down reproduction; absolute numbers are not comparable to the paper's.** The released GenD checkpoint is
  evaluated on our test sets for reference.
* Learning-rate schedule: one cosine decay from 3e-4 to 1e-6 over the run, no warm-up. The paper (Sec. 3.1) uses a
  cyclic cosine schedule: each cycle a 1-epoch linear warm-up from 1e-5 to 3e-4, then cosine decay over 9 epochs to
  1e-5 (official config at `387a422`: `lr_scheduler="cyclic"`, 10 epochs per cycle). Batch 48 instead of the paper's 128.
* **8 frames per video at test time as well** (paper: 32), including when we evaluate the released GenD checkpoint, so
  its numbers here are not the paper's.
* Validation on FaceShifter instead of the paper's validation sets.
* V-REx (M2): its best checkpoint fell inside the 20 % warm-up (penalty never active), so it is evaluated at the best
  validated checkpoint after activation; this rule was set after seeing seed-0 logs (M2 is therefore exploratory). For
  PDC + V-REx the same rule was fixed in advance (`--select_after 384`).
* Hardware: every baseline, PDC, PDC + V-REx and shuffled-control run at main scale was trained on a Tesla T4; V-REx,
  patch, λ-sensitivity and scale-up runs partly on A100 / RTX PRO 6000 (same code, fp16 autocast, same batches).
  See `results/run_hardware.csv`.

## 4. Results

Full tables: [`results/summary.md`](results/summary.md) (main study) and [`results/followup.md`](results/followup.md)
(exploratory follow-ups). Raw outputs: `results/eval_final/`, `results/eval_followup/` (per-video scores, validation
tables, selection, diagnostics).

### Main study (3 seeds; ΔAUC vs same-seed baseline; seeds with 95 % CI > 0 / < 0)

| Method | FF++ (4 seen) | FaceShifter (unseen gen.) | DFD (unseen dataset) |
|---|---|---|---|
| Baseline (GenD, our run) | 0.866 ± 0.018 | 0.861 ± 0.009 | 0.961 ± 0.004 |
| **M1 PDC (selected)** | 0.891 (Δ +0.024; +3/−0; p = 0.008) | 0.885 (Δ +0.024; +3/−0; p = 0.014) | 0.959 (Δ −0.002; +0/−0) |
| M2 V-REx (exploratory) | 0.895 (Δ +0.028; +2/−0) | 0.869 (Δ +0.008; +0/−0) | 0.972 (Δ +0.012; +2/−0) |
| M3 LayerNorm SWA | 0.907 (Δ +0.041; +3/−0) | 0.844 (Δ −0.017; +0/−1) | 0.968 (Δ +0.008; +2/−0) |
| M4 patch pooling | 0.866 (Δ −0.000; +1/−1) | 0.857 (Δ −0.004; +0/−1) | 0.961 (Δ +0.001; +0/−1) |

### Exploratory follow-ups (planned before running; motivated by main-study test results)

| Comparison | FF++ | FaceShifter | DFD | Hypothesis |
|---|---|---|---|---|
| **PDC + V-REx − baseline** (3 seeds) | **+0.055 (+3/−0, p = 0.041)** | +0.022 (+2/−0, p = 0.18) | +0.010 (+2/−0, p = 0.12) | H2a: supported on FF++; 2/3 seeds on the others |
| PDC + V-REx − PDC (3 seeds) | +0.031 (+3/−0, p = 0.11) | −0.002 (+0/−0) | **+0.012 (+3/−0, p = 0.006)** | H2b: supported |
| PDC − baseline at 3× data (2 seeds) | +0.025 (+2/−0) | +0.007 (+0/−0) | −0.011 (+0/−1) | H1: FF++ yes, FaceShifter **no** |
| baseline at 3× data − baseline (2 seeds) | +0.040 (+2/−0) | +0.015 (+0/−0) | +0.005 (+1/−0) | — |

### Controls

| Control | FF++ | FaceShifter | DFD | Reading |
|---|---|---|---|---|
| PDC with **shuffled pairs** (seeds 0, 1) | 0.880 | 0.888 | 0.958 | similar FF++/FaceShifter gains to PDC → source-video correspondence seems not required (2 seeds only); seed 0 is significantly worse than baseline on DFD (Δ −0.012 [−0.025, −0.003]) |
| PDC λ = 0.1 / 0.5 / 2.0 (seed 0) | 0.875 / 0.886 / 0.867 | 0.878 / 0.891 / 0.873 | 0.970 / 0.956 / 0.950 | λ = 2 removes the gain (reported only, no tuning) |
| V-REx before activation = baseline re-run (seeds 0, 1) | Δ −0.003 / +0.003 | +0.002 / +0.006 | −0.004 / −0.002 | run-to-run noise ≈ 0.003 AUC |
| Released GenD checkpoint (full training) | 0.977 | 0.859 | 0.961 | reference only |

### Mechanism diagnostics (test pairs; mean over seeds)

| | cos∠(d, ū) on unseen FaceShifter (ū = mean test displacement of the 4 training manipulations, a proxy for the training u) | spread (sd) of per-manipulation AUC | NeuralTextures AUC |
|---|---|---|---|
| Baseline | 0.183 | 0.172 | 0.624 |
| PDC | 0.264 | 0.145 | 0.684 |
| PDC + V-REx | 0.227 | 0.097 | 0.784 |

### Interpretation and limitations

* PDC improves in-domain AUC (+0.024 at 1×, 3/3 seeds, p = 0.008; +0.025 at 3× data, CI > 0 at 2/2 seeds but p = 0.25
  with only 2 seeds). Its unseen-generator gain (+0.024 at 1×,
  3/3 seeds) **does not clearly survive 3× training data** (+0.007, not significant) — part of it reflects an
  under-trained baseline.
* Among baseline, PDC and PDC + V-REx, the combination has the highest mean AUC on FF++ and DFD (V-REx alone is
  marginally higher on DFD, 0.972 vs 0.971; PDC alone on FaceShifter, 0.885 vs 0.883). Its FaceShifter and DFD gains
  over the baseline are not significant across seeds (p = 0.18 and 0.12); its DFD gain over PDC is consistent
  (+0.012 at every seed, p = 0.006).
* Higher pair-direction alignment does not by itself predict AUC: PDC + V-REx has a lower cosine than PDC (0.227 vs
  0.264), and PDC with λ = 2 has the highest (0.326) but no AUC gain.
  The combination was not re-run at 3× data.
* All runs are scaled down; DFD is near ceiling (AUC ≈ 0.96) with 400 videos (AUC s.e. ≈ 1–1.6 points); validation and
  FaceShifter test share FF++ actors; checkpoint selection takes the best of 12 validation looks (same for all arms).

## 5. Reproducing everything

**What was run exactly as written here.** Every training run, the DFD build, the sanity checks, the evaluations,
`summarize.py` and `compare.py` were executed as these modules (in Colab, from Google Drive). Two differences from the
executed versions: (i) `prepare_ffpp.py` packages the FF++ manifest/download/pre-processing steps that we executed as
notebook cells (same logic and parameters), and (ii) paths were later centralised in `gend_v2/paths.py` and options
were added for the follow-ups (`--arm pdcvrex`, `--select_after`, `--data`, `--only_extra`); defaults are unchanged.
These two changes were not re-executed end-to-end before submission.

### Environment (as run)

Google Colab, Python 3.13, `torch 2.14.1+cu130` (22 of 23 runs; `baseline_s0` ran on `2.11.0+cu130`, recorded in each
`config.json`), torchvision = the matching Colab build (not recorded per run), `transformers 4.56.2`, `lightning 2.5.5`,
`onnxruntime-gpu 1.31.1`, CUDA 13.0 runtime. GPU: Tesla T4 16 GB (training uses ≈12 GB at batch 48) or A100.

**Effect of the torch version change.** Colab upgraded torch after `baseline_s0` was trained, so all seed-0 comparisons
mix torch 2.11.0 (baseline) and 2.14.1 (other arms). `vrex_s0` before its penalty switches on is the same procedure as
`baseline_s0` (same T4, same batch fingerprint `425b31dcfc25`, λ = 0, best validation step 320 in both) under torch
2.14.1. It differs from `baseline_s0` by −0.003 / +0.002 / −0.004 test AUC (FF++ / FaceShifter / DFD; row
"V-REx before activation" in the controls table), far below the effects we report.

```bash
git clone https://github.com/aadits1965/teamcv48.git && cd teamcv48
git clone https://github.com/yermandy/GenD.git /content/GenD && git -C /content/GenD checkout 387a422
pip install -r requirements.txt
export PYTHONPATH=$PWD
export GEND_OFFICIAL_DIR=/content/GenD                      # official repo
export GEND_V2_DIR=/content/drive/MyDrive/GenD_PCR_Project/v2   # persistent: manifests, crop archives, runs, eval outputs
export GEND_WORK_DIR=/content/data                          # scratch: raw videos, extracted crops
```

### Data (FaceForensics++ terms must be accepted first; no data is redistributed here)

Request access at <https://github.com/ondyari/FaceForensics>; the scripts use the official downloader
(`faceforensics_download_v4.py`). DFD is distributed with FF++.

```bash
python -m gend_v2.prepare_ffpp                 # manifest (official splits) + download (~4 GB) + authors' face pipeline (~70 min on T4)
python -m gend_v2.dfd --n_real 200 --n_fake 200 --seed 0   # DFD test subset (+ reader equivalence checks)
python -m gend_v2.prepare_full --workers 8     # only for the scale-up follow-up (remaining 240 train pairs)
```

Produces in `$GEND_V2_DIR`: `manifest/` (identical to `results/manifest/`), `crops/<split>__<method>.tar`.

### Training (one command per run; ≈50 min on a T4, ≈15 min on an A100; resumable with `--resume`)

```bash
R=$GEND_V2_DIR
# main study: baseline + 4 candidates, 3 seeds
for s in 0 1 2; do for a in baseline pdc vrex patch; do
  python -m gend_v2.train --arm $a --seed $s --pairs 24 --epochs 6 --out $R/runs/${a}_s$s --resume
done; done
# controls
for s in 0 1; do python -m gend_v2.ablate_pdc --seed $s --out $R/runs_ablation/pdc_shuffled_s$s; done
python -m gend_v2.train --arm pdc --seed 0 --pairs 24 --epochs 6 --lam 0.1 --out $R/runs_sensitivity/pdc_lam0.1_s0
python -m gend_v2.train --arm pdc --seed 0 --pairs 24 --epochs 6 --lam 2.0 --out $R/runs_sensitivity/pdc_lam2.0_s0
# follow-up F2: proposed method (PDC + V-REx)
for s in 0 1 2; do
  python -m gend_v2.train --arm pdcvrex --seed $s --pairs 24 --epochs 6 --select_after 384 --out $R/runs_followup/pdcvrex_s$s
done
# follow-up F1: scale-up
for s in 0 1; do for a in baseline pdc; do
  python -m gend_v2.train --arm $a --seed $s --pairs 24 --epochs 6 --data full --out $R/runs_followup/${a}_full_s$s
done; done
```

Pre-training checks (shapes, gradients, identical batches, memory/throughput): `python -m gend_v2.sanity --out $R/step6`.

### Evaluation (reproduces every reported number)

```bash
python -m gend_v2.evaluate --seeds 0 1 2 --workers 8 --out $R/eval_final \
  --extra_runs runs_sensitivity/pdc_lam0.1_s0 runs_sensitivity/pdc_lam2.0_s0 runs_ablation/pdc_shuffled_s0 runs_ablation/pdc_shuffled_s1
python -m gend_v2.evaluate --only_extra --no_official --seeds 0 1 2 --workers 8 --out $R/eval_followup \
  --extra_runs runs_followup/baseline_full_s0 runs_followup/pdc_full_s0 runs_followup/baseline_full_s1 \
               runs_followup/pdc_full_s1 runs_followup/pdcvrex_s0 runs_followup/pdcvrex_s1 runs_followup/pdcvrex_s2
python -m gend_v2.summarize --eval $R/eval_final --out results/summary.md
python -m gend_v2.compare --eval_dirs $R/eval_final $R/eval_followup --out results/followup.md
```

`summarize` and `compare` need only `pandas/numpy/scipy` and also run directly on the files shipped in `results/`
(e.g. `python -m gend_v2.compare --eval_dirs results/eval_final results/eval_followup --out /tmp/f.md`).

### Evaluating the shipped checkpoints without retraining

```bash
cp -r checkpoints/runs checkpoints/runs_sensitivity checkpoints/runs_ablation checkpoints/runs_followup $GEND_V2_DIR/
# after the data commands above, run the two evaluate commands
```

## 6. Checkpoints

`checkpoints/<group>/<run>/` contains `best.pt` (`{"trainable": state_dict, "step", "val"}` — only the trainable
LayerNorm and head tensors, ≈0.4 MB; the frozen CLIP weights come from `openai/clip-vit-large-patch14`), `config.json`
(all hyper-parameters, batch fingerprint, GPU), `log.csv` (training loss terms every 20 steps), `val.csv` (validation
every 160 steps) and `done.json`. Baseline runs include all weight snapshots (needed for M3); V-REx runs include the
post-activation snapshots used for evaluation. Load with:

```python
import torch
from gend_v2.model import Detector
det = Detector("baseline").cuda().eval()        # use "patch" for patch_s* runs
det.load_trainable_state(torch.load("checkpoints/runs_followup/pdcvrex_s0/best.pt")["trainable"])
```

## 7. Verification performed

* **Same pre-processing as the authors.** `gend_v2/fast_frames.py` only replaces the per-frame `iio.imread(index=k)`
  with one sequential decode; detection, face choice, alignment and saving are the authors' functions. Crops are
  **pixel-identical** to the authors' reader on FF++ videos of every manipulation type and on DFD real videos; for DFD
  fakes, frames and masks match the authors' per-frame read (5 videos × 8 frames) — `results/checks/`.
* **DFD masks lack frame-count metadata** (all 200): the authors' reader cannot open them; only then is the count taken
  by demuxing, which equals the decoded count on every checked file.
* **Pairing convention verified on pixels:** fake `a_b` belongs to real `a` (border distance ≈6–12 vs ≈64 to `b`).
  Exact same-frame pairs: DF 100 %, FS/NT 57 %, F2F 14 %; other pairs use the nearest saved frame of the same video.
* **Per-arm sanity:** parameter counts, gradients only on trainable tensors, finite losses, novelty terms active,
  M4 = baseline at initialisation (`results/checks/step6/sanity.json`).

## 8. Repository layout

```
gend_v2/            all code (training, losses, model wrapper, data, evaluation, analysis)
results/            eval outputs, summaries, manifests, verification checks, run_hardware.csv
checkpoints/        trained weights + logs for every reported run
report/             one-page report (PDF + LaTeX)
PREREGISTRATION.md  follow-up hypotheses written before running them (+ addendum)
```

An earlier pilot (pair-conditioned ranking loss on 640 centre-cropped full frames) is superseded by this protocol and
not used in any comparison.

## 9. Citation

```bibtex
@inproceedings{yermakov2026gend,
  title     = {Deepfake Detection that Generalizes Across Benchmarks},
  author    = {Yermakov, Andrii and {\v{C}}ech, Jan and Matas, Ji{\v{r}}{\'\i} and Fritz, Mario},
  booktitle = {WACV},
  year      = {2026}
}
```
