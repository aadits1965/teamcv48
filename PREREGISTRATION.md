# Pre-registered follow-up experiments

**Written:** 8 Nov 2026, ~16:45 IST — **after** the main study's test results were seen (`results/eval_final`),
**before** any follow-up run was started. Everything below is reported whatever the outcome. Neither experiment
changes the main study's frozen method selection (`results/eval_final/selection.json`: M1 PDC at all seeds).

## F1 — Scale-up check: does the PDC gain survive with 3× training data?

**Motivation.** The main study uses 120 of 360 FF++ train pairs; our baseline is clearly under-trained relative to the
released GenD checkpoint (NeuralTextures AUC 0.62 vs 0.95). A gain that only appears for an under-trained model would
be much less useful.

* Data: all 360 official FF++ train pairs (`--data full`: 720 originals + 4 × 720 fakes, 8 frames/video), same
  authors' face pipeline (`gend_v2/prepare_full.py`). Validation and test sets unchanged.
* Arms: `baseline` and `pdc` (λ = 0.5), seeds 0 and 1.
* Everything else identical to the main study: 6 epochs, 24 pairs/batch, AdamW 3e-4, cosine, fp16, checkpoint =
  best validation video AP (evaluated every 160 steps), same evaluation code.
* **H1:** `pdc_full − baseline_full` ΔAUC > 0 on FF++ (4 seen) and on FaceShifter at both seeds (paired bootstrap
  95 % CI). Also reported: `baseline_full − baseline` (effect of 3× data on the baseline) and all DFD numbers.
* Outputs: `runs_followup/{baseline,pdc}_full_s{0,1}`.

## F2 — Combination: PDC + V-REx

**Motivation.** In the main study PDC helped on FaceShifter but not DFD, while V-REx (exploratory) helped on DFD.
We test whether the two objectives are complementary.

* Objective: `L = L_GenD + 0.5 · L_PDC + 10 · Var_e[R_e]`, V-REx penalty off for the first 20 % of steps.
* Checkpoint selection: best validation video AP **among steps > 384** (after the V-REx penalty is active) — the
  rule applied post hoc to V-REx in the main study, now fixed in advance (`--select_after 384`).
* Data and protocol as the main study (`--data v2`), seeds 0, 1, 2.
* **H2a:** `pdcvrex − baseline` ΔAUC > 0 on all three test sets. **H2b:** `pdcvrex − pdc` ΔAUC > 0 on DFD.
* Output: `runs_followup/pdcvrex_s{0,1,2}`.

## Analysis (fixed now)

`python -m gend_v2.evaluate --only_extra --extra_runs <follow-up runs> --out eval_followup --seeds 0 1 2 --no_official`
then `python -m gend_v2.compare --eval_dirs eval_final eval_followup --out results/followup.md`
(pairs listed in `gend_v2/compare.py::PLANNED`). If a run cannot finish by 9 Nov 12:00 IST, the completed runs are
reported and the missing ones are stated as missing.

## Addendum (written 9 Nov 2026, after all follow-up runs finished)

* All seven follow-up runs were launched on 8 Nov with the settings above. `pdcvrex_s0` finished on 9 Nov ≈13:27 IST,
  ≈1.5 h after the 12:00 cut-off stated above; all others finished earlier. We report all seven, including `pdcvrex_s0`.
* Test results of the four scale-up runs and of `pdcvrex_s1` were seen on 8 Nov ≈21:48 IST, while `pdcvrex_s0` and
  `pdcvrex_s2` were still training. Those two runs used exactly the pre-registered command (no setting was changed after
  that point), and their checkpoints were chosen automatically on validation, so the early look could not affect them.
* Outcome: **H1** supported on FF++ (2/2 seeds), **not** supported on FaceShifter (0/2 significant). **H2a** supported on
  FF++ (3/3), FaceShifter and DFD at 2/3 seeds each. **H2b** supported (3/3 seeds, p = 0.006). See `results/followup.md`.
