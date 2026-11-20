# Pre-registered follow-up comparisons (paired bootstrap, 95% CI)

| A | B | test set | AUC A | AUC B | ΔAUC [95% CI] |
|---|---|---|---|---|---|
| pdc_full_s0 | baseline_full_s0 | FF++ (4 seen) | 0.949 | 0.914 | +0.036 [+0.024,+0.048] |
| pdc_full_s0 | baseline_full_s0 | FaceShifter (unseen gen.) | 0.875 | 0.878 | -0.004 [-0.026,+0.018] |
| pdc_full_s0 | baseline_full_s0 | DFD (cross-dataset) | 0.970 | 0.970 | +0.000 [-0.012,+0.011] |
| pdc_full_s1 | baseline_full_s1 | FF++ (4 seen) | 0.896 | 0.881 | +0.015 [+0.005,+0.025] |
| pdc_full_s1 | baseline_full_s1 | FaceShifter (unseen gen.) | 0.886 | 0.869 | +0.017 [-0.002,+0.035] |
| pdc_full_s1 | baseline_full_s1 | DFD (cross-dataset) | 0.940 | 0.962 | -0.022 [-0.037,-0.009] |
| pdcvrex_s0 | baseline_s0 | FF++ (4 seen) | 0.925 | 0.866 | +0.059 [+0.043,+0.077] |
| pdcvrex_s0 | baseline_s0 | FaceShifter (unseen gen.) | 0.870 | 0.867 | +0.003 [-0.027,+0.033] |
| pdcvrex_s0 | baseline_s0 | DFD (cross-dataset) | 0.968 | 0.965 | +0.003 [-0.011,+0.014] |
| pdcvrex_s1 | baseline_s1 | FF++ (4 seen) | 0.922 | 0.849 | +0.073 [+0.055,+0.091] |
| pdcvrex_s1 | baseline_s1 | FaceShifter (unseen gen.) | 0.892 | 0.851 | +0.041 [+0.011,+0.073] |
| pdcvrex_s1 | baseline_s1 | DFD (cross-dataset) | 0.971 | 0.958 | +0.013 [+0.001,+0.026] |
| pdcvrex_s2 | baseline_s2 | FF++ (4 seen) | 0.918 | 0.885 | +0.034 [+0.023,+0.045] |
| pdcvrex_s2 | baseline_s2 | FaceShifter (unseen gen.) | 0.887 | 0.866 | +0.021 [+0.003,+0.041] |
| pdcvrex_s2 | baseline_s2 | DFD (cross-dataset) | 0.973 | 0.959 | +0.014 [+0.006,+0.024] |
| pdcvrex_s0 | pdc_s0 | FF++ (4 seen) | 0.925 | 0.886 | +0.039 [+0.027,+0.053] |
| pdcvrex_s0 | pdc_s0 | FaceShifter (unseen gen.) | 0.870 | 0.891 | -0.020 [-0.044,+0.002] |
| pdcvrex_s0 | pdc_s0 | DFD (cross-dataset) | 0.968 | 0.956 | +0.012 [+0.004,+0.021] |
| pdcvrex_s1 | pdc_s1 | FF++ (4 seen) | 0.922 | 0.877 | +0.045 [+0.032,+0.059] |
| pdcvrex_s1 | pdc_s1 | FaceShifter (unseen gen.) | 0.892 | 0.880 | +0.011 [-0.008,+0.033] |
| pdcvrex_s1 | pdc_s1 | DFD (cross-dataset) | 0.971 | 0.960 | +0.010 [+0.004,+0.018] |
| pdcvrex_s2 | pdc_s2 | FF++ (4 seen) | 0.918 | 0.909 | +0.009 [+0.002,+0.016] |
| pdcvrex_s2 | pdc_s2 | FaceShifter (unseen gen.) | 0.887 | 0.885 | +0.002 [-0.012,+0.015] |
| pdcvrex_s2 | pdc_s2 | DFD (cross-dataset) | 0.973 | 0.960 | +0.013 [+0.006,+0.023] |
| baseline_full_s0 | baseline_s0 | FF++ (4 seen) | 0.914 | 0.866 | +0.048 [+0.034,+0.062] |
| baseline_full_s0 | baseline_s0 | FaceShifter (unseen gen.) | 0.878 | 0.867 | +0.011 [-0.011,+0.034] |
| baseline_full_s0 | baseline_s0 | DFD (cross-dataset) | 0.970 | 0.965 | +0.005 [+0.000,+0.010] |
| baseline_full_s1 | baseline_s1 | FF++ (4 seen) | 0.881 | 0.849 | +0.032 [+0.021,+0.043] |
| baseline_full_s1 | baseline_s1 | FaceShifter (unseen gen.) | 0.869 | 0.851 | +0.019 [-0.002,+0.040] |
| baseline_full_s1 | baseline_s1 | DFD (cross-dataset) | 0.962 | 0.958 | +0.004 [-0.003,+0.013] |

## Seed-level summary (mean Δ, seeds with CI>0 / CI<0, one-sample t-test over seeds)

| A family vs B family | test set | mean Δ | +/− | p |
|---|---|---|---|---|
| pdc_full vs baseline_full | FF++ (4 seen) | +0.025 | +2/−0 (n=2) | 0.246 |
| pdc_full vs baseline_full | FaceShifter (unseen gen.) | +0.007 | +0/−0 (n=2) | 0.637 |
| pdc_full vs baseline_full | DFD (cross-dataset) | -0.011 | +0/−1 (n=2) | 0.511 |
| pdcvrex vs baseline | FF++ (4 seen) | +0.055 | +3/−0 (n=3) | 0.041 |
| pdcvrex vs baseline | FaceShifter (unseen gen.) | +0.022 | +2/−0 (n=3) | 0.184 |
| pdcvrex vs baseline | DFD (cross-dataset) | +0.010 | +2/−0 (n=3) | 0.115 |
| pdcvrex vs pdc | FF++ (4 seen) | +0.031 | +3/−0 (n=3) | 0.107 |
| pdcvrex vs pdc | FaceShifter (unseen gen.) | -0.002 | +0/−0 (n=3) | 0.815 |
| pdcvrex vs pdc | DFD (cross-dataset) | +0.012 | +3/−0 (n=3) | 0.006 |
| baseline_full vs baseline | FF++ (4 seen) | +0.040 | +2/−0 (n=2) | 0.123 |
| baseline_full vs baseline | FaceShifter (unseen gen.) | +0.015 | +0/−0 (n=2) | 0.152 |
| baseline_full vs baseline | DFD (cross-dataset) | +0.005 | +1/−0 (n=2) | 0.028 |
