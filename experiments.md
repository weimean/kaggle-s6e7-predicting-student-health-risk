# Experiment Log

CV = 5-fold stratified OOF **balanced accuracy**, seed 42, identical fold assignment throughout (`artifacts/fold_id.npy`).
Model for all Milestone-3 experiments: LightGBM `class_weight='balanced'`, lr 0.05, ≤1000 trees, early stop 50 (= notebook 02 baseline).
Keep-rule: an experiment is kept only if ΔCV ≥ +0.0005 (≈ OOF-level standard error measured in notebook 02).

| # | Hypothesis | Change | CV | Δ vs baseline | LB | Verdict |
|---|-----------|--------|----|---------------|----|---------|
| 0 | (baseline) | LGBM weighted, native cat, NaN passthrough | 0.94894 | — | 0.94667 | reference |
| E1 | bmi-missingness is signal (EDA §5); explicit flags give the tree a clean handle | +13 per-column missing flags, +row missing count | 0.94912 | +0.00018 | — | drop (below bar; native NaN routing already captures it) |
| E2a | median/mode imputation destroys missingness signal | impute all NaNs, no flags | 0.90776 | −0.04118 | — | drop (confirmed: catastrophic; imputation is an information-destroying assertion) |
| E2b | flags rescue imputation by preserving the signal explicitly | impute all NaNs + E1 flags | 0.94900 | +0.00006 | — | drop (rescued to baseline, no gain; pairs-with-flags rule noted for NN/linear models) |
| E3 | ordinal encoding of 3 ordered cats lets one threshold express "≥ medium" | stress/sleep_quality/activity → ordered ints | 0.94895 | +0.00001 | — | drop (confirmed-null: 3-level cats fully covered by native handling) |
| E4 | explicit interaction axes shortcut tree-approximated conjunctions (EDA §4) | 5 ratio/product features (stress×sleep, activity×steps, …) | 0.94916 | +0.00022 | 0.94942 | drop (best of ladder, still below bar; booster already approximates) |

**Milestone 3 conclusion:** zero keepers — baseline features carry forward. Signal is fully exposed to the GBM; remaining gains must come from the model side (trees cap, tuning, per-class thresholds, ensembling). Note: CV↔LB gap flipped sign (+0.0023 → −0.0003), so treat the gap as noise ≈ ±0.003, decide on CV only.
