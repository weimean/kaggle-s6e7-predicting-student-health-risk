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

## Milestone 4 (model-side levers; CV numbers are thresholded where noted)

| # | Hypothesis | Change | CV | Δ | LB | Verdict |
|---|-----------|--------|----|---|----|---------|
| T1 | defaults undertrained + suboptimal; Optuna fold-0 proxy finds better | 25 TPE trials, cap 2000 | 0.94821 | −0.00073 | — | proxy overfit caught by full-CV re-verify; defaults ≈ optimal (num_leaves 32 ≈ default 31); keep for probs (see T2) |
| T2 | argmax is not the bal.-acc.-optimal decision | per-class multipliers fit on OOF (fit×2.04, unh×1.80) | 0.94961 | +0.00067 vs base | — | **keep — largest gain since class weights.** Tuned model + thresholds beats baseline + thresholds (0.94943): compare models under the deployed decision rule |
| T3 | seed averaging cancels run variance | 3 seeds, avg probs | 0.94831 (argmax) | +0.0001 | — | marginal; bagging already decorrelates |
| T4 | XGBoost adds decorrelated errors | hand-picked params, same folds | 0.93486 | −0.014 | — | too weak to blend (weight 0); diversity present (corr 0.81) but base quality absent |
| T5 | CatBoost's ordered target stats differ enough to blend | same folds, auto balanced weights | 0.94923 | +0.00029 | — | best single family; blend 0.3 LGBM/0.7 Cat |
| T6 | blend + re-fit thresholds | weights + multipliers on OOF | **0.94968** | +0.00074 | **0.94946** | **final A** |
| T7 | robustness hedge | seed-avg LGBM + thresholds | 0.94961 | — | 0.94918 | final B |

**Milestone 4 conclusion:** score arc 0.333 → 0.875 (GBM) → 0.949 (loss↔metric alignment) → 0.9497 (decision-rule alignment + blend). Nearly all points came from alignment, not sophistication. Final public LB 0.94946 (top ≈ 0.953).

## Milestone 5 (extension: diversity + stacking)

| # | Hypothesis | Change | CV | Δ | LB | Verdict |
|---|-----------|--------|----|---|----|---------|
| E5a | a third GBM implementation decorrelates cheaply | HistGradientBoosting, near-default, balanced weights | 0.94956 | +0.00062 | — | **best single model in project**; implementation diversity > parameter search |
| E5b | an NN's smooth boundaries add diversity | sklearn MLP, impute+flags+one-hot+scale, majority undersampled, thresholds | 0.94465 | −0.00429 | — | most diverse (corr 0.78) but below quality floor; blend weight 0. Largest threshold gain (+0.0085) = plain-loss mis-centering |
| E5c | 4-way blend | grid weights {lgbm 0.1, cat 0.2, hgb 0.7, mlp 0} + thresholds | **0.94990** | +0.00096 | **0.94978** | **new best; submitted** |
| E5d | learned combiner beats fixed weights | LR stack on 12 OOF prob columns, balanced | 0.94985 | +0.00091 | — | statistical tie with blend; stacking needs many heterogeneous members to pay |

**Final standing: public LB 0.94978 (CV 0.94990), top ≈ 0.953. Score arc: 0.333 → 0.875 → 0.949 → 0.9497 → 0.9498.**
