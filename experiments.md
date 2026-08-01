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

## Milestone 6 (last headroom: NN member + competitor intelligence)

| # | Hypothesis | Change | CV | Δ | LB | Verdict |
|---|-----------|--------|----|---|----|---------|
| E6 | exact-value TE (the one verified public FE edge) lifts trees too | 8 exact-value cats → multiclass TE → LGBM | 0.94451 | −0.00443 | — | **hurts trees** — they split exact values natively; TE columns are noise to a booster. Public ablation was TE→NN, not TE→tree |
| NN | torch MLP w/ embeddings + weighted CE (side venv, Intel-mac torch 2.2.2) | src/train_nn.py | 0.94819 | −0.00075 | — | real member (sklearn MLP was 0.936) but blend-neutral alone (corr 0.87–0.93) |
| E7 | TE lifts NNs specifically (verified public combo) | TE cols into NN numeric block | 0.94831 | +0.00012 over NN | — | direction confirmed, magnitude needs true RealMLP (periodic embeddings, n_ens, EMA) |
| E8 | 5-way blend prices diversity vs quality | grid {hgb 0.5, nn_te 0.25, lgbm 0.1, cat 0.1, lgbm_te 0.05} + thr(1.15, 1.15) | **0.95009** | +0.00019 | **0.95006** | **new best; >0.95 target met.** Both "failed" models are in the final blend on decorrelation value |

**Competitor scan (6 top kernels):** target reverse-engineered as threshold rule (sleep<6h/stress/activity) + synthesis noise → honest ceiling ≈ 0.9506 single-model; missingness flags independently confirmed dead; class-weights + fixed prior correction over-correct (tuned search is safe); original external dataset a dead end; public LB >0.951 is shared-file blending with S6E6-style private-split risk.

**Final: public LB 0.95006 (CV 0.95009). Arc: 0.333 → 0.875 → 0.949 → 0.9497 → 0.9498 → 0.9501.**

## Milestone 7 (the queue from the competitor scan)

| # | Hypothesis | Change | CV | Δ | LB | Verdict |
|---|-----------|--------|----|---|----|---------|
| E9 | CatBoost's ordered target stats are the tree-safe form of exact-value TE | ev_* cats fed natively to CatBoost | 0.94982 | +0.00059 | — | **keep — best single model of the project.** Same features, third function class, third different outcome |
| E10 | true RealMLP reaches the published 0.9506 frontier | pytabkit RealMLP-TD, 6 epochs CPU, prior-corrected | 0.94731 | — | — | short of published (CPU budget); greedy blend weight 0 — quality floor law confirmed again |
| E11 | HGB (blend weight 0.5) rewards tuning | 9-config grid, winner max_leaf_nodes=23, l2=1 | 0.94963 | +0.00007 | — | below bar; third "defaults were fine." Kept in pool at zero cost |
| E12 | seed-averaging stabilizes the noisiest member | nn_te × seeds 42/1/2 | 0.94825 avg | — | — | steadier member for the blend |
| E13 | greedy selection prices 8 members better than a grid | Caruana w/ replacement, 80 rounds | **0.95036** | +0.00027 | **0.95048** | **final; public rank ~339/2192 at the time.** Thresholds converged to 1.0 — alignment migrated into the members |

**Final standing: public LB 0.95048 (CV 0.95036). Arc: 0.333 → 0.875 → 0.949 → 0.9497 → 0.9498 → 0.9501 → 0.9505.**

## Private leaderboard reveal (2026-08-01)

| Submission | CV | Public | Private |
|---|---|---|---|
| 02 lgbm baseline | 0.94894 | 0.94667 | 0.94704 |
| 03 features | 0.94916 | 0.94942 | 0.94949 |
| 04 final blend | 0.94968 | 0.94946 | 0.94965 |
| 04 robust | 0.94961 | 0.94918 | 0.94964 |
| 05 stack | 0.94990 | 0.94978 | 0.95000 |
| 06 nn blend | 0.95009 | 0.95006 | 0.95002 |
| 07 grand blend | 0.95036 | 0.95048 | **0.95040** |

**Final: private LB 0.95040 — rank 209 of 3356 (top 6.2%), up 386 places from public rank 595.** CV predicted the private score within 0.00004. The shakeup the competitor scan warned about (shared-file public blends carrying S6E6-style private-split risk) happened: the public blends above us fell, the CV-anchored entry rose.
