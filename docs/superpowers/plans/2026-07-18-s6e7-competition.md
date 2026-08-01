# S6E7 Explain-as-We-Build Competition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a top-scoring (>0.95 accuracy) solution to Kaggle Playground S6E7 through four annotated notebooks that teach EDA, CV strategy, feature engineering, and gradient boosting to an intermediate practitioner.

**Architecture:** Four sequential Jupyter notebooks, each ending in a real Kaggle submission. Notebook 02 establishes the CV harness that all later experiments reuse. From notebook 03 onward, every modeling idea is logged in `experiments.md` as hypothesis → change → CV score → keep/drop.

**Tech Stack:** Python 3.13, uv, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, matplotlib, Jupyter, Kaggle CLI.

## Global Constraints

- Every notebook is written to teach: each modeling step carries markdown explaining *why*, what the alternatives were, and how to choose. This is a spec requirement, not decoration.
- Metric is **balanced accuracy** (mean per-class recall) on hard labels; classes `at-risk` (86%), `unhealthy` (8.4%), `fit` (5.8%). Majority floor = ⅓ (verified by probe submission); target > 0.95. All CV scoring uses `sklearn.metrics.balanced_accuracy_score`.
- All CV uses the same fixed seed (42) and the same fold assignment so experiment deltas are comparable.
- `data/` and `submissions/` are gitignored; notebooks and `experiments.md` are committed.
- Notebooks must run top-to-bottom cleanly before commit (`jupyter nbconvert --execute`).
- Commit messages: plain, no AI attribution (user rule).
- Submissions via `kaggle competitions submit playground-series-s6e7 -f <file> -m "<message>"`; record the public LB score in the notebook and `experiments.md` after each.

---

### Task 1: Environment setup

**Files:**
- Create: `pyproject.toml`, `.python-version` (via `uv init`)
- Modify: `.gitignore` (already has `.venv/`, `data/`, `submissions/`)

**Interfaces:**
- Produces: a `.venv` that all notebooks use as their Jupyter kernel (`uv run jupyter ...`).

- [ ] **Step 1: Initialize uv project and add dependencies**

```bash
cd ~/kaggle/playground-series-s6e7
uv init --bare --python 3.13
uv add pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn optuna jupyter ipykernel
```

- [ ] **Step 2: Verify all imports work**

Run: `uv run python -c "import pandas, numpy, sklearn, lightgbm, xgboost, catboost, matplotlib, seaborn, optuna; print('all imports OK')"`
Expected: `all imports OK`

- [ ] **Step 3: Verify data files exist**

Run: `ls data/train.csv data/test.csv data/sample_submission.csv`
Expected: all three listed (downloaded during brainstorming).

- [ ] **Step 4: Create directories and commit**

```bash
mkdir -p notebooks submissions
git add pyproject.toml uv.lock .python-version .gitignore
git commit -m "Set up uv environment with ML dependencies"
```

---

### Task 2: Notebook 01 — EDA + naive baseline (Milestone 1)

**Files:**
- Create: `notebooks/01-eda.ipynb`

**Interfaces:**
- Produces: documented EDA findings that notebook 02/03 cite when justifying choices; first baseline LB score as the metric floor.

- [ ] **Step 1: Write the EDA notebook** with these sections, each with a markdown cell explaining what we're looking for and why before the code:
  1. **Load & first look** — dtypes, shapes, `id` handling. Teach: why we set `id` aside.
  2. **Target distribution** — bar chart, exact proportions. Teach: what 86/8.4/5.8 imbalance means for an *accuracy* metric (a model can ignore `fit` almost entirely and still score well; per-class errors matter unevenly).
  3. **Numeric features by class** — histograms/KDEs of the 7 numeric features split by target. Teach: we're hunting for separability, not pretty plots.
  4. **Categorical features by class** — normalized crosstabs of the 6 categoricals vs target. Teach: reading conditional distributions.
  5. **Missingness analysis** — per-column missing rates (bar chart), missingness correlation matrix, and the key question: *is missingness informative?* Compute target distribution conditional on each column being missing vs present. Teach: MCAR/MAR/MNAR distinction in practical terms.
  6. **Train vs test drift check** — compare feature distributions train vs test (summary stats + a few overlaid histograms). Teach: why we check (CV trust depends on train≈test).
  7. **Duplicates & synthetic-data quirks** — duplicate rows, suspicious value patterns. Teach: Playground data is synthetic (generated from a real dataset); quirks are common and exploitable.
  8. **EDA conclusions** — a markdown summary listing concrete, falsifiable takeaways that later notebooks will act on.

- [ ] **Step 2: Add the naive baseline section** — predict `at-risk` for every test row, write `submissions/01-majority.csv` matching `sample_submission.csv` format (`id,health_condition`).

- [ ] **Step 3: Execute the notebook top-to-bottom**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01-eda.ipynb`
Expected: exits 0, all cells have output.

- [ ] **Step 4: Submit baseline and record score**

```bash
kaggle competitions submit playground-series-s6e7 -f submissions/01-majority.csv -m "majority-class baseline"
kaggle competitions submissions playground-series-s6e7 | head -5
```

Expected: public score ≈ 0.859. Record the actual score in the notebook's final markdown cell.

- [ ] **Step 5: Commit**

```bash
git add notebooks/01-eda.ipynb
git commit -m "Milestone 1: EDA and majority-class baseline"
```

---

### Task 3: Notebook 02 — CV harness + first LightGBM (Milestone 2)

**Files:**
- Create: `notebooks/02-baseline-cv.ipynb`

**Interfaces:**
- Consumes: EDA conclusions from notebook 01.
- Produces: the canonical CV pattern reused by notebooks 03/04 — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, a `run_cv(model_factory, X, y, X_test) -> (oof_preds, test_preds, fold_scores)` function returning class-probability arrays, and `make_submission(test_probs, name) -> Path` which argmaxes probabilities to labels and writes `submissions/<name>.csv`. Notebooks 03/04 copy these cells forward verbatim (teaching clarity over DRY).

- [ ] **Step 1: Write the CV notebook** with these sections, markdown-first as in Task 2:
  1. **Why CV before models** — teach: without a validation scheme you trust, every later experiment is noise. The public LB is itself a sample; chasing it overfits.
  2. **Choosing the scheme** — teach why `StratifiedKFold(5, shuffle=True, random_state=42)` fits here (no groups, no time order; stratification is doubly important because balanced accuracy makes small-class recall estimates noisy; 5 folds balances stability vs runtime at 690k rows). Name the alternatives (GroupKFold, TimeSeriesSplit, repeated CV) and when each would apply.
  3. **The harness** — implement `run_cv` and `make_submission` as specified in Interfaces. OOF **balanced accuracy** is THE number every experiment reports.
  4. **First model: LightGBM, near-default** — categoricals via `astype('category')` (teach: LightGBM's native categorical handling vs one-hot vs ordinal), missing values left as NaN (teach: GBMs route NaNs natively — a deliberate baseline choice that Milestone 3 experiments will challenge). `multiclass` objective, modest params (`n_estimators=1000`, `learning_rate=0.05`, early stopping on each fold). Run with and without `class_weight='balanced'` — under balanced accuracy, weighting should *help*; show it.
  5. **Results** — OOF balanced accuracy, per-class recall (confusion matrix), fold-score spread. Teach: balanced accuracy IS mean per-class recall, so the confusion matrix diagonal is the score decomposition; watch where the model sacrifices `fit`/`unhealthy` recall.
  6. **CV vs LB** — submit, then record CV score, LB score, and the gap in a markdown table. Teach: this gap is our trust calibration for the rest of the competition.

- [ ] **Step 2: Execute top-to-bottom**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02-baseline-cv.ipynb`
Expected: exits 0. OOF balanced accuracy expected ≈ 0.93+ (LB top is 0.953).

- [ ] **Step 3: Submit and record**

```bash
kaggle competitions submit playground-series-s6e7 -f submissions/02-lgbm-baseline.csv -m "LightGBM 5-fold baseline"
kaggle competitions submissions playground-series-s6e7 | head -5
```

Record CV, LB, and gap in the notebook.

- [ ] **Step 4: Commit**

```bash
git add notebooks/02-baseline-cv.ipynb
git commit -m "Milestone 2: stratified 5-fold CV harness and LightGBM baseline"
```

---

### Task 4: Notebook 03 — feature engineering + missing data experiments (Milestone 3)

**Files:**
- Create: `notebooks/03-features.ipynb`, `experiments.md`

**Interfaces:**
- Consumes: `run_cv`/`make_submission` pattern and baseline OOF score from notebook 02 (copied forward).
- Produces: a `build_features(train_df, test_df) -> (X, X_test)` cell encapsulating all *kept* transformations, reused by notebook 04; `experiments.md` log.

- [ ] **Step 1: Create `experiments.md`** with this structure and the baseline row:

```markdown
# Experiment Log

CV = 5-fold stratified OOF balanced accuracy, seed 42. Baseline: notebook 02 LightGBM.

| # | Hypothesis | Change | CV | Δ vs baseline | LB | Verdict |
|---|-----------|--------|----|---------------|----|---------|
| 0 | (baseline) | LGBM near-default, native cat, NaN passthrough | _fill_ | — | _fill_ | reference |
```

- [ ] **Step 2: Run the experiment ladder** in notebook 03. Each experiment: markdown cell stating the hypothesis and why it's plausible (citing notebook 01 findings where relevant) → code → CV result → verdict row appended to `experiments.md`. Planned ladder (results may reorder/prune it):
  1. **Missingness indicators** — add `<col>_missing` flags + row-wise missing count. Hypothesis: if missingness is informative (per EDA §5), explicit flags help even though LightGBM routes NaNs.
  2. **Imputation A/B** — median/mode imputation vs NaN passthrough, with and without indicators. Teach: imputation mostly helps linear/NN models; for GBMs it can *hurt* by destroying the missingness signal.
  3. **Ordinal encoding of ordered categoricals** — `stress_level`, `sleep_quality`, `physical_activity_level` have natural orders; map to integers. Hypothesis: monotone splits become cheaper for the tree.
  4. **Domain-derived interactions** — e.g. steps-per-exercise-minute, calories-per-step, BMI × activity, sleep duration × quality. Teach: GBMs learn interactions, but explicit ratios can shortcut what trees approximate with many splits.
  5. **Binning / rounding quirks** — if EDA §7 found synthetic-data artifacts (e.g. value clustering), encode them (frequency encoding, exact-value flags).
- Keep-rule: an experiment is kept only if CV improves ≥ 0.0005 (balanced accuracy is noisier than plain accuracy because small-class recall has high variance; notebook 02 measures the fold spread to justify the threshold); teach why we need a threshold at all.

- [ ] **Step 3: Consolidate** kept transformations into a single `build_features` cell; rerun CV on the combined set (teach: features that help alone can interact; combined ≠ sum of deltas). Final section: submission from the best combined model.

- [ ] **Step 4: Execute top-to-bottom**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03-features.ipynb`
Expected: exits 0.

- [ ] **Step 5: Submit best variant and record**

```bash
kaggle competitions submit playground-series-s6e7 -f submissions/03-features.csv -m "feature engineering round"
```

Update `experiments.md` with the LB score.

- [ ] **Step 6: Commit**

```bash
git add notebooks/03-features.ipynb experiments.md
git commit -m "Milestone 3: feature engineering experiments with log"
```

---

### Task 5: Notebook 04 — tuning, imbalance, ensemble, final (Milestone 4)

**Files:**
- Create: `notebooks/04-final.ipynb`
- Modify: `experiments.md` (continue the log)

**Interfaces:**
- Consumes: `build_features` from notebook 03, CV harness pattern, experiment log.
- Produces: final two selected submissions on Kaggle.

- [ ] **Step 1: Hyperparameter tuning section** — Optuna study (~30–50 trials on a 3-fold subsample for speed, teach the tradeoff) over the parameters that matter for LightGBM: `num_leaves`, `min_child_samples`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `lambda_l1/l2`. Teach: which parameters matter and why; diminishing returns; re-verify the winner on the full 5-fold before trusting it. Log to `experiments.md`.

- [ ] **Step 2: Imbalance section** — two competing approaches, run both:
  - Class weights / `is_unbalance` refinement beyond notebook 02's first pass. Teach: under balanced accuracy, upweighting minority classes aligns training loss with the metric; explore weight scaling beyond sklearn's 'balanced' heuristic.
  - **Per-class threshold optimization on OOF probabilities** — since the metric is accuracy on hard labels, tune the argmax: search per-class probability scaling factors on OOF predictions to maximize balanced accuracy. Teach: decoupling probability estimation from decision rule; why OOF (not train) predictions make this safe; why this matters even with class-weighted training.

- [ ] **Step 3: Ensemble section** —
  1. Seed averaging of the tuned LightGBM (3 seeds). Teach: variance reduction, cheapest ensemble there is.
  2. Train XGBoost and CatBoost on the same features/folds (teach: same-fold discipline makes OOF blending valid; CatBoost's ordered target encoding differs enough to decorrelate errors).
  3. Weighted probability blend, weights optimized on OOF. Compare vs best single model; teach when blending fails (highly correlated errors).

- [ ] **Step 4: Final selection** — pick the 2 final submissions (Kaggle allows 2): the best-CV blend and the most robust simpler model. Teach: why you hedge across the CV/LB gap rather than trust the public LB. Write a closing markdown recap of the whole arc: what moved the score, what didn't, and the transferable lessons.

- [ ] **Step 5: Execute top-to-bottom**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04-final.ipynb`
Expected: exits 0.

- [ ] **Step 6: Submit finals and record**

```bash
kaggle competitions submit playground-series-s6e7 -f submissions/04-final-blend.csv -m "tuned blend"
kaggle competitions submit playground-series-s6e7 -f submissions/04-robust.csv -m "robust single model"
```

Update `experiments.md`; verify final selection on the Kaggle site.

- [ ] **Step 7: Commit**

```bash
git add notebooks/04-final.ipynb experiments.md
git commit -m "Milestone 4: tuning, imbalance handling, ensemble, final submissions"
```

---

### Task 6: Notebook 05 — extension: diverse members + stacking (Milestone 5)

Added 2026-07-19 after Milestone 4 landed at LB 0.94946 vs the >0.95 spec target. PyTorch unavailable (Intel Mac + Python 3.13 has no compatible wheels), so the NN member is sklearn's MLPClassifier.

**Files:** Create `notebooks/05-stacking.ipynb`; continue `experiments.md`.

- [ ] E5a: HistGradientBoostingClassifier (native NaN + categorical, `class_weight='balanced'`), same folds, cached OOF/test probs.
- [ ] E5b: MLPClassifier on impute+flags+one-hot+standardized features (in-fold stats), majority class undersampled for balance, threshold-tuned.
- [ ] 4-way blend weight grid (LGBM seed-avg, CatBoost, HGB, MLP) + threshold refit; compare vs LogisticRegression stacking on concatenated OOF probs.
- [ ] Submit best; update `experiments.md`; commit.

---

**Status 2026-07-19: COMPLETE.** All tasks + extensions (milestones 5–7) executed. Final: public LB 0.95048 (CV 0.95036), rank ≈ 339/2192. Spec success criteria all met, including >0.95. Full record: experiments.md (E1–E13), README.md.
