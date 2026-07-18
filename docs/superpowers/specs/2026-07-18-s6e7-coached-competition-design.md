# S6E7 Explain-as-We-Build Competition — Design

**Date:** 2026-07-18
**Competition:** [Playground Series S6E7 — Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7)
**Deadline:** 2026-07-31 (accelerated timeline)

## Goal

Use this competition as a structured learning experience for an intermediate ML practitioner. Claude writes and runs all code; every step is annotated with the reasoning behind it. Success is measured by understanding and process quality first, leaderboard score second.

## Competition facts

- **Task:** 3-class classification of `health_condition`: `at-risk` (86%), `unhealthy` (8.4%), `fit` (5.8%)
- **Data:** 690,088 train rows, 295,753 test rows, 13 features (numeric: sleep_duration, heart_rate, bmi, calorie_expenditure, step_count, exercise_duration, water_intake; categorical: diet_type, stress_level, sleep_quality, physical_activity_level, smoking_alcohol, gender)
- **Missing values:** substantial, in nearly every column (stress_level worst at ~12%)
- **Metric:** accuracy (hard-label submissions; majority-class floor ≈ 0.859; leaderboard top ≈ 0.953)

## Working model

- **Claude** writes the notebooks and runs everything, but each notebook is written to teach, not just to score: every step carries the *why* — why this CV scheme, why this encoding, what the alternatives were and how to choose between them.
- At genuinely open decision points (e.g. impute vs. missingness-as-signal), Claude shows the experiment rather than asserting the answer.
- **User** reads, asks questions anytime, and steers priorities. The experiment log is Claude's auditable record of hypothesis → result.
- Learning focus areas (all four): EDA & data intuition, validation & CV strategy, feature engineering & missing data, gradient boosting mastery.

## Milestones

Each milestone ends in a real Kaggle submission.

1. **EDA + naive baseline** (~1 session): systematic exploration of features, missingness patterns, target imbalance; trivial baseline submission to establish the metric floor.
2. **CV + first GBM** (~1 session): stratified k-fold setup; verify CV tracks the public leaderboard; first LightGBM submission. This is the trust anchor for all later work.
3. **Feature engineering + missing data** (~1–2 sessions): experiment log starts. Every idea logged in `experiments.md` as hypothesis → change → CV score → keep/drop. Encodings, missingness indicators, imputation strategies, interactions.
4. **Tuning, imbalance & ensemble** (~1–2 sessions): hyperparameter tuning that matters; class weights vs. per-class threshold optimization (relevant because the metric is accuracy on imbalanced classes); blend best models for the final submission.

Accelerated: milestones 1–2 are single focused sessions; no gating on mastery — if the CV setup works, move on.

## Repo layout

```
playground-series-s6e7/
├── data/               # competition data (gitignored)
├── notebooks/          # annotated notebooks: 01-eda, 02-baseline-cv, 03-features, 04-final
├── experiments.md      # hypothesis log (from milestone 3)
├── submissions/        # generated CSVs (gitignored)
├── docs/superpowers/specs/  # this design doc
└── pyproject.toml      # uv-managed environment
```

## Environment

`uv`-managed virtualenv (Python 3.13) with: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, matplotlib, seaborn, optuna, jupyter, kaggle.

## Success criteria

- Every modeling choice in the final notebook is explained well enough that the user could defend it.
- CV score predicts LB score within a known, stable gap.
- Final accuracy beats 0.95 (top of leaderboard ≈ 0.953).
- `experiments.md` shows hypothesis-driven iteration, not random tweaking.

## Out of scope

- Neural-network approaches, pseudo-labeling, and other advanced topics — unless milestones 1–4 finish early and the user wants extensions.
