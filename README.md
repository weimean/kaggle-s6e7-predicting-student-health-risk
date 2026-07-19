# Playground S6E7 — Predicting Student Health Risk

A learning-first run at [Kaggle Playground S6E7](https://www.kaggle.com/competitions/playground-series-s6e7):
3-class classification of `health_condition` (`at-risk` 86% / `unhealthy` 8.4% / `fit` 5.8%),
690k train rows, 13 features with heavy missingness, scored by **balanced accuracy**.

**Final: public LB 0.95048 (CV 0.95036), rank ≈ 339/2192 — top 15.5%.**
Built explain-as-we-build style: every notebook teaches the *why* behind each step,
and every modeling idea lives in [`experiments.md`](experiments.md) as
hypothesis → change → CV → keep/drop.

## The arc

| stage | score | what earned it |
|---|---|---|
| majority baseline (metric probe) | 0.333 | discovered the metric is balanced accuracy, not accuracy |
| LightGBM, default loss | 0.875 | a GBM with sane defaults |
| + class-weighted loss | 0.949 | **align objective with metric** (+0.074, the biggest lever) |
| + features / + tuning | 0.949 | two measured zeros — honest redirects, not failures |
| + per-class thresholds | 0.9496 | **align decision rule with metric** |
| + cross-implementation blend (CatBoost, HGB) | 0.9498 | implementation diversity |
| + torch NN member | 0.9501 | function-class diversity |
| + CatBoost-native exact-value TE + greedy 8-member blend | **0.9505** | the one verified feature edge, encoded the tree-safe way |

## Repo map

```
notebooks/01-eda.ipynb          EDA + the metric-probe baseline
notebooks/02-baseline-cv.ipynb  CV harness (the trust anchor), weighted LightGBM
notebooks/03-features.ipynb     feature-engineering ladder (5 hypotheses, 0 keepers)
notebooks/04-final.ipynb        tuning, per-class thresholds, first ensemble
notebooks/05-stacking.ipynb     HGB + MLP members, blend vs stacking
notebooks/06-headroom.ipynb     torch NN, competitor intelligence, TE experiments
notebooks/07-final-push.ipynb   cat_ev, RealMLP, greedy grand blend
src/                            heavy training scripts (torch jobs run in .venv-torch)
experiments.md                  the complete hypothesis ledger (E1–E13)
docs/superpowers/               design spec + implementation plan
artifacts/ (gitignored)         cached OOF/test probabilities per member
```

## Lessons the ledger paid for

1. **Probe the metric before trusting it.** One "dumb" submission distinguished accuracy from balanced accuracy and reshaped the whole strategy.
2. **Alignment beats sophistication.** Loss↔metric (class weights) and decision-rule↔metric (thresholds) earned more than every clever idea combined.
3. **A CV you trust is the actual asset.** Fixed folds, OOF everything; seven submissions, never off by more than 0.001.
4. **Negative results are purchases.** Imputation −0.041 (destroys missingness signal), feature ladder 0, tuning ≤0 three times — each redirected effort with evidence.
5. **Features and function classes are a joint hypothesis.** Exact-value target encoding: −0.0044 on LightGBM, +0.0001 on our MLP, **+0.0006 via CatBoost's ordered statistics** — same idea, three outcomes.
6. **Diversity pays only above a quality floor.** Weak-but-different members (sklearn MLP, RealMLP-on-CPU) were zeroed by every combiner; the committee prices contribution, not novelty.

## Reproduce

```bash
uv sync                                   # main env (py3.13)
uv venv --python 3.12 .venv-torch         # torch side-env (Intel-mac torch 2.2.2)
VIRTUAL_ENV=.venv-torch uv pip install "torch==2.2.2" "numpy==1.26.4" \
    "scipy==1.13.1" "scikit-learn==1.5.2" "pandas==2.2.3" pytabkit
kaggle competitions download playground-series-s6e7 -p data && unzip -d data data/*.zip
# notebooks in order (02+ cache heavy results to artifacts/); torch jobs:
.venv-torch/bin/python src/train_nn.py    # etc.
```
