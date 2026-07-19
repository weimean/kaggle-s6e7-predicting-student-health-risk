# How This Project Was Run

This repo doubles as a case study in AI-paired ML work. This document describes the
operating system of the collaboration — what I did, what Claude did, and the process
mechanics that made a two-day, seven-notebook competition run produce a score where
every point is traceable to a logged decision.

## The shape of the collaboration

I ran this **explain-as-we-build**: Claude wrote and executed all code, and every step
carried the reasoning — why this CV scheme, what the alternatives were, how to choose.
My job was direction and judgment: I chose the format (over a "you code, I coach"
alternative we considered first), set the accelerated timeline, questioned the reasoning
at each milestone checkpoint ("how did we get so high in one go?", "what's our
competitors' edge?"), and made every go/no-go call — including when to extend past the
original plan and when to stop.

The notebooks read as lessons because that's what they were: the deliverable was
understanding first, leaderboard position second. The final rank (top 15.5%) is the
evidence that the understanding was real.

## Spec → plan → milestones

Before any data touched a model, we wrote a [design spec](docs/superpowers/specs/) and
an [implementation plan](docs/superpowers/plans/) with explicit milestones, each ending
in a real Kaggle submission:

1. **EDA + naive baseline** — including the metric-probe submission that falsified our
   assumed metric on day one
2. **CV harness first** — the trust anchor every later number depends on
3. **Feature engineering as logged experiments**
4. **Tuning, decision thresholds, ensembling**
5. – 7. **Extensions**, each re-planned when evidence justified more work

Milestone boundaries were checkpoints: results summarized, questions answered, next
move agreed. The plan was a living document — when the metric turned out to be balanced
accuracy instead of accuracy, the spec, plan, and strategy were amended the same hour.

## The experiment-ledger contract

Every modeling idea in [experiments.md](experiments.md) follows one contract:

- **State the hypothesis first**, with a mechanism — an experiment without a "why"
  is a lottery ticket.
- **Change one thing.** Same folds (persisted fold assignment, asserted identical in
  every notebook), same seed, same model config as the baseline.
- **Keep only if ΔCV ≥ +0.0005** — the OOF noise floor measured from fold spread in
  notebook 02. Below that is fold luck, and a pile of lucky keeps is how you overfit
  your own CV.
- **Drops are logged with the same care.** The ledger's most valuable rows are
  negative: imputation −0.041, missingness flags +0.0002, tuning ≤ 0 three times.

## Engineering patterns that made it work

- **Artifact caching everywhere.** Every heavy training run persists its out-of-fold
  and test probabilities to `artifacts/`; notebooks load caches when present. A crash
  after 3 hours of CatBoost (it happened) costs minutes, not hours, and the final
  8-member blend assembles from cache in seconds.
- **A side-venv for the awkward dependency.** PyTorch has no wheels for this machine
  (Intel Mac + Python 3.13), so torch jobs run in a pinned Python 3.12 venv
  (`torch 2.2.2` + a coherent 2024-era stack) as scripts, while notebooks stay in the
  main environment and consume the artifacts.
- **Parallel background training.** Long jobs ran as background processes —
  at one point CatBoost-with-target-statistics and a RealMLP were training while a
  scouting agent read competitor kernels, all concurrently.
- **Competitor intelligence as a first-class experiment source.** An agent dissected
  the six most-voted public kernels into *score-evidenced* techniques (not folklore).
  Its top finding produced our best single model; its "verified" feature edge also
  cost us 4 points when extrapolated to the wrong function class — both outcomes
  went into the ledger.

## What transfers to real ML work

Nothing here is Kaggle-specific:

- **Probe your metric** before optimizing it — cheap experiments that distinguish
  competing beliefs beat assumptions.
- **Build the validation instrument before the models**, and calibrate it against
  reality early (our CV predicted the leaderboard within ±0.003 for all 7 submissions).
- **Ledgers over vibes.** Hypothesis-driven changes with a pre-committed keep
  threshold prevent both random tweaking and self-deception.
- **The decision rule is a separate object from the model** — align both with the
  metric you're actually judged on.
- **Ensemble members are priced by contribution, not solo score** — and diversity
  only pays above a quality floor.
