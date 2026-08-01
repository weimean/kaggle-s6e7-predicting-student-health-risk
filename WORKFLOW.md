# How This Project Was Run

This repo doubles as a case study in AI-paired ML work. This document describes how
the collaboration ran: what I did, what Claude did, and the process mechanics behind a
two-day, seven-notebook competition run in which every point of the final score traces
to a logged decision.

## The shape of the collaboration

I ran this **explain-as-we-build**: Claude wrote and executed all code, and every step
carried the reasoning: why this CV scheme, what the alternatives were, how to choose.
My job was direction and judgment. I chose the format (over a "you code, I coach"
alternative we considered first), set the accelerated timeline, questioned the
reasoning at each milestone checkpoint ("how did we get so high in one go?", "what's
our competitors' edge?"), and made every go/no-go call, including when to extend past
the original plan and when to stop.

The notebooks read as lessons because that's what they were: the deliverable was
understanding first, leaderboard position second. The final rank (209 of 3,356 on the
private leaderboard, top 6.2%, a 386-place climb at the reveal) is the evidence that
the understanding was real.

## Spec → plan → milestones

Before any data touched a model, we wrote a [design spec](docs/superpowers/specs/) and
an [implementation plan](docs/superpowers/plans/) with explicit milestones, each ending
in a real Kaggle submission:

1. **EDA + naive baseline**, including the metric-probe submission that falsified our
   assumed metric on day one
2. **CV harness first**, the trust anchor for every later number
3. **Feature engineering as logged experiments**
4. **Tuning, decision thresholds, ensembling**
5. – 7. **Extensions**, each re-planned when evidence justified more work

At each milestone boundary we summarized results, answered my questions, and agreed on
the next move. We treated the plan as a living document: when the metric turned out to
be balanced accuracy instead of accuracy, we amended the spec, plan, and strategy the
same hour.

## The experiment-ledger contract

Every modeling idea in [experiments.md](experiments.md) follows one contract:

- **State the hypothesis first**, with a mechanism. An experiment we can't explain is
  a lottery ticket.
- **Change one thing.** Same folds (persisted fold assignment, asserted identical in
  every notebook), same seed, same model config as the baseline.
- **Keep only if ΔCV ≥ +0.0005**, the OOF noise floor measured from fold spread in
  notebook 02. Anything smaller is fold luck, and a pile of lucky keeps overfits your
  own CV.
- **Log drops with the same care.** The most valuable rows in the ledger are negative:
  imputation −0.041, missingness flags +0.0002, tuning ≤ 0 three times.

## Engineering patterns that made it work

- **Artifact caching everywhere.** Every heavy training run persists its out-of-fold
  and test probabilities to `artifacts/`; notebooks load caches when present. A crash
  after 3 hours of CatBoost (it happened) costs minutes instead of hours, and the
  final 8-member blend assembles from cache in seconds.
- **A side-venv for the awkward dependency.** PyTorch has no wheels for this machine
  (Intel Mac + Python 3.13), so torch jobs run in a pinned Python 3.12 venv
  (`torch 2.2.2` + a coherent 2024-era stack) as scripts, while notebooks stay in the
  main environment and consume the artifacts.
- **Parallel background training.** Long jobs ran as background processes; at one
  point CatBoost-with-target-statistics and a RealMLP were training while a scouting
  agent read competitor kernels.
- **Competitor intelligence as a first-class experiment source.** An agent dissected
  the six most-voted public kernels into score-evidenced techniques. Its top finding
  produced our best single model; its "verified" feature edge cost us 4 points when we
  extrapolated it to the wrong function class. Both outcomes went into the ledger.

## What transfers to real ML work

Each of these habits applies beyond Kaggle:

- **Probe your metric** before optimizing it. Cheap experiments that distinguish
  competing beliefs beat assumptions.
- **Build the validation instrument before the models**, and calibrate it against
  reality early (our CV predicted both leaderboards within ±0.003 for all 7
  submissions).
- **Keep a ledger.** Hypothesis-driven changes with a pre-committed keep threshold
  prevent random tweaking and self-deception.
- **Treat the decision rule as a separate object from the model**, and align both
  with the metric you're judged on.
- **Price ensemble members by contribution rather than solo score.** Diversity pays
  only above a quality floor.
