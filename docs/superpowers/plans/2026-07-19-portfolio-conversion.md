# Portfolio Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the working S6E7 repo into the public portfolio repo `kaggle-s6e7` — project-first README, WORKFLOW.md, three generated figures, LICENSE, notebook header cells, pushed to GitHub with real history.

**Architecture:** Additive conversion: new docs and a figure-generation script on top of the untouched working artifacts (notebooks, ledger, src, specs). One commit per task; final task publishes.

**Tech Stack:** matplotlib (figures), existing uv env, `gh` CLI for publishing.

## Global Constraints

- Voice: README/WORKFLOW.md in the user's first person; notebooks' original voice untouched.
- Framing: project is the headline; AI pairing disclosed in "How this was built" + WORKFLOW.md.
- Figures come from committed `src/make_figures.py`, never pasted; invoke the `dataviz` skill before writing chart code.
- No re-execution of notebooks (markdown-only header cells).
- No data files or `artifacts/` in git (already ignored — verify before push).
- Commits: plain messages, no AI attribution.

---

### Task 1: Figures (`src/make_figures.py` → `assets/`)

**Files:**
- Create: `src/make_figures.py`, `assets/score-arc.png`, `assets/cv-lb-calibration.png`, `assets/member-correlations.png`

**Interfaces:**
- Produces: the three PNG paths above, referenced verbatim by Task 2's README.

- [ ] **Step 1:** Invoke the `dataviz` skill, then write `src/make_figures.py` producing:
  1. **score-arc.png** — step chart of the arc with annotated levers. Data inline: `[("majority\n(metric probe)", 1/3), ("LightGBM\ndefault loss", 0.87542), ("+ class weights", 0.94894), ("+ thresholds", 0.94961), ("+ blend", 0.94968), ("+ NN member", 0.95009), ("+ cat_ev & greedy", 0.95036)]` (CV values; annotate final public LB 0.95048). Broken y-axis or log-gap treatment so the 0.94–0.951 region is readable.
  2. **cv-lb-calibration.png** — scatter of the 7 (CV, LB) submission pairs with y=x reference: `(None, 0.33333), (0.94894, 0.94667), (0.94916, 0.94942), (0.94968, 0.94946), (0.94961, 0.94918), (0.94990, 0.94978), (0.95009, 0.95006), (0.95036, 0.95048)` — drop the CV-less probe from the scatter; annotate max |gap| < 0.003.
  3. **member-correlations.png** — heatmap of final-blend member error correlations computed from `artifacts/*_oof.npy` (same member set and prior-correction as `notebooks/07-final-push.ipynb`), annotated with greedy weights `{lgbm .013, cat_ev .388, hgb .113, hgb_tuned .325, nn_te .15, lgbm_te .013}`.
- [ ] **Step 2:** Run `uv run python src/make_figures.py`; expect the three PNGs written to `assets/`. Read each PNG to verify it renders correctly and labels are legible.
- [ ] **Step 3:** Commit: `git add src/make_figures.py assets/ && git commit -m "Add figure generation for portfolio README"`

### Task 2: README rewrite

**Files:**
- Modify: `README.md` (full rewrite)

**Interfaces:**
- Consumes: `assets/*.png` from Task 1; links to `WORKFLOW.md` (Task 3), `experiments.md`, `notebooks/`.

- [ ] **Step 1:** Rewrite `README.md`, first person, project-first, with sections:
  1. **Title + hook:** "# Kaggle Playground S6E7 — Predicting Student Health Risk" + one-paragraph hook: hypothesis-driven run, public LB 0.95048, top 15.5% of 2192, every point accounted for in the experiment ledger.
  2. **Result figure:** embed `assets/score-arc.png`; keep the existing arc table (reuse from current README).
  3. **The method:** metric probe (0.333 discovery story), CV-as-instrument (embed `assets/cv-lb-calibration.png`), ledger discipline (+0.0005 keep-rule), alignment levers, diversity levers (embed `assets/member-correlations.png`).
  4. **The story in 7 notebooks:** table mapping each notebook to its one-line finding (from current README repo map + milestone conclusions).
  5. **Lessons:** keep the six lessons from the current README verbatim.
  6. **How this was built:** honest paragraph — built with Claude (Claude Code) as the pair: Claude wrote code/analysis and proposed hypotheses; I set direction, questioned reasoning at checkpoints, made every go/no-go call; link WORKFLOW.md.
  7. **Reproduce:** keep current reproduce section, plus `uv run python src/make_figures.py`.
- [ ] **Step 2:** Verify every relative link and image path resolves: `uv run python -c "import re,pathlib; md=pathlib.Path('README.md').read_text(); links=[m for m in re.findall(r'\]\(([^)#]+)\)', md) if not m.startswith('http')]; missing=[l for l in links if not pathlib.Path(l).exists()]; print('missing:', missing); assert not missing"`
- [ ] **Step 3:** Commit: `git add README.md && git commit -m "Rewrite README as portfolio landing page"`

### Task 3: WORKFLOW.md

**Files:**
- Create: `WORKFLOW.md`

**Interfaces:**
- Consumed by: README's "How this was built" link.

- [ ] **Step 1:** Write `WORKFLOW.md`, first person, sections:
  1. **The shape of the collaboration** — explain-as-we-build: Claude codes and explains, I steer at milestone checkpoints; why I chose it over you-drive-I-coach.
  2. **Spec → plan → milestones** — link `docs/superpowers/specs/` and `docs/superpowers/plans/`; milestones each end in a real submission.
  3. **The experiment ledger contract** — hypothesis stated first; one change; fixed folds/seed; keep only if ΔCV ≥ +0.0005 (noise floor measured in notebook 02); drops logged with the same care. Link `experiments.md`.
  4. **Engineering patterns that made it work** — artifact caching for crash-cheap long notebooks; the Python 3.12 side-venv for torch on an Intel Mac; parallel background training; a scouting agent that dissected 6 public kernels into score-evidenced techniques.
  5. **What transfers** — the same protocol applied to real ML work: metric probes, CV trust before modeling, ledgers over vibes, decision rules as first-class objects.
- [ ] **Step 2:** Re-run the Task 2 Step 2 link-check command against `WORKFLOW.md` (replace the filename in the command).
- [ ] **Step 3:** Commit: `git add WORKFLOW.md && git commit -m "Add WORKFLOW.md documenting the human+AI process"`

### Task 4: LICENSE + notebook headers + hygiene sweep

**Files:**
- Create: `LICENSE` (MIT, copyright 2026 Tsoi Wai Mun)
- Modify: `notebooks/01..07*.ipynb` (one prepended markdown cell each, via nbformat script — no execution)

- [ ] **Step 1:** Write `LICENSE` with the standard MIT text, `Copyright (c) 2026 Tsoi Wai Mun`.
- [ ] **Step 2:** Prepend to each notebook a markdown cell: `*Part N of 7 in a [hypothesis-driven run at Kaggle Playground S6E7](../README.md) — see the [experiment ledger](../experiments.md) for every hypothesis tested.*` via an nbformat loop script run with `uv run python`; assert cell count increased by exactly 1 per notebook.
- [ ] **Step 3:** Hygiene sweep: `grep -rn "/private/tmp\|scratchpad\|claude-501" README.md WORKFLOW.md notebooks/*.ipynb src/*.py || echo CLEAN` — expect CLEAN (fix any hits); `git status --short` — expect only intended changes; confirm `git ls-files data artifacts submissions` is empty.
- [ ] **Step 4:** Commit: `git add LICENSE notebooks/ && git commit -m "Add MIT license and notebook context headers"`

### Task 5: Publish to GitHub

**Files:** none (remote operation)

- [ ] **Step 1:** `gh auth status` — if not authenticated, stop and give the user: `gh auth login`, then `gh repo create kaggle-s6e7 --public --source . --push`.
- [ ] **Step 2:** Show the user a final pre-push summary (repo name, file tree top level, that history pushes as-is) and get explicit confirmation — publishing is outward-facing.
- [ ] **Step 3:** `gh repo create kaggle-s6e7 --public --source . --push` (or `git push -u origin main` if repo pre-created). Expected: remote created, history pushed.
- [ ] **Step 4:** Verify: `gh repo view kaggle-s6e7 --web` opens; spot-check README renders with figures and one notebook renders on github.com.
