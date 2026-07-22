# A/B Testing Platform (learning prototype)

A self-contained experimentation platform built to *understand* how tools like
Statsig, Eppo or Optimizely work end to end — not to run in production. Every
internal that real platforms hide (hash buckets, alpha spending, CUPED theta,
the simulator's true effect sizes) is stored and rendered.

**Stack**: one FastAPI app with modular routers ("features as services"),
server-rendered Jinja2 pages (no JS build), SQLite for everything, and a
built-in traffic simulator with a fast-forward clock — a 4-week experiment
plays out in seconds.

## Features

- Feature flags with QA gating, targeting (platform / client / semver app
  version), and controlled exposure
- Gradual rollouts with pause / resume / rollback (kill switch) / launch
- Deterministic assignment: two independent sha256 hashes so ramping a rollout
  never reshuffles anyone's variant; first-touch stickiness; configurable
  randomization unit (user / device / session) and exposure trigger
  (assignment vs exposure)
- Custom proportion / continuous / ratio metrics over a raw event stream, with
  attribution windows, missing-value policy, and winsorization
- Experiments layered on flags with success / guardrail / supporting metric
  roles
- Group sequential testing: O'Brien-Fleming and Pocock alpha-spending
  boundaries (hand-rolled, verified against published Lan-DeMets tables),
  early stopping for efficacy, futility stopping via conditional power,
  repeated (sequential-valid) confidence intervals
- CUPED variance reduction from a simulated pre-experiment period
- Sample ratio mismatch detection (chi-square, p < 0.001), plus a rolling SRM
  monitor
- Difference and non-inferiority tests; delta-method analysis for ratio metrics
- Sample size & expected-duration calculator with sequential inflation factors
- Continuous rollout monitoring (daily per-variant trends, guardrail lights)
  and post-launch release analytics
- Traffic simulator with configurable **ground-truth effects** and injectable
  anomalies (`srm_bug`, `guardrail_degrade`) — so you can verify the stats
  recover the truth

## Run

Uses the `agentic` conda environment (needs fastapi, uvicorn, jinja2, numpy,
scipy, pandas, statsmodels, python-multipart; pytest for tests):

```bash
conda run -n agentic uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

Tests and the empirical calibration report:

```bash
conda run -n agentic pytest                     # 55 tests, ~10 s
conda run -n agentic python -m scripts.calibration   # ~1 min
```

The database is a single `platform.db` file in the repo root (gitignored).
Delete it — or use the simulator's Reset button — to start over.

## Suggested walkthrough

1. **Simulator** → Init population (20 000 users, 14 pre-period days). The
   pre-period collects each user's CUPED covariates.
2. **Flags** → open `new-lesson-player` → move draft → **qa** (only the 25 QA
   users get it — try the assignment debugger) → **rollout** at 10% → ramp to
   50%. Advance a few days in between and watch the monitoring page.
3. **Simulator** → set true effects for `treatment`, e.g.
   `lesson_complete_uplift_pp = 0.04` and `latency_add_ms = 60`.
4. **Experiments** → create one on the flag (5 looks, O'Brien-Fleming, CUPED
   on) → attach `course_completion_rate` as success, `page_latency` as
   guardrail, `watch_time` as supporting → Compute design (baseline is
   pre-filled from the pre-period) → Start.
5. Alternate **Simulator → advance 3–4 days** and **Analysis → run look**.
   Watch the alpha-spending ledger fill in, the boundary descend, and the
   latency guardrail catch the +60 ms you configured.
6. Stop early for efficacy (or breach the guardrail), **launch** the flag from
   its page, advance more days, and read the **release analytics**.
7. Break things on purpose: set anomaly mode `srm_bug` and watch SRM fire on
   the next look; try `guardrail_degrade` during a rollout.

## Layout

```
app/
├── main.py            FastAPI app + dashboard
├── schema.sql         all tables, heavily commented (start reading here)
├── db.py              thin sqlite3 helpers (no ORM on purpose)
├── assignment.py      hashing, targeting, rollout gate, stickiness, exposure
├── metrics_engine.py  event stream -> per-unit analysis frames
├── seed.py            demo app, example flag, metric catalog
├── routers/           flags, assign (SDK), metrics, experiments, analysis, simulator
├── stats/             one concern per file:
│   ├── sequential.py    alpha-spending boundaries (the hand-rolled core)
│   ├── difference.py    z-tests for proportions and means
│   ├── cuped.py         3-line variance reduction
│   ├── ratio.py         delta method
│   ├── srm.py           chi-square mismatch alarm
│   ├── noninferiority.py, sample_size.py, intervals.py
├── simulator/         population, behavior (the true DGP), clock, runner
└── templates/         Jinja2 pages, inline-SVG charts
scripts/calibration.py empirical proof: FPR ~ 5%, power ~ 80%, coverage ~ 95%,
                       CUPED ~ rho², SRM fires, peeking inflation demo
tests/                 assignment, stats vs textbook values, engine edge cases, e2e
```

## Deliberate simplifications

- One process, one SQLite file; per-request connections, no migrations
- Two variants per flag for analysis (control + one treatment)
- Targeting rules are AND-ed; no OR groups
- Session-randomized flags resolve once per user-day (still shows why session
  randomization + user metrics is a footgun)
- Guardrails: one-sided α = 0.05 each, no multiplicity adjustment
- Non-inferiority reuses the two-sided efficacy boundary (conservative)
- Effects apply only on feature-surface-touch days (that's the exposure
  dilution you can see when comparing assignment- vs exposure-triggered
  analysis)
