"""Empirical calibration report: does the platform's statistics machinery
deliver what it promises?

Two layers:
  1. STATISTICAL (pure numpy, thousands of replications) — validates the math
     in app/stats against its own guarantees: type I error, power, coverage,
     CUPED reduction, SRM behavior, and the peeking-inflation demo.
  2. PLATFORM (three short simulator runs) — validates the full pipeline:
     an A/A run stays quiet, an srm_bug run trips the SRM alarm, and a real
     effect is recovered with the right sign.

Run:  python -m scripts.calibration          (~1 minute)
"""

import os
import sys
import tempfile

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.stats import sequential
from app.stats.cuped import cuped_adjust
from app.stats.difference import diff_means, diff_proportions
from app.stats.sample_size import n_per_arm
from app.stats.srm import srm_test

RNG = np.random.default_rng(2026)
OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(name: str, value: float, lo: float, hi: float, note: str = ""):
    status = OK if lo <= value <= hi else BAD
    print(f"  {status}  {name}: {value:.4f}  (expected {lo:.3f}–{hi:.3f}) {note}")


# ---------------------------------------------------------------------------

def aa_false_positive_rate(reps: int = 4000, n: int = 800):
    print("\n== A/A: fixed-horizon false positive rate (alpha = 0.05) ==")
    rejects = 0
    for _ in range(reps):
        a, b = RNG.normal(0, 1, n), RNG.normal(0, 1, n)
        if diff_means(a, b).p < 0.05:
            rejects += 1
    check("FPR", rejects / reps, 0.04, 0.06)


def peeking_vs_sequential(reps: int = 20_000, looks: int = 5):
    print("\n== A/A: naive peeking vs alpha-spending (5 looks) ==")
    ts = np.linspace(1 / looks, 1, looks)
    incs = RNG.normal(0, np.sqrt(np.diff(np.concatenate([[0], ts]))), (reps, looks))
    z = incs.cumsum(axis=1) / np.sqrt(ts)

    naive = (np.abs(z) >= 1.96).any(axis=1).mean()
    print(f"       naive 'stop when p<0.05 at any look' type I: {naive:.4f}"
          f"  <-- the problem sequential testing solves")
    bounds = np.array([lk.z_boundary for lk in
                       sequential.boundaries(list(ts), 0.05, "obrien_fleming")])
    check("O'Brien-Fleming overall type I", (np.abs(z) >= bounds).any(axis=1).mean(),
          0.043, 0.057)
    bounds_p = np.array([lk.z_boundary for lk in
                         sequential.boundaries(list(ts), 0.05, "pocock")])
    check("Pocock overall type I", (np.abs(z) >= bounds_p).any(axis=1).mean(),
          0.043, 0.057)


def power_at_mde(reps: int = 2000):
    print("\n== Power: true effect = MDE, n from the calculator ==")
    base, mde = 0.10, 0.02
    n = n_per_arm("proportion", base, mde)          # designed for 80%
    hits = 0
    for _ in range(reps):
        x_c = RNG.binomial(n, base)
        x_t = RNG.binomial(n, base + mde)
        if diff_proportions(x_c, n, x_t, n).p < 0.05:
            hits += 1
    check(f"rejection rate (n={n}/arm)", hits / reps, 0.76, 0.85)


def ci_coverage(reps: int = 4000, n: int = 600):
    print("\n== Coverage: 95% CI contains the true effect ==")
    true = 0.3
    covered = 0
    for _ in range(reps):
        a = RNG.normal(0, 1, n)
        b = RNG.normal(true, 1, n)
        r = diff_means(a, b)
        covered += r.ci_low <= true <= r.ci_high
    check("coverage", covered / reps, 0.94, 0.96)


def cuped_reduction(reps: int = 300, n: int = 2000, rho: float = 0.6):
    print("\n== CUPED: variance reduction ~ rho^2, CIs strictly narrower ==")
    reductions, narrower = [], 0
    for _ in range(reps):
        x = RNG.normal(0, 1, 2 * n)
        y = rho * x + RNG.normal(0, np.sqrt(1 - rho**2), 2 * n)
        y[n:] += 0.05
        adj = cuped_adjust(y, x)
        reductions.append(adj.variance_reduction)
        raw = diff_means(y[:n], y[n:])
        cu = diff_means(adj.y_adj[:n], adj.y_adj[n:])
        narrower += (cu.ci_high - cu.ci_low) < (raw.ci_high - raw.ci_low)
    check(f"mean variance reduction (rho^2 = {rho**2:.2f})",
          float(np.mean(reductions)), rho**2 - 0.03, rho**2 + 0.03)
    check("fraction of reps with narrower CI", narrower / reps, 0.999, 1.0)


def srm_behavior(reps: int = 2000, n: int = 20_000):
    print("\n== SRM: fires on a 10% one-arm drop, silent on healthy data ==")
    fired_healthy = fired_broken = 0
    for _ in range(reps):
        t = RNG.binomial(n, 0.5)
        if srm_test({"c": n - t, "t": t}, {"c": 1, "t": 1}).flagged:
            fired_healthy += 1
        t2 = RNG.binomial(n, 0.5 * 0.9 / (0.5 + 0.5 * 0.9))   # treatment loses 10%
        if srm_test({"c": n - t2, "t": t2}, {"c": 1, "t": 1}).flagged:
            fired_broken += 1
    check("false alarm rate (threshold p<.001)", fired_healthy / reps, 0.0, 0.004)
    check("detection rate (10% drop @ 20k units)", fired_broken / reps, 0.99, 1.0)


# ---------------------------------------------------------------------------

def platform_end_to_end():
    print("\n== Platform: full-pipeline simulator runs (small population) ==")
    # isolated throwaway DB
    tmp = tempfile.mkdtemp()
    from app import db
    db.DB_PATH = os.path.join(tmp, "calib.db")
    db.init_db()
    from app import seed
    seed.seed_if_empty()
    from app import metrics_engine
    from app.simulator import clock, runner

    def fresh_run(seed_val, effects, anomaly):
        runner.init_simulation(4000, seed_val, pre_period_days=7)
        db.execute("UPDATE flags SET state='experiment', rollout_percent=100 WHERE id=1")
        for param, val in effects:
            db.execute("INSERT INTO sim_effects (flag_id, variant_key, parameter, value) "
                       "VALUES (1,'treatment',?,?) ON CONFLICT (flag_id, variant_key, parameter) "
                       "DO UPDATE SET value=excluded.value", (param, val))
        if anomaly != "none":
            clock.set_state(anomaly_mode=anomaly)
        runner.advance_days(8)
        flag = dict(db.query_one("SELECT * FROM flags WHERE id=1"))
        counts = {r["variant_key"]: r["n"] for r in db.query(
            "SELECT variant_key, COUNT(*) AS n FROM exposures WHERE flag_id=1 "
            "GROUP BY variant_key")}
        srm_res = srm_test(counts, {"control": 1, "treatment": 1})
        metric = dict(db.query_one("SELECT * FROM metrics WHERE key='watch_time'"))
        frame = metrics_engine.compute_metric(metric, flag, 0, 7)
        ctrl = frame[frame.variant == "control"]["y"].values
        treat = frame[frame.variant != "control"]["y"].values
        return srm_res, diff_means(ctrl, treat)

    srm_aa, res_aa = fresh_run(11, [], "none")
    print(f"       A/A run: watch-time effect {res_aa.effect:+.1f}s "
          f"(p={res_aa.p:.3f}), SRM p={srm_aa.p:.3f}")
    check("A/A |z| below 2.5", abs(res_aa.z), 0, 2.5, "(no phantom effect)")
    check("A/A SRM p above threshold", srm_aa.p, 0.001, 1.0)

    srm_bug, _ = fresh_run(12, [], "srm_bug")
    check("srm_bug run: SRM p below alarm threshold", srm_bug.p, 0.0, 1e-6,
          "(alarm fires)")

    srm_fx, res_fx = fresh_run(13, [("watch_time_multiplier", 1.15)], "none")
    print(f"       effect run: true watch x1.15 -> measured {res_fx.effect:+.1f}s, "
          f"z={res_fx.z:.1f}")
    check("effect recovered (z > 2.5)", res_fx.z, 2.5, float("inf"))
    check("effect run SRM quiet", srm_fx.p, 0.001, 1.0)


if __name__ == "__main__":
    print("A/B platform calibration report")
    print("=" * 60)
    aa_false_positive_rate()
    peeking_vs_sequential()
    power_at_mde()
    ci_coverage()
    cuped_reduction()
    srm_behavior()
    platform_end_to_end()
    print("\ndone.")
