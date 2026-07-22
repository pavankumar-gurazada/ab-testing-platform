"""Stats module tests against textbook/published values and brute force."""

import numpy as np
import pytest
from scipy import stats as sps

from app.stats import cuped, difference, intervals, noninferiority, ratio, sequential, srm
from app.stats.sample_size import n_per_arm


# ---------------------------------------------------------------------------
# sequential.py — the hand-rolled core, checked against published tables
# ---------------------------------------------------------------------------

def test_obf_boundaries_match_lan_demets_k5():
    """Lan-DeMets O'Brien-Fleming spending, K=5 equal spacing, two-sided
    alpha=0.05. Published boundaries (e.g. gsDesign, Jennison & Turnbull):
    4.877, 3.357, 2.680, 2.290, 2.031."""
    looks = sequential.boundaries([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05,
                                  spending="obrien_fleming")
    got = [lk.z_boundary for lk in looks]
    expected = [4.877, 3.357, 2.680, 2.290, 2.031]
    assert got == pytest.approx(expected, abs=0.01)


def test_pocock_boundaries_k5():
    """Lan-DeMets Pocock-type spending, K=5, alpha=0.05 two-sided:
    published ~2.438, 2.427, 2.410, 2.397, 2.386 (nearly flat)."""
    looks = sequential.boundaries([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.05,
                                  spending="pocock")
    got = [lk.z_boundary for lk in looks]
    expected = [2.438, 2.427, 2.410, 2.397, 2.386]
    assert got == pytest.approx(expected, abs=0.02)


def test_single_look_reduces_to_fixed_design():
    looks = sequential.boundaries([1.0], alpha=0.05)
    assert looks[0].z_boundary == pytest.approx(1.96, abs=0.001)
    assert looks[0].alpha_spent_cumulative == pytest.approx(0.05, abs=1e-9)


def test_total_alpha_spend_is_alpha():
    looks = sequential.boundaries([0.25, 0.5, 0.75, 1.0], alpha=0.05)
    assert sum(lk.alpha_spent_this_look for lk in looks) == pytest.approx(0.05, abs=1e-6)
    assert looks[-1].alpha_spent_cumulative == pytest.approx(0.05, abs=1e-9)


def test_obf_boundaries_monte_carlo_type1():
    """Simulate 40k Brownian H0 paths observed at the look times; the fraction
    crossing ANY boundary must be ~alpha. This is the property the recursion
    exists to guarantee — verify it by brute force."""
    ts = np.array([0.25, 0.5, 0.75, 1.0])
    looks = sequential.boundaries(list(ts), alpha=0.05)
    bounds = np.array([lk.z_boundary for lk in looks])
    rng = np.random.default_rng(7)
    n = 40_000
    increments = rng.normal(0, np.sqrt(np.diff(np.concatenate([[0], ts]))), (n, len(ts)))
    s = increments.cumsum(axis=1)                  # score-scale paths
    z = s / np.sqrt(ts)
    crossed = (np.abs(z) >= bounds).any(axis=1)
    # binomial SE at p=0.05, n=40k is ~0.0011; allow 4 sigma
    assert crossed.mean() == pytest.approx(0.05, abs=0.005)


def test_conditional_power_extremes():
    # trend already at the final boundary with most information -> CP high
    assert sequential.conditional_power(2.5, 0.8, 2.0) > 0.9
    # flat/negative trend -> CP low
    assert sequential.conditional_power(0.0, 0.5, 2.0) < 0.05


# ---------------------------------------------------------------------------
# difference.py
# ---------------------------------------------------------------------------

def test_diff_proportions_textbook():
    """120/1000 vs 150/1000: pooled p = .135,
    z = .03 / sqrt(.135*.865*2/1000) = 1.9630, p = .0496."""
    res = difference.diff_proportions(120, 1000, 150, 1000)
    assert res.effect == pytest.approx(0.03, abs=1e-12)
    assert abs(res.z) == pytest.approx(1.9630, abs=0.001)
    assert res.p == pytest.approx(0.0496, abs=0.001)
    assert res.ci_low < 0.03 < res.ci_high


def test_diff_means_matches_scipy_welch():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 3, 500), rng.normal(10.5, 3, 500)
    res = difference.diff_means(a, b)
    t_stat, p_scipy = sps.ttest_ind(b, a, equal_var=False)
    assert res.z == pytest.approx(t_stat, abs=0.01)     # z ~ t at n=500
    assert res.p == pytest.approx(p_scipy, abs=0.005)


# ---------------------------------------------------------------------------
# ratio.py — delta method vs bootstrap
# ---------------------------------------------------------------------------

def test_delta_method_se_matches_bootstrap():
    rng = np.random.default_rng(1)
    n = 2000
    den = rng.poisson(3, n) + 1.0
    num = rng.poisson(den * 0.7)                        # correlated num/den
    _, se_delta = ratio.ratio_stats(num, den)
    boot = np.empty(3000)
    for i in range(3000):
        idx = rng.integers(0, n, n)
        boot[i] = num[idx].mean() / den[idx].mean()
    assert se_delta == pytest.approx(boot.std(ddof=1), rel=0.08)


# ---------------------------------------------------------------------------
# cuped.py
# ---------------------------------------------------------------------------

def test_cuped_theta_equals_regression_slope():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 2, 4000)
    y = 3 + 1.7 * x + rng.normal(0, 1, 4000)
    res = cuped.cuped_adjust(y, x)
    slope = np.polyfit(x, y, 1)[0]
    assert res.theta == pytest.approx(slope, abs=1e-6)


def test_cuped_variance_reduction_is_rho_squared():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20_000)
    y = 0.6 * x + rng.normal(0, 0.8, 20_000)
    rho2 = np.corrcoef(x, y)[0, 1] ** 2
    res = cuped.cuped_adjust(y, x)
    assert res.variance_reduction == pytest.approx(rho2, abs=0.01)
    assert res.y_adj.var() < y.var()


def test_cuped_does_not_bias_the_effect():
    """Adjustment must leave the between-arm difference of means (in
    expectation) unchanged: check on a large sample with a real effect."""
    rng = np.random.default_rng(4)
    n = 50_000
    x = rng.normal(0, 1, 2 * n)
    y = 1.0 * x + rng.normal(0, 1, 2 * n)
    y[n:] += 0.1                                        # true effect on arm 2
    res = cuped.cuped_adjust(y, x)
    raw = y[n:].mean() - y[:n].mean()
    adj = res.y_adj[n:].mean() - res.y_adj[:n].mean()
    assert adj == pytest.approx(0.1, abs=0.02)
    assert adj == pytest.approx(raw, abs=0.02)


def test_cuped_handles_missing_covariates():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 1000)
    y = x + rng.normal(0, 1, 1000)
    x[::10] = np.nan
    res = cuped.cuped_adjust(y, x)
    assert np.isfinite(res.y_adj).all()
    assert res.variance_reduction > 0.2


# ---------------------------------------------------------------------------
# srm.py
# ---------------------------------------------------------------------------

def test_srm_hand_value():
    """5050/4950 on a 50/50 split: chi2 = (50^2/5000)*2 = 1.0, p ~ .317."""
    res = srm.srm_test({"control": 5050, "treatment": 4950},
                       {"control": 1, "treatment": 1})
    assert res.chi2 == pytest.approx(1.0, abs=1e-9)
    assert res.p == pytest.approx(0.3173, abs=0.001)
    assert not res.flagged


def test_srm_fires_on_real_mismatch():
    res = srm.srm_test({"control": 5000, "treatment": 4500},
                       {"control": 1, "treatment": 1})
    assert res.flagged and res.p < 1e-6


def test_srm_respects_unequal_weights():
    res = srm.srm_test({"control": 9000, "treatment": 1000},
                       {"control": 9, "treatment": 1})
    assert not res.flagged


# ---------------------------------------------------------------------------
# noninferiority.py
# ---------------------------------------------------------------------------

def test_noninferiority_clear_cases():
    # zero effect, tight SE, margin 0.05 -> clearly non-inferior
    res = difference.EffectResult(0.0, 0.01, 0, 1, -0.02, 0.02, 0.5, 0.5, 1000, 1000)
    assert noninferiority.noninferiority_test(res, 0.05, "increase").non_inferior
    # effect -0.08 with margin 0.05 -> inferior
    res_bad = difference.EffectResult(-0.08, 0.01, -8, 0, -0.1, -0.06, 0.5, 0.42, 1000, 1000)
    assert not noninferiority.noninferiority_test(res_bad, 0.05, "increase").non_inferior


def test_noninferiority_direction_decrease():
    # latency +8ms with margin 20ms and lower-is-better -> non-inferior
    res = difference.EffectResult(8.0, 3.0, 2.7, 0.008, 2.1, 13.9, 800, 808, 1000, 1000)
    assert noninferiority.noninferiority_test(res, 20.0, "decrease").non_inferior
    # latency +30ms breaches the 20ms margin
    res_bad = difference.EffectResult(30.0, 3.0, 10, 0, 24, 36, 800, 830, 1000, 1000)
    assert not noninferiority.noninferiority_test(res_bad, 20.0, "decrease").non_inferior


# ---------------------------------------------------------------------------
# sample_size.py (pure math parts) & intervals.py
# ---------------------------------------------------------------------------

def test_n_per_arm_proportion_textbook():
    """0.10 -> 0.12, alpha .05, power .8: classic answer ~3800-4000/arm."""
    n = n_per_arm("proportion", 0.10, 0.02)
    assert 3500 < n < 4100


def test_n_per_arm_continuous():
    """d = mde/sd = 0.1 -> n = 2*(1.96+0.84)^2/0.1^2 ~ 1571/arm."""
    n = n_per_arm("continuous", 100, 1.0, sd=10.0)
    assert n == pytest.approx(1571, abs=10)


def test_sequential_design_inflates_n():
    fixed = n_per_arm("proportion", 0.10, 0.02, planned_looks=1)
    obf5 = n_per_arm("proportion", 0.10, 0.02, planned_looks=5, spending="obrien_fleming")
    pocock5 = n_per_arm("proportion", 0.10, 0.02, planned_looks=5, spending="pocock")
    assert fixed < obf5 < pocock5


def test_repeated_ci_wider_than_wald():
    lo_w, hi_w = intervals.wald_ci(0.1, 0.02)
    lo_r, hi_r = intervals.repeated_ci(0.1, 0.02, boundary_z=4.877)
    assert lo_r < lo_w and hi_r > hi_w
