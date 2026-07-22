"""Ratio metrics via the delta method.

A ratio metric like completions-per-enrollment is sum(num_i)/sum(den_i) per
arm = mean(num)/mean(den). Units are i.i.d. but the RATIO of means is not a
mean of unit-level values, so the usual SE formula doesn't apply. First-order
Taylor expansion (the delta method) gives:

    Var(N̄/D̄) ≈ (1/n) * ( var(num) - 2R·cov(num,den) + R²·var(den) ) / mean(den)²

with R = mean(num)/mean(den). Verified against a bootstrap in the test suite.
"""

import numpy as np
from scipy import stats as sps

from .difference import EffectResult


def ratio_stats(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Per-arm ratio and its delta-method standard error."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    mean_d = den.mean()
    if n < 2 or mean_d == 0:
        return float("nan"), float("nan")
    r = num.mean() / mean_d
    cov = np.cov(num, den, ddof=1)          # [[var_n, cov], [cov, var_d]]
    var_r = (cov[0, 0] - 2 * r * cov[0, 1] + r**2 * cov[1, 1]) / (n * mean_d**2)
    return float(r), float(np.sqrt(max(var_r, 0.0)))


def diff_ratios(num_c: np.ndarray, den_c: np.ndarray,
                num_t: np.ndarray, den_t: np.ndarray,
                alpha: float = 0.05) -> EffectResult:
    r_c, se_c = ratio_stats(num_c, den_c)
    r_t, se_t = ratio_stats(num_t, den_t)
    effect = r_t - r_c
    se = float(np.sqrt(se_c**2 + se_t**2))
    z = effect / se
    p = 2 * sps.norm.sf(abs(z))
    zc = sps.norm.ppf(1 - alpha / 2)
    return EffectResult(float(effect), se, float(z), float(p),
                        float(effect - zc * se), float(effect + zc * se),
                        r_c, r_t, len(num_c), len(num_t))
