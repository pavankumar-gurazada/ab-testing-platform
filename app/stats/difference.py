"""Two-sample difference tests for proportion and continuous metrics.

Library usage:
  * proportions: statsmodels proportions_ztest (pooled-SE z test) and
    confint_proportions_2indep (Wald CI on the difference)
  * means: Welch z assembled by hand from numpy moments so the standard error
    is visible — SE = sqrt(s_c^2/n_c + s_t^2/n_t) — with p/CI from scipy.

At experiment sample sizes (hundreds+ per arm) z and t are indistinguishable;
we use z throughout so fixed-horizon and sequential analyses share one scale.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest


@dataclass
class EffectResult:
    effect: float          # treatment - control (absolute)
    se: float
    z: float
    p: float               # two-sided
    ci_low: float
    ci_high: float
    control_mean: float
    treatment_mean: float
    control_n: int
    treatment_n: int


def diff_proportions(x_c: int, n_c: int, x_t: int, n_t: int,
                     alpha: float = 0.05) -> EffectResult:
    """x = conversions, n = units, for control (c) and treatment (t)."""
    z, p = proportions_ztest([x_t, x_c], [n_t, n_c])   # pooled SE under H0
    ci_low, ci_high = confint_proportions_2indep(
        x_t, n_t, x_c, n_c, method="wald", alpha=alpha)
    p_c, p_t = x_c / n_c, x_t / n_t
    effect = p_t - p_c
    # Unpooled (Wald) SE — the SE that goes with the CI and sequential z path.
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    return EffectResult(effect, float(se), float(z), float(p),
                        float(ci_low), float(ci_high), p_c, p_t, n_c, n_t)


def diff_means(y_c: np.ndarray, y_t: np.ndarray, alpha: float = 0.05) -> EffectResult:
    """Welch two-sample z on raw (or CUPED-adjusted) per-unit values."""
    y_c, y_t = np.asarray(y_c, float), np.asarray(y_t, float)
    n_c, n_t = len(y_c), len(y_t)
    m_c, m_t = y_c.mean(), y_t.mean()
    se = np.sqrt(y_c.var(ddof=1) / n_c + y_t.var(ddof=1) / n_t)
    effect = m_t - m_c
    z = effect / se
    p = 2 * sps.norm.sf(abs(z))
    zc = sps.norm.ppf(1 - alpha / 2)
    return EffectResult(float(effect), float(se), float(z), float(p),
                        float(effect - zc * se), float(effect + zc * se),
                        float(m_c), float(m_t), n_c, n_t)
