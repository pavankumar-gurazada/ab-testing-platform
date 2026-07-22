"""CUPED variance reduction (Controlled-experiment Using Pre-Experiment Data,
Deng et al. 2013).

The whole method is three lines of math. For each unit, alongside the
experiment metric y we have a PRE-EXPERIMENT covariate x (same metric before
the experiment started). x is independent of assignment — randomization
happened after — so subtracting anything based on x cannot bias the effect;
it can only soak up variance:

    theta  = cov(y, x) / var(x)          (pooled across both arms)
    y_adj  = y - theta * (x - mean(x))
    var(y_adj) = var(y) * (1 - rho^2)    where rho = corr(y, x)

The variance falls by rho^2 — pre/post correlation of ~0.6 cuts variance ~36%
and CI width ~20%, for free. The adjusted values then flow through the exact
same difference test as raw values.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CupedResult:
    y_adj: np.ndarray
    theta: float
    variance_reduction: float   # 1 - var(y_adj)/var(y), i.e. ~rho^2


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> CupedResult:
    """y: metric values; x: pre-period covariate (NaN = unit missing a
    pre-period — those units get x = mean(x), i.e. zero adjustment)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    x = np.where(np.isnan(x), np.nanmean(x), x)

    var_x = x.var(ddof=1)
    if var_x == 0 or len(y) < 3:
        return CupedResult(y, 0.0, 0.0)

    theta = np.cov(y, x, ddof=1)[0, 1] / var_x
    y_adj = y - theta * (x - x.mean())

    var_y = y.var(ddof=1)
    reduction = 1.0 - y_adj.var(ddof=1) / var_y if var_y > 0 else 0.0
    return CupedResult(y_adj, float(theta), float(reduction))
