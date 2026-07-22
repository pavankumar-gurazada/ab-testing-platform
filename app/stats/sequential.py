"""Group sequential testing: alpha-spending boundaries and conditional power.

THE PROBLEM. Looking at experiment results repeatedly and stopping when
p < 0.05 inflates the false positive rate far above 5% (five looks ≈ 14%).
Group sequential designs fix this by pre-committing to a spending function
alpha(t) that parcels out the total alpha budget over information time
t = n_observed / n_target, and by raising the significance bar at each look
so the TOTAL crossing probability under H0 stays exactly alpha.

SPENDING FUNCTIONS (Lan & DeMets 1983 approximations):
  O'Brien-Fleming: alpha(t) = 2 - 2*Phi(z_{alpha/2} / sqrt(t))
      spends almost nothing early (huge early bar), ~alpha at the end;
      the final look needs z only slightly above the fixed-design 1.96.
  Pocock:          alpha(t) = alpha * ln(1 + (e-1)*t)
      spends roughly evenly; cheap early stops, but the final bar is
      noticeably higher and max sample size inflates more.

BOUNDARY COMPUTATION (the hand-rolled part — no mature Python library).
Under H0 the test statistic path is a Brownian motion in information time:
work with the score process S_k = Z_k*sqrt(t_k), whose increments are
independent N(0, t_k - t_{k-1}). Armitage/McPherson/Rowe recursion:

  f_1 = density of S_1;  at each look k find boundary b_k (via brentq) so
  P(no crossing before, S_k >= b_k) equals the alpha newly spent at look k;
  then propagate the sub-density of surviving paths through the normal
  transition kernel to look k+1 (numerical integration on a grid).

Two-sided testing uses symmetric boundaries +-c_k, treated as two mirrored
one-sided problems each with an alpha/2 budget: the upper boundary spends
f(t, alpha/2) (this is the Lan-DeMets/gsDesign convention behind the
published tables). The probability of one path crossing both boundaries is
negligible and ignored.

Verified in tests against published Lan-DeMets values, e.g. OBF K=5 equal
spacing, alpha=0.05 two-sided: 4.877, 3.357, 2.680, 2.290, 2.031.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps
from scipy.optimize import brentq

GRID_POINTS = 800
GRID_LO_SIGMAS = 8.0     # lower integration limit, in sqrt(t) units


def spending_obrien_fleming(t: float, alpha: float) -> float:
    """Cumulative two-sided alpha spent at information fraction t."""
    t = min(max(t, 1e-9), 1.0)
    return float(2 - 2 * sps.norm.cdf(sps.norm.ppf(1 - alpha / 2) / np.sqrt(t)))


def spending_pocock(t: float, alpha: float) -> float:
    t = min(max(t, 1e-9), 1.0)
    return float(alpha * np.log(1 + (np.e - 1) * t))


SPENDING = {"obrien_fleming": spending_obrien_fleming, "pocock": spending_pocock}


@dataclass
class Look:
    number: int
    information_fraction: float
    z_boundary: float             # reject (stop for efficacy) if |z| >= this
    alpha_spent_cumulative: float
    alpha_spent_this_look: float
    nominal_p: float              # two-sided p-value the boundary corresponds to


def boundaries(info_fractions: list[float], alpha: float = 0.05,
               spending: str = "obrien_fleming") -> list[Look]:
    """Efficacy z-boundaries for looks at the given information fractions.

    info_fractions: increasing, in (0, 1]; the final planned look should be 1.
    """
    spend_fn = SPENDING[spending]
    ts = [min(max(float(t), 1e-6), 1.0) for t in info_fractions]
    # per-side (upper-boundary) spending: one-sided problem with alpha/2 budget
    cum_upper = [spend_fn(t, alpha / 2) for t in ts]

    looks: list[Look] = []
    grid = None          # grid points for S (score scale)
    density = None       # sub-density of surviving (non-crossed) paths on grid

    prev_t = 0.0
    for k, t in enumerate(ts):
        target_upper = cum_upper[k] - (cum_upper[k - 1] if k else 0.0)
        target_upper = max(target_upper, 1e-12)
        dt = t - prev_t
        sd = np.sqrt(max(dt, 1e-12))

        if k == 0:
            # P(S_1 >= b) = target  =>  b = sd * z_target
            b_k = sd * sps.norm.ppf(1 - min(target_upper, 0.5))
        else:
            def upper_crossing(b: float) -> float:
                # P(survived to k-1, S_k >= b) via the transition kernel
                tail = sps.norm.sf((b - grid) / sd)
                return float(np.trapezoid(density * tail, grid)) - target_upper

            lo, hi = 0.0, GRID_LO_SIGMAS * np.sqrt(t)
            b_k = brentq(upper_crossing, lo, hi, xtol=1e-10)

        z_k = b_k / np.sqrt(t)
        # reported spend is the TWO-SIDED total (both symmetric boundaries)
        looks.append(Look(k + 1, t, float(z_k), float(2 * cum_upper[k]),
                          float(2 * target_upper), float(2 * sps.norm.sf(z_k))))

        # propagate the surviving sub-density to the next look
        if k < len(ts) - 1:
            new_grid = np.linspace(-GRID_LO_SIGMAS * np.sqrt(t), b_k, GRID_POINTS)
            if k == 0:
                new_density = sps.norm.pdf(new_grid / sd) / sd
            else:
                # f_k(s) = ∫ f_{k-1}(u) * phi((s - u)/sd)/sd du   (numerically)
                kernel = sps.norm.pdf((new_grid[:, None] - grid[None, :]) / sd) / sd
                new_density = np.trapezoid(kernel * density[None, :], grid, axis=1)
            grid, density = new_grid, new_density
        prev_t = t

    return looks


def conditional_power(z_now: float, t_now: float, z_final_boundary: float) -> float:
    """P(cross the final boundary | data so far), assuming the CURRENT TREND
    continues (drift estimated from the observed path).

    On the score scale B = z*sqrt(t) is Brownian with drift theta: estimate
    theta_hat = B_now / t_now, then the remaining increment is
    N(theta_hat*(1-t), 1-t). Used for futility: if even the current trend
    gives little chance of final success, stop and save the traffic.
    """
    t_now = min(max(t_now, 1e-9), 1.0 - 1e-9)
    b_now = z_now * np.sqrt(t_now)
    theta_hat = b_now / t_now
    remaining = 1.0 - t_now
    mean_final = b_now + theta_hat * remaining
    sd_final = np.sqrt(remaining)
    return float(sps.norm.sf((z_final_boundary - mean_final) / sd_final))


FUTILITY_CP_THRESHOLD = 0.20


# Maximum-sample-size inflation over a fixed design (two-sided alpha=0.05,
# power 0.80, equally spaced looks). Jennison & Turnbull (2000), Ch. 2 / the
# gsDesign package. For K > 5 the K=5 value is used (the curve is nearly flat).
INFLATION = {
    "obrien_fleming": {1: 1.000, 2: 1.008, 3: 1.017, 4: 1.024, 5: 1.028},
    "pocock":         {1: 1.000, 2: 1.110, 3: 1.166, 4: 1.202, 5: 1.229},
}


def inflation_factor(planned_looks: int, spending: str) -> float:
    table = INFLATION[spending]
    return table[min(max(planned_looks, 1), 5)]
