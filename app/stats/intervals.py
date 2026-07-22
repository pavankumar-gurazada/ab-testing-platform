"""Confidence intervals, including sequential-look ("repeated") intervals.

The naive Wald CI (effect +- 1.96*SE) is only valid for a SINGLE
pre-committed look. At interim looks of a sequential design the analysis is
conditioned on "we are still running" and will be repeated — reporting 95%
Wald CIs at every look means far more than 5% of experiments produce at least
one non-covering interval.

The repeated CI (Jennison & Turnbull) fixes this by using the look's own
boundary multiplier c_k instead of 1.96:

    effect +- c_k * SE

Because sum of crossing probabilities over all looks is exactly alpha, the
probability that ANY look's repeated interval excludes the truth is at most
alpha. Early looks (huge c_k under O'Brien-Fleming) give very wide intervals
— honest about how little is known mid-experiment.
"""

from scipy import stats as sps


def wald_ci(effect: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    z = sps.norm.ppf(1 - alpha / 2)
    return effect - z * se, effect + z * se


def repeated_ci(effect: float, se: float, boundary_z: float) -> tuple[float, float]:
    """Sequential-valid CI at a look whose efficacy boundary is boundary_z."""
    return effect - boundary_z * se, effect + boundary_z * se
