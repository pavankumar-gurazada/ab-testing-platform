"""Non-inferiority tests: "treatment is not worse than control by more than
a margin delta".

A difference test asks: is the effect != 0? Sometimes the right question is
weaker: a cheaper/simpler variant is worth shipping as long as it doesn't hurt
the metric by more than delta. Absence of significance in a difference test
does NOT establish this ("no evidence of harm" != "evidence of no harm") —
you need to shift the null hypothesis:

  metric where higher is better (direction='increase'):
      H0: effect <= -delta   vs   H1: effect > -delta
      z_NI = (effect + delta) / SE,   reject (non-inferior) for large z

  metric where lower is better (direction='decrease'):
      H0: effect >= +delta   vs   H1: effect < +delta
      z_NI = (delta - effect) / SE

Both are ONE-SIDED tests. The reported CI stays the ordinary two-sided CI on
the effect — the standard presentation: non-inferiority holds when the CI
stays on the good side of the margin.

Reuses the SEs from difference.py / ratio.py; only the null moves.
"""

from dataclasses import dataclass

from scipy import stats as sps

from .difference import EffectResult


@dataclass
class NiResult:
    z: float
    p: float                 # one-sided
    margin: float
    non_inferior: bool       # at the given alpha


def noninferiority_test(res: EffectResult, margin: float,
                        direction: str = "increase",
                        alpha: float = 0.05) -> NiResult:
    """res: an EffectResult from diff_means/diff_proportions/diff_ratios.
    margin: positive absolute delta, in the metric's units."""
    assert margin > 0, "margin must be positive"
    if direction == "increase":
        z = (res.effect + margin) / res.se
    else:
        z = (margin - res.effect) / res.se
    p = float(sps.norm.sf(z))
    return NiResult(float(z), p, margin, bool(p < alpha))
