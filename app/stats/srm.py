"""Sample ratio mismatch (SRM) detection.

If randomization and logging are healthy, observed per-variant unit counts
match the configured split up to binomial noise. A chi-square goodness-of-fit
test catches deviations. SRM is a data-QUALITY alarm, not a metric result:
when it fires, every downstream estimate is suspect (some class of units is
missing from one arm — crashes, logging loss, targeting bugs).

The conventional threshold is very strict (p < 0.001) because platforms run
the check continuously and false alarms erode trust.
"""

from dataclasses import dataclass

from scipy import stats as sps

SRM_P_THRESHOLD = 0.001


@dataclass
class SrmResult:
    chi2: float
    p: float
    flagged: bool
    observed: dict          # variant_key -> observed count
    expected: dict          # variant_key -> expected count


def srm_test(observed_counts: dict[str, int], weights: dict[str, float]) -> SrmResult:
    """observed_counts: units per variant. weights: configured split weights
    (any scale; normalized here)."""
    keys = sorted(observed_counts)
    obs = [observed_counts[k] for k in keys]
    total_w = sum(weights[k] for k in keys)
    total_n = sum(obs)
    exp = [total_n * weights[k] / total_w for k in keys]
    chi2, p = sps.chisquare(obs, exp)
    return SrmResult(float(chi2), float(p), bool(p < SRM_P_THRESHOLD),
                     dict(zip(keys, obs)), {k: round(e, 1) for k, e in zip(keys, exp)})
