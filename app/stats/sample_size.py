"""Sample size and expected-duration calculations.

Power math via statsmodels (NormalIndPower); the effect size is standardized
first:
  proportion: Cohen's h via proportion_effectsize(baseline+mde, baseline)
  continuous: d = mde / sd

For group sequential designs the MAXIMUM sample size is the fixed-design n
times a spending-function inflation factor (see sequential.INFLATION) — the
price paid for the option to stop early. The EXPECTED n is lower.

Duration = remaining units needed / measured enrollment rate. The rate is
measured from the actual entry events of the last 7 simulated days, so it
already includes targeting, rollout %, and exposure dilution; before any
traffic exists we estimate it from population * mean activity * rollout% *
exposure rate.
"""

import math

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from .. import db
from ..simulator.runner import EXPOSURE_RATE
from .sequential import inflation_factor


def n_per_arm(metric_type: str, baseline: float, mde: float,
              alpha: float = 0.05, power: float = 0.8,
              sd: float | None = None,
              planned_looks: int = 1,
              spending: str = "obrien_fleming") -> int:
    """Max n per arm for a two-arm, equal-split, two-sided test."""
    if metric_type == "proportion":
        es = proportion_effectsize(min(baseline + mde, 0.999), baseline)
    else:
        if not sd or sd <= 0:
            raise ValueError("continuous/ratio metrics need a standard deviation")
        es = mde / sd
    n = NormalIndPower().solve_power(effect_size=abs(es), alpha=alpha, power=power,
                                     ratio=1.0, alternative="two-sided")
    return math.ceil(n * inflation_factor(planned_looks, spending))


def measured_entry_rate(flag: dict) -> float:
    """Units entering the analysis population per simulated day, measured over
    the last 7 days of actual traffic (0 if none)."""
    table = "exposures" if flag["exposure_trigger"] == "exposure" else "assignments"
    row = db.query_one(f"""
        SELECT COUNT(*) AS n, MAX(sim_day) - MIN(sim_day) + 1 AS days
        FROM {table} WHERE flag_id = ? AND sim_day > (
            SELECT MAX(sim_day) - 7 FROM {table} WHERE flag_id = ?)
    """, (flag["id"], flag["id"]))
    if row and row["n"]:
        return row["n"] / max(row["days"], 1)
    return 0.0


def estimated_entry_rate(flag: dict) -> float:
    """Pre-traffic estimate: population * mean activity * rollout% * exposure."""
    pop = db.query_one("SELECT COUNT(*) AS n, AVG(activity) AS act FROM users")
    if not pop or not pop["n"]:
        return 0.0
    rate = pop["n"] * (pop["act"] or 0.3) * flag["rollout_percent"] / 100.0
    if flag["exposure_trigger"] == "exposure":
        rate *= EXPOSURE_RATE
    return rate


def expected_duration_days(n_total: int, flag: dict) -> float | None:
    """Days until n_total units have entered, at the current entry rate.
    Rough: first-day cohorts are biggest (most-active users enter first), so
    this is an optimistic-to-fair estimate — good enough for planning."""
    rate = measured_entry_rate(flag) or estimated_entry_rate(flag)
    if rate <= 0:
        return None
    already = db.query_one(
        f"SELECT COUNT(*) AS n FROM "
        f"{'exposures' if flag['exposure_trigger'] == 'exposure' else 'assignments'} "
        f"WHERE flag_id = ?", (flag["id"],))["n"]
    remaining = max(n_total - already, 0)
    return round(remaining / rate, 1)
