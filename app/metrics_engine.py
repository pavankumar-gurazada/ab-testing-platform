"""Metrics engine: turn the raw event stream into a per-unit analysis frame.

compute_metric() returns one row per experiment unit:

    unit_id, variant, y                      (proportion / continuous)
    unit_id, variant, num, den               (ratio — analyzed via delta method)
    ... plus x_pre                           (CUPED covariate, when available)

Pipeline (steps 1-3 in SQL, 4-6 in pandas):
  1. population: units + variant + entry time, per the flag's exposure trigger
  2. join events inside the attribution window [entry, entry + window], capped
     at the analysis time
  3. aggregate per unit per the metric spec
  4. missing values: 'zero' fills non-converting units, 'exclude' drops them
  5. winsorize: cap the value at the pooled winsorize_pct quantile
  6. attach the pre-period covariate matching the metric's event (CUPED)
"""

import numpy as np
import pandas as pd

from . import db

# Which pre-period covariate pairs with which event stream (for CUPED).
PRE_COVARIATE_FOR_EVENT = {
    "lesson_complete": "pre_lessons_completed",
    "quiz_submit": "pre_quiz_score",
    "page_view": None,                # no latency pre-covariate collected
    "enrollment": "pre_lessons_completed",
    "course_complete": "pre_lessons_completed",
}

AGG_SQL = {"any": "MAX(1)", "count": "COUNT(*)", "sum": "SUM(e.value)", "mean": "AVG(e.value)"}


def _population_sql(exposure_trigger: str) -> str:
    """Units in the experiment with their variant and entry time. Entry is the
    FIRST assignment or exposure (per the flag's trigger) at day >= start_day."""
    table = "exposures" if exposure_trigger == "exposure" else "assignments"
    return f"""
        SELECT unit_id, user_id, variant_key AS variant, MIN(sim_time) AS entry_time
        FROM {table}
        WHERE flag_id = :flag_id AND sim_day >= :start_day AND sim_day <= :as_of_day
        GROUP BY unit_id
    """


def _aggregate_sql(event_name: str, agg: str, exposure_trigger: str, col: str) -> str:
    """Per-unit aggregate of one event stream inside the attribution window.

    Events are attached to units through user_id: metrics are computed over
    the events of the user who owns the unit (for user-randomized flags the
    unit IS the user)."""
    return f"""
        SELECT p.unit_id, {AGG_SQL[agg]} AS {col}
        FROM ({_population_sql(exposure_trigger)}) p
        JOIN events e
          ON e.user_id = p.user_id
         AND e.event_name = :{col}_event
         AND e.sim_time >= p.entry_time
         AND e.sim_time <= MIN(
               COALESCE(p.entry_time + :window_days, 1e18),
               :as_of_time)
        GROUP BY p.unit_id
    """


def compute_metric(metric: dict, flag: dict, start_day: int, as_of_day: int) -> pd.DataFrame:
    """Build the analysis frame for one metric of one experiment/flag."""
    params = {
        "flag_id": flag["id"],
        "start_day": start_day,
        "as_of_day": as_of_day,
        "window_days": metric["attribution_window_days"],
        # events up to the END of as_of_day (sim_time is day + fraction)
        "as_of_time": as_of_day + 1.0,
        "num_event": metric["numerator_event"],
        "den_event": metric["denominator_event"],
    }

    with db.get_conn() as conn:
        pop = pd.read_sql_query(_population_sql(flag["exposure_trigger"]), conn, params=params)
        num = pd.read_sql_query(
            _aggregate_sql(metric["numerator_event"], metric["numerator_agg"],
                           flag["exposure_trigger"], "num"), conn, params=params)
        den = None
        if metric["type"] == "ratio":
            den = pd.read_sql_query(
                _aggregate_sql(metric["denominator_event"], metric["denominator_agg"],
                               flag["exposure_trigger"], "den"), conn, params=params)
        pre = pd.read_sql_query(
            "SELECT user_id, pre_lessons_completed, pre_watch_time, pre_quiz_score FROM users",
            conn)

    df = pop.merge(num, on="unit_id", how="left")
    if den is not None:
        df = df.merge(den, on="unit_id", how="left")

    # -- 4. missing values ---------------------------------------------------
    if metric["missing_value_policy"] == "zero":
        df["num"] = df["num"].fillna(0.0)
        if "den" in df:
            df["den"] = df["den"].fillna(0.0)
    else:  # 'exclude': only units with at least one numerator event count
        df = df.dropna(subset=["num"])
        if "den" in df:
            df["den"] = df["den"].fillna(0.0)

    # -- 5. winsorize (pooled quantile, both arms together) --------------------
    n_capped = 0
    if metric["winsorize_pct"] and metric["type"] != "proportion" and len(df):
        cap = df["num"].quantile(metric["winsorize_pct"])
        n_capped = int((df["num"] > cap).sum())
        df["num"] = df["num"].clip(upper=cap)

    # -- 6. CUPED covariate ----------------------------------------------------
    pre_col = PRE_COVARIATE_FOR_EVENT.get(metric["numerator_event"])
    if pre_col:
        # watch-time metrics pair better with pre_watch_time than a count
        if metric["numerator_agg"] in ("sum", "mean") and metric["numerator_event"] == "lesson_complete":
            pre_col = "pre_watch_time"
        df = df.merge(pre[["user_id", pre_col]].rename(columns={pre_col: "x_pre"}),
                      on="user_id", how="left")
    else:
        df["x_pre"] = np.nan

    if metric["type"] == "ratio":
        out = df[["unit_id", "variant", "num", "den", "x_pre"]].copy()
    else:
        out = df[["unit_id", "variant", "num", "x_pre"]].rename(columns={"num": "y"})
    out.attrs["n_capped"] = n_capped
    return out


def measure_baseline(metric: dict) -> tuple[float | None, float | None]:
    """(mean, sd) of the metric over the PRE-PERIOD (sim_day < 0), across all
    users — used to seed sample-size calculations before the experiment runs.

    Ratio metrics are linearized (u_i = (num_i - R*den_i) / mean(den)), whose
    variance is exactly the delta-method variance, so the returned sd plugs
    straight into the continuous sample-size formula.
    """
    with db.get_conn() as conn:
        df = pd.read_sql_query(f"""
            SELECT u.user_id,
                   {AGG_SQL[metric['numerator_agg']].replace('e.value', 'e.value')} AS num
            FROM users u
            JOIN events e ON e.user_id = u.user_id
             AND e.event_name = :ev AND e.sim_day < 0
            GROUP BY u.user_id
        """, conn, params={"ev": metric["numerator_event"]})
        n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        den_df = None
        if metric["type"] == "ratio":
            den_df = pd.read_sql_query(f"""
                SELECT u.user_id, {AGG_SQL[metric['denominator_agg']]} AS den
                FROM users u
                JOIN events e ON e.user_id = u.user_id
                 AND e.event_name = :ev AND e.sim_day < 0
                GROUP BY u.user_id
            """, conn, params={"ev": metric["denominator_event"]})

    if not n_users or df.empty:
        return None, None

    y = df["num"]
    if metric["missing_value_policy"] == "zero":
        y = pd.concat([y, pd.Series([0.0] * (n_users - len(df)))], ignore_index=True)
    if metric["winsorize_pct"] and metric["type"] != "proportion":
        y = y.clip(upper=y.quantile(metric["winsorize_pct"]))

    if metric["type"] == "proportion":
        p = len(df) / n_users     # share of users with >= 1 numerator event
        return round(p, 4), round((p * (1 - p)) ** 0.5, 4)
    if metric["type"] == "ratio":
        m = df.merge(den_df, on="user_id", how="outer").fillna(0.0)
        if m["den"].sum() == 0:
            return None, None
        r = m["num"].sum() / m["den"].sum()
        u = (m["num"] - r * m["den"]) / m["den"].mean()
        return round(float(r), 4), round(float(u.std(ddof=1)), 4)
    return round(float(y.mean()), 4), round(float(y.std(ddof=1)), 4)


def variant_summary(frame: pd.DataFrame, metric_type: str) -> pd.DataFrame:
    """Human-readable per-variant summary for UI tables."""
    if metric_type == "ratio":
        g = frame.groupby("variant").agg(n=("unit_id", "count"), num=("num", "sum"),
                                         den=("den", "sum"))
        g["value"] = g["num"] / g["den"].replace(0, pd.NA)
    else:
        g = frame.groupby("variant").agg(n=("unit_id", "count"), value=("y", "mean"))
    return g.reset_index()
