"""Simulated-clock state helpers.

current_sim_day is the next day to be simulated (day 0 = experiment-era start).
The CUPED pre-period runs on NEGATIVE days (-14..-1), so 'entered the
experiment on day >= 0' and 'pre-period behavior' never overlap.
"""

from .. import db


def get_state() -> dict:
    return dict(db.query_one("SELECT * FROM sim_state WHERE id = 1"))


def set_state(**kwargs) -> None:
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    db.execute(f"UPDATE sim_state SET {sets} WHERE id = 1", tuple(kwargs.values()))


def reset_all() -> None:
    """Wipe simulation output (population, events, assignments, analyses) but
    keep the configured entities: flags, variants, rules, metrics, experiments'
    definitions stay; their runtime state is reset to draft."""
    with db.get_conn() as conn:
        for table in ("events", "exposures", "assignments", "users",
                      "look_results", "analysis_looks", "rollout_history", "sim_effects"):
            conn.execute(f"DELETE FROM {table}")
        conn.execute("UPDATE experiments SET state='draft', start_sim_day=NULL, end_sim_day=NULL")
        conn.execute("UPDATE flags SET state='draft', rollout_percent=0, paused=0")
        conn.execute("UPDATE users SET pre_lessons_completed=NULL, pre_watch_time=NULL, "
                     "pre_quiz_score=NULL")
        conn.execute("UPDATE sim_state SET current_sim_day=0, anomaly_mode='none', "
                     "pre_period_days=0, population_size=0 WHERE id=1")
