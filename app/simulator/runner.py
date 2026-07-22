"""The daily simulation loop: assignment -> exposure -> effects -> events.

Reproducibility: each simulated day gets its own RNG substream seeded by
(global_seed, day), so simulating 10 days in one call or in ten calls yields
identical data.

Exposure model: a user who visits on a given day "touches the feature
surface" (opens the lesson player) with probability EXPOSURE_RATE. Only on
touch days do treatment effects apply — the surface IS the feature — and the
first touch is logged as the unit's exposure. This is exactly why
exposure-triggered analysis is sharper than assignment-triggered: assigned
units that never touch the surface dilute the measured effect.
"""

import numpy as np

from .. import db
from ..assignment import resolve, load_flag_config
from . import behavior, clock
from .population import generate_population

EXPOSURE_RATE = 0.7        # P(an active user's day touches the feature surface)
SRM_BUG_DROP = 0.25        # srm_bug anomaly: fraction of treatment records lost
                           # (story: the treatment build crashes and loses telemetry;
                           #  the users still GET the treatment — only the records vanish)

LIVE_STATES = ("qa", "rollout", "experiment", "launched", "rolled_back")


def init_simulation(population_size: int, seed: int, pre_period_days: int) -> dict:
    clock.reset_all()
    with db.get_conn() as conn:
        conn.execute("DELETE FROM users")
    generate_population(population_size, seed)
    clock.set_state(rng_seed=seed, population_size=population_size,
                    pre_period_days=pre_period_days)
    if pre_period_days > 0:
        run_pre_period(pre_period_days)
    return clock.get_state()


def _load_users() -> list[dict]:
    return [dict(u) for u in db.query("SELECT * FROM users")]


def _attrs(user: dict) -> dict:
    return {"user_id": user["user_id"], "platform": user["platform"],
            "client": user["client"], "app_version": user["app_version"],
            "is_qa": bool(user["is_qa"])}


def _unit_id(user: dict, flag: dict, day: int) -> str:
    """Map the flag's randomization unit onto this user-day.

    Session units use the day's first session id — a simplification (one
    resolution per user-day instead of per session) that still demonstrates
    the key property: the unit CHANGES every day, so the same user flips
    between variants across days. Session randomization + user-level metrics
    is a classic footgun; this makes it observable.
    """
    if flag["randomization_unit"] == "device":
        return user["device_id"]
    if flag["randomization_unit"] == "session":
        return f"{user['user_id']}:s{day}:0"
    return user["user_id"]


def _load_funnel_state(users: list[dict]) -> dict:
    """Cumulative per-user funnel state (enrolled? lesson completions so far),
    rebuilt from the events table so advances are resumable."""
    state = {u["user_id"]: {"enrolled": False, "completions": 0, "enrollments": 0} for u in users}
    counts = db.query(
        "SELECT user_id, "
        "  SUM(event_name = 'enrollment')      AS n_enroll, "
        "  SUM(event_name = 'course_complete') AS n_courses, "
        "  SUM(event_name = 'lesson_complete') AS n_lessons "
        "FROM events GROUP BY user_id")
    for row in counts:
        if row["user_id"] in state:
            # An enrollment is 'open' until closed by a course_complete.
            state[row["user_id"]]["enrolled"] = row["n_enroll"] > row["n_courses"]
            state[row["user_id"]]["completions"] = row["n_lessons"]
    return state


def _load_flag_context() -> list[dict]:
    """Everything the day loop needs per flag, loaded once per advance call."""
    ctx = []
    for f in db.query(f"SELECT * FROM flags WHERE state IN {LIVE_STATES!r}"):
        flag = dict(f)
        variants, rules = load_flag_config(flag["id"])
        effects = {}   # variant_key -> {parameter: value}
        for e in db.query("SELECT * FROM sim_effects WHERE flag_id = ?", (flag["id"],)):
            effects.setdefault(e["variant_key"], {})[e["parameter"]] = e["value"]
        sticky = {}
        if flag["randomization_unit"] != "session":     # sessions are never sticky
            sticky = {a["unit_id"]: dict(a) for a in db.query(
                "SELECT unit_id, variant_key, rollout_bucket, variant_bucket "
                "FROM assignments WHERE flag_id = ?", (flag["id"],))}
        exposed = {r["unit_id"] for r in db.query(
            "SELECT unit_id FROM exposures WHERE flag_id = ?", (flag["id"],))}
        ctx.append({"flag": flag, "variants": variants, "rules": rules,
                    "effects": effects, "sticky": sticky, "exposed": exposed})
    return ctx


def _simulate_day(day: int, users: list[dict], funnel: dict, flag_ctx: list[dict],
                  seed: int, anomaly: str) -> dict:
    """Simulate one day; returns row batches for bulk insert."""
    rng = np.random.default_rng([seed, day & 0x7FFFFFFF, 1 if day < 0 else 0])
    new_events, new_assignments, new_exposures = [], [], []

    for user in users:
        # -- activity: does the user visit today? -----------------------
        p_active = user["activity"]
        for fc in flag_ctx:
            # activity effects only influence users already exposed to the variant
            unit = _unit_id(user, fc["flag"], day)
            if unit in fc["exposed"]:
                sticky = fc["sticky"].get(unit)
                if sticky:
                    mult = fc["effects"].get(sticky["variant_key"], {}).get("activity_multiplier")
                    if mult:
                        p_active = min(1.0, p_active * mult)
        if rng.random() >= p_active:
            continue

        # -- flag resolution + today's treatment effects -----------------
        effects_today: dict = {}
        for fc in flag_ctx:
            flag = fc["flag"]
            unit = _unit_id(user, flag, day)
            res = resolve(flag, fc["variants"], fc["rules"], unit, _attrs(user),
                          sticky=fc["sticky"].get(unit))
            touched = rng.random() < EXPOSURE_RATE
            drop_record = (anomaly == "srm_bug"
                           and res.variant_key != "control"
                           and rng.random() < SRM_BUG_DROP)

            if res.enrolled:
                if flag["randomization_unit"] != "session":
                    fc["sticky"][unit] = {"variant_key": res.variant_key,
                                          "rollout_bucket": res.rollout_bucket,
                                          "variant_bucket": res.variant_bucket}
                if not drop_record:
                    new_assignments.append(
                        (flag["id"], flag["randomization_unit"], unit, user["user_id"],
                         res.variant_key, res.rollout_bucket, res.variant_bucket,
                         day, float(day)))

            in_variant_population = res.reason in ("sticky", "new_assignment", "launched")
            if touched and in_variant_population:
                if res.reason != "launched" and unit not in fc["exposed"]:
                    fc["exposed"].add(unit)
                    if not drop_record:
                        new_exposures.append((flag["id"], unit, user["user_id"],
                                              res.variant_key, day, float(day)))
                # Effects apply on touch days only (the surface IS the feature).
                effects_today.update(fc["effects"].get(res.variant_key, {}))
                if anomaly == "guardrail_degrade" and res.variant_key != "control":
                    effects_today["latency_add_ms"] = \
                        effects_today.get("latency_add_ms", 0.0) + 150.0
                    effects_today["lesson_complete_uplift_pp"] = \
                        effects_today.get("lesson_complete_uplift_pp", 0.0) - 0.01

        new_events.extend(
            behavior.simulate_user_day(user, day, effects_today, rng,
                                       funnel[user["user_id"]]))

    return {"events": new_events, "assignments": new_assignments,
            "exposures": new_exposures}


def _flush(batch: dict) -> None:
    with db.get_conn() as conn:
        conn.executemany(
            "INSERT INTO events (user_id, session_id, event_name, value, sim_day, sim_time) "
            "VALUES (?,?,?,?,?,?)", batch["events"])
        conn.executemany(
            "INSERT OR IGNORE INTO assignments (flag_id, unit_type, unit_id, user_id, "
            " variant_key, rollout_bucket, variant_bucket, sim_day, sim_time) "
            "VALUES (?,?,?,?,?,?,?,?,?)", batch["assignments"])
        conn.executemany(
            "INSERT INTO exposures (flag_id, unit_id, user_id, variant_key, sim_day, sim_time) "
            "VALUES (?,?,?,?,?,?)", batch["exposures"])


def advance_days(n: int) -> dict:
    """Simulate the next n days of traffic. Returns a summary."""
    state = clock.get_state()
    users = _load_users()
    funnel = _load_funnel_state(users)
    flag_ctx = _load_flag_context()
    totals = {"events": 0, "assignments": 0, "exposures": 0}

    start = state["current_sim_day"]
    for day in range(start, start + n):
        batch = _simulate_day(day, users, funnel, flag_ctx,
                              state["rng_seed"], state["anomaly_mode"])
        _flush(batch)
        for k in totals:
            totals[k] += len(batch[k])
    clock.set_state(current_sim_day=start + n)
    totals["from_day"], totals["to_day"] = start, start + n - 1
    return totals


def run_pre_period(days: int) -> dict:
    """Simulate the pre-experiment period on negative days with NO flags live,
    then store per-user aggregates as CUPED covariates."""
    state = clock.get_state()
    users = _load_users()
    funnel = _load_funnel_state(users)
    totals = {"events": 0}
    for day in range(-days, 0):
        batch = _simulate_day(day, users, funnel, [], state["rng_seed"], "none")
        _flush(batch)
        totals["events"] += len(batch["events"])

    # Aggregate pre-period behavior into the CUPED covariate columns.
    # One GROUP BY + UPDATE..FROM (a correlated per-row subquery is ~100x slower).
    with db.get_conn() as conn:
        conn.execute("""
            UPDATE users SET
              pre_lessons_completed = agg.n_lessons,
              pre_watch_time = agg.watch,
              pre_quiz_score = agg.quiz
            FROM (SELECT user_id,
                         SUM(event_name = 'lesson_complete')                    AS n_lessons,
                         SUM(CASE WHEN event_name = 'lesson_complete'
                                  THEN value ELSE 0 END)                        AS watch,
                         AVG(CASE WHEN event_name = 'quiz_submit'
                                  THEN value END)                               AS quiz
                  FROM events WHERE sim_day < 0 GROUP BY user_id) AS agg
            WHERE agg.user_id = users.user_id
        """)
        # users with no pre-period events at all: zero counts, quiz stays NULL
        conn.execute("UPDATE users SET pre_lessons_completed = 0, pre_watch_time = 0 "
                     "WHERE pre_lessons_completed IS NULL")
    clock.set_state(pre_period_days=days)
    return totals
