"""Simulator service: population init, clock advance, true effects, anomalies."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db
from ..simulator import clock, runner

router = APIRouter(prefix="/simulator", tags=["simulator"])


@router.get("", response_class=HTMLResponse)
def simulator_page(request: Request):
    from ..main import render
    state = clock.get_state()
    flags = db.query("SELECT * FROM flags ORDER BY id")
    effects = db.query(
        "SELECT se.*, f.name AS flag_name, f.key AS flag_key FROM sim_effects se "
        "JOIN flags f ON f.id = se.flag_id ORDER BY se.flag_id, se.variant_key")
    daily = db.query(
        "SELECT sim_day, COUNT(*) AS n_events, COUNT(DISTINCT user_id) AS n_users "
        "FROM events GROUP BY sim_day ORDER BY sim_day DESC LIMIT 21")
    pre_stats = db.query_one(
        "SELECT COUNT(*) AS n, AVG(pre_lessons_completed) AS avg_lessons, "
        "       AVG(pre_watch_time) AS avg_watch "
        "FROM users WHERE pre_lessons_completed IS NOT NULL")
    return render(request, "simulator.html", state=state, flags=flags, effects=effects,
                  daily=daily, pre_stats=pre_stats,
                  parameters=["lesson_complete_uplift_pp", "watch_time_multiplier",
                              "quiz_score_shift", "latency_add_ms", "activity_multiplier"])


@router.post("/init")
def init(population_size: int = Form(20000), seed: int = Form(42),
         pre_period_days: int = Form(14)):
    runner.init_simulation(population_size, seed, pre_period_days)
    return RedirectResponse("/simulator", status_code=303)


@router.post("/advance")
def advance(days: int = Form(1)):
    runner.advance_days(max(1, min(days, 120)))
    return RedirectResponse("/simulator", status_code=303)


@router.post("/effects")
def set_effect(flag_id: int = Form(), variant_key: str = Form("treatment"),
               parameter: str = Form(), value: float = Form()):
    db.execute(
        "INSERT INTO sim_effects (flag_id, variant_key, parameter, value) VALUES (?,?,?,?) "
        "ON CONFLICT (flag_id, variant_key, parameter) DO UPDATE SET value = excluded.value",
        (flag_id, variant_key, parameter, value))
    return RedirectResponse("/simulator", status_code=303)


@router.post("/effects/{effect_id}/delete")
def delete_effect(effect_id: int):
    db.execute("DELETE FROM sim_effects WHERE id = ?", (effect_id,))
    return RedirectResponse("/simulator", status_code=303)


@router.post("/anomaly")
def set_anomaly(mode: str = Form("none")):
    clock.set_state(anomaly_mode=mode)
    return RedirectResponse("/simulator", status_code=303)


@router.post("/reset")
def reset():
    clock.reset_all()
    return RedirectResponse("/simulator", status_code=303)
