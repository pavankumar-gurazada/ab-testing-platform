"""Flag service: CRUD, targeting rules, variants, and the rollout lifecycle
(draft -> qa -> rollout -> experiment -> launched, with pause/rollback).

GET routes render HTML; POST routes are form handlers that redirect back
(post-redirect-get), so the UI is plain forms with no JavaScript.
"""

import json
import secrets

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db

router = APIRouter(prefix="/flags", tags=["flags"])

VALID_TRANSITIONS = {
    "draft": ["qa"],
    "qa": ["rollout", "rolled_back"],
    "rollout": ["experiment", "launched", "rolled_back"],
    "experiment": ["launched", "rolled_back"],
    "launched": [],
    "rolled_back": ["qa"],  # a fixed build can restart from QA
}


def _sim_day() -> int:
    return db.query_one("SELECT current_sim_day FROM sim_state WHERE id=1")["current_sim_day"]


def _log(flag_id: int, action: str, percent: float | None = None, note: str = ""):
    db.execute(
        "INSERT INTO rollout_history (flag_id, action, percent, sim_day, note) VALUES (?,?,?,?,?)",
        (flag_id, action, percent, _sim_day(), note))


def _flag(flag_id: int) -> dict:
    return dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag_id,)))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@router.get("", response_class=HTMLResponse)
def list_flags(request: Request):
    from ..main import render
    rows = db.query(
        "SELECT f.*, a.name AS app_name, "
        "  (SELECT COUNT(*) FROM assignments WHERE flag_id = f.id) AS n_assigned, "
        "  (SELECT COUNT(*) FROM exposures  WHERE flag_id = f.id) AS n_exposed "
        "FROM flags f JOIN applications a ON a.id = f.application_id ORDER BY f.id")
    apps = db.query("SELECT * FROM applications ORDER BY name")
    return render(request, "flags/list.html", flags=rows, applications=apps)


@router.get("/{flag_id}", response_class=HTMLResponse)
def flag_detail(request: Request, flag_id: int):
    from ..main import render
    flag = _flag(flag_id)
    variants = db.query("SELECT * FROM variants WHERE flag_id = ? ORDER BY is_control DESC, id", (flag_id,))
    rules = [dict(r) for r in db.query("SELECT * FROM targeting_rules WHERE flag_id = ?", (flag_id,))]
    for r in rules:
        r["value_display"] = json.loads(r["value"])
    history = db.query(
        "SELECT * FROM rollout_history WHERE flag_id = ? ORDER BY id DESC LIMIT 30", (flag_id,))
    counts = db.query_one(
        "SELECT (SELECT COUNT(*) FROM assignments WHERE flag_id=?) AS assigned, "
        "       (SELECT COUNT(*) FROM exposures  WHERE flag_id=?) AS exposed", (flag_id, flag_id))
    by_variant = db.query(
        "SELECT variant_key, COUNT(*) AS n FROM assignments WHERE flag_id = ? "
        "GROUP BY variant_key ORDER BY variant_key", (flag_id,))
    experiments = db.query("SELECT * FROM experiments WHERE flag_id = ? ORDER BY id DESC", (flag_id,))
    return render(request, "flags/detail.html", flag=flag, variants=variants, rules=rules,
                  history=history, counts=counts, by_variant=by_variant,
                  experiments=experiments,
                  next_states=VALID_TRANSITIONS.get(flag["state"], []))


# ---------------------------------------------------------------------------
# Mutations (forms)
# ---------------------------------------------------------------------------

@router.post("")
def create_flag(application_id: int = Form(), key: str = Form(), name: str = Form(),
                description: str = Form(""), randomization_unit: str = Form("user"),
                exposure_trigger: str = Form("exposure")):
    flag_id = db.execute(
        "INSERT INTO flags (application_id, key, name, description, salt, "
        " randomization_unit, exposure_trigger) VALUES (?,?,?,?,?,?,?)",
        (application_id, key, name, description, secrets.token_hex(8),
         randomization_unit, exposure_trigger))
    # Every flag starts with a standard control/treatment pair; editable after.
    db.execute("INSERT INTO variants (flag_id, key, name, is_control, weight) VALUES (?,?,?,?,?)",
               (flag_id, "control", "Control", 1, 1))
    db.execute("INSERT INTO variants (flag_id, key, name, is_control, weight) VALUES (?,?,?,?,?)",
               (flag_id, "treatment", "Treatment", 0, 1))
    _log(flag_id, "set_state", note="created (draft)")
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)


@router.post("/{flag_id}/variants")
def update_variant_weights(flag_id: int, request: Request,
                           control_weight: float = Form(1), treatment_weight: float = Form(1)):
    db.execute("UPDATE variants SET weight = ? WHERE flag_id = ? AND is_control = 1",
               (control_weight, flag_id))
    db.execute("UPDATE variants SET weight = ? WHERE flag_id = ? AND is_control = 0",
               (treatment_weight, flag_id))
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)


@router.post("/{flag_id}/rules")
def add_rule(flag_id: int, attribute: str = Form(), operator: str = Form(), value: str = Form()):
    # Normalize the form value into JSON: 'in' takes a comma list, is_qa a bool.
    if operator == "in":
        encoded = json.dumps([v.strip() for v in value.split(",") if v.strip()])
    elif attribute == "is_qa":
        encoded = json.dumps(value.lower() in ("true", "1", "yes"))
    else:
        encoded = json.dumps(value.strip())
    db.execute("INSERT INTO targeting_rules (flag_id, attribute, operator, value) VALUES (?,?,?,?)",
               (flag_id, attribute, operator, encoded))
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)


@router.post("/{flag_id}/rules/{rule_id}/delete")
def delete_rule(flag_id: int, rule_id: int):
    db.execute("DELETE FROM targeting_rules WHERE id = ? AND flag_id = ?", (rule_id, flag_id))
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)


@router.post("/{flag_id}/state")
def set_state(flag_id: int, state: str = Form()):
    flag = _flag(flag_id)
    if state not in VALID_TRANSITIONS.get(flag["state"], []):
        return RedirectResponse(f"/flags/{flag_id}", status_code=303)
    db.execute("UPDATE flags SET state = ? WHERE id = ?", (state, flag_id))
    _log(flag_id, "set_state", note=f"{flag['state']} -> {state}")
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)


@router.post("/{flag_id}/rollout")
def set_rollout(flag_id: int, percent: float = Form()):
    percent = max(0.0, min(100.0, percent))
    db.execute("UPDATE flags SET rollout_percent = ? WHERE id = ?", (percent, flag_id))
    _log(flag_id, "set_percent", percent=percent)
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)


@router.post("/{flag_id}/pause")
def pause(flag_id: int):
    db.execute("UPDATE flags SET paused = 1 WHERE id = ?", (flag_id,))
    _log(flag_id, "pause", note="no new units will be enrolled")
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)


@router.post("/{flag_id}/resume")
def resume(flag_id: int):
    db.execute("UPDATE flags SET paused = 0 WHERE id = ?", (flag_id,))
    _log(flag_id, "resume")
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)


@router.post("/{flag_id}/rollback")
def rollback(flag_id: int):
    """Kill switch: everyone (including already-assigned units) gets control."""
    db.execute("UPDATE flags SET state = 'rolled_back', paused = 0 WHERE id = ?", (flag_id,))
    _log(flag_id, "rollback", note="kill switch: all traffic to control")
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)


@router.post("/{flag_id}/launch")
def launch(flag_id: int):
    """Ship it: treatment becomes the default experience for all users."""
    db.execute("UPDATE flags SET state = 'launched', rollout_percent = 100 WHERE id = ?", (flag_id,))
    _log(flag_id, "launch", percent=100, note="treatment is now the default")
    return RedirectResponse(f"/flags/{flag_id}", status_code=303)
