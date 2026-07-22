"""Assignment service — what a client SDK would call.

  GET  /api/assign  -> resolve a variant (JSON), with the reason exposed
  POST /api/expose  -> log that the unit actually saw the feature
  GET  /flags/{id}/debug -> interactive assignment debugger page
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .. import assignment, db

router = APIRouter(tags=["assign"])


def _sim_now() -> tuple[int, float]:
    day = db.query_one("SELECT current_sim_day FROM sim_state WHERE id=1")["current_sim_day"]
    return day, float(day)


def _attrs_for_unit(unit_id: str) -> dict:
    """Look up a simulated user's attributes; unknown units get defaults so the
    debugger can also probe arbitrary ids."""
    row = db.query_one(
        "SELECT * FROM users WHERE user_id = ? OR device_id = ?", (unit_id, unit_id))
    if row:
        return {"user_id": row["user_id"], "platform": row["platform"], "client": row["client"],
                "app_version": row["app_version"], "is_qa": bool(row["is_qa"])}
    return {"user_id": unit_id, "platform": "web", "client": "browser",
            "app_version": "4.0.0", "is_qa": False}


@router.get("/api/assign")
def api_assign(flag_key: str, unit_id: str, persist: bool = True):
    flag = db.query_one("SELECT * FROM flags WHERE key = ?", (flag_key,))
    if not flag:
        return JSONResponse({"error": f"no flag with key {flag_key!r}"}, status_code=404)
    day, t = _sim_now()
    res = assignment.get_variant(dict(flag), unit_id, _attrs_for_unit(unit_id),
                                 sim_day=day, sim_time=t, persist=persist)
    return {"flag_key": flag_key, "unit_id": unit_id, "variant": res.variant_key,
            "reason": res.reason, "detail": res.detail,
            "rollout_bucket": res.rollout_bucket, "variant_bucket": res.variant_bucket}


@router.post("/api/expose")
def api_expose(flag_key: str, unit_id: str):
    flag = db.query_one("SELECT * FROM flags WHERE key = ?", (flag_key,))
    if not flag:
        return JSONResponse({"error": f"no flag with key {flag_key!r}"}, status_code=404)
    day, t = _sim_now()
    attrs = _attrs_for_unit(unit_id)
    res = assignment.get_variant(dict(flag), unit_id, attrs, sim_day=day, sim_time=t)
    assignment.record_exposure(dict(flag), unit_id, attrs.get("user_id"),
                               res.variant_key, day, t)
    return {"flag_key": flag_key, "unit_id": unit_id, "variant": res.variant_key,
            "exposed": True}


@router.get("/flags/{flag_id}/debug", response_class=HTMLResponse)
def debug_page(request: Request, flag_id: int, unit_id: str = ""):
    from ..main import render
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag_id,)))
    result = None
    attrs = None
    if unit_id:
        attrs = _attrs_for_unit(unit_id)
        day, t = _sim_now()
        # persist=False: the debugger explains, it never enrolls.
        result = assignment.get_variant(flag, unit_id, attrs, sim_day=day, sim_time=t,
                                        persist=False)
    return render(request, "flags/debug.html", flag=flag, unit_id=unit_id,
                  attrs=attrs, result=result)
