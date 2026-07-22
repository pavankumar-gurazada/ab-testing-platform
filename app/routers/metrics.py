"""Metric definition service: the catalog of continuous / proportion / ratio
metrics computed over the event stream, plus a live per-variant preview."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, metrics_engine

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_class=HTMLResponse)
def list_metrics(request: Request):
    from ..main import render
    metrics = db.query("SELECT * FROM metrics ORDER BY id")
    events = db.query("SELECT DISTINCT event_name FROM events ORDER BY event_name")
    event_names = [e["event_name"] for e in events] or \
        ["enrollment", "lesson_start", "lesson_complete", "course_complete",
         "quiz_submit", "page_view"]
    apps = db.query("SELECT * FROM applications ORDER BY name")
    flags = db.query("SELECT * FROM flags ORDER BY id")
    return render(request, "metrics/list.html", metrics=metrics,
                  event_names=event_names, applications=apps, flags=flags)


@router.post("")
def create_metric(application_id: int = Form(), key: str = Form(), name: str = Form(),
                  type: str = Form(), numerator_event: str = Form(),
                  numerator_agg: str = Form("count"),
                  denominator_event: str = Form(""), denominator_agg: str = Form("count"),
                  attribution_window_days: str = Form(""),
                  missing_value_policy: str = Form("zero"),
                  winsorize_pct: str = Form(""), direction: str = Form("increase"),
                  description: str = Form("")):
    db.execute(
        "INSERT INTO metrics (application_id, key, name, type, numerator_event, "
        " numerator_agg, denominator_event, denominator_agg, attribution_window_days, "
        " missing_value_policy, winsorize_pct, direction, description) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (application_id, key, name, type, numerator_event,
         "any" if type == "proportion" else numerator_agg,
         denominator_event or None, denominator_agg if type == "ratio" else None,
         int(attribution_window_days) if attribution_window_days else None,
         missing_value_policy,
         float(winsorize_pct) if winsorize_pct else None, direction, description))
    return RedirectResponse("/metrics", status_code=303)


@router.get("/{metric_id}/preview", response_class=HTMLResponse)
def preview(request: Request, metric_id: int, flag_id: int):
    """Compute the metric for a flag's current population — the fastest way to
    see the engine (windows, missing policy, winsorization) at work."""
    from ..main import render
    metric = dict(db.query_one("SELECT * FROM metrics WHERE id = ?", (metric_id,)))
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag_id,)))
    as_of = db.query_one("SELECT current_sim_day FROM sim_state")["current_sim_day"]
    frame = metrics_engine.compute_metric(metric, flag, start_day=0, as_of_day=as_of)
    summary = metrics_engine.variant_summary(frame, metric["type"]) if len(frame) else None
    return render(request, "metrics/preview.html", metric=metric, flag=flag,
                  summary=summary.to_dict("records") if summary is not None else [],
                  n_units=len(frame), n_capped=frame.attrs.get("n_capped", 0), as_of=as_of)
