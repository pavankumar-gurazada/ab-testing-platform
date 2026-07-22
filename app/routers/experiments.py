"""Experiment service: setup, metric roles, statistical design, lifecycle.

An experiment attaches inference to a flag: hypothesis, test type, metric
roles (success / guardrail / supporting), and the sequential design (looks,
spending function, target n). The flag delivers; the experiment analyzes.
"""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, metrics_engine
from ..stats import sample_size, sequential

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _exp(exp_id: int) -> dict:
    return dict(db.query_one("SELECT * FROM experiments WHERE id = ?", (exp_id,)))


def _sim_day() -> int:
    return db.query_one("SELECT current_sim_day FROM sim_state")["current_sim_day"]


@router.get("", response_class=HTMLResponse)
def list_experiments(request: Request):
    from ..main import render
    exps = db.query(
        "SELECT e.*, f.name AS flag_name, f.key AS flag_key FROM experiments e "
        "JOIN flags f ON f.id = e.flag_id ORDER BY e.id DESC")
    flags = db.query("SELECT * FROM flags ORDER BY id")
    return render(request, "experiments/list.html", experiments=exps, flags=flags)


@router.post("")
def create_experiment(flag_id: int = Form(), name: str = Form(),
                      hypothesis: str = Form(""), test_type: str = Form("difference"),
                      noninferiority_margin: str = Form(""),
                      alpha: float = Form(0.05), power: float = Form(0.8),
                      planned_looks: int = Form(1),
                      spending_function: str = Form("obrien_fleming"),
                      use_cuped: bool = Form(False)):
    exp_id = db.execute(
        "INSERT INTO experiments (flag_id, name, hypothesis, test_type, "
        " noninferiority_margin, alpha, power, planned_looks, spending_function, use_cuped) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (flag_id, name, hypothesis, test_type,
         float(noninferiority_margin) if noninferiority_margin else None,
         alpha, power, max(1, min(planned_looks, 10)), spending_function, int(use_cuped)))
    return RedirectResponse(f"/experiments/{exp_id}", status_code=303)


@router.get("/{exp_id}", response_class=HTMLResponse)
def experiment_detail(request: Request, exp_id: int):
    from ..main import render
    exp = _exp(exp_id)
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (exp["flag_id"],)))
    attached = db.query(
        "SELECT em.*, m.name, m.key, m.type, m.direction FROM experiment_metrics em "
        "JOIN metrics m ON m.id = em.metric_id WHERE em.experiment_id = ? "
        "ORDER BY CASE em.role WHEN 'success' THEN 0 WHEN 'guardrail' THEN 1 ELSE 2 END",
        (exp_id,))
    all_metrics = db.query("SELECT * FROM metrics ORDER BY name")
    effects = db.query(
        "SELECT * FROM sim_effects WHERE flag_id = ? ORDER BY variant_key", (exp["flag_id"],))
    looks_taken = db.query_one(
        "SELECT COUNT(*) AS n FROM analysis_looks WHERE experiment_id = ?", (exp_id,))["n"]

    # primary success metric's measured baseline, to prefill the design form
    primary = next((dict(m) for m in attached if m["role"] == "success"), None)
    baseline = sd = None
    if primary:
        metric = dict(db.query_one("SELECT * FROM metrics WHERE id = ?", (primary["metric_id"],)))
        baseline, sd = metrics_engine.measure_baseline(metric)

    # boundary preview for the planned design
    boundary_preview = sequential.boundaries(
        [i / exp["planned_looks"] for i in range(1, exp["planned_looks"] + 1)],
        exp["alpha"], exp["spending_function"]) if exp["planned_looks"] else []

    return render(request, "experiments/detail.html", exp=exp, flag=flag,
                  attached=attached, all_metrics=all_metrics, effects=effects,
                  baseline=baseline, baseline_sd=sd, looks_taken=looks_taken,
                  boundary_preview=boundary_preview, primary=primary)


@router.post("/{exp_id}/metrics")
def attach_metric(exp_id: int, metric_id: int = Form(), role: str = Form()):
    db.execute(
        "INSERT INTO experiment_metrics (experiment_id, metric_id, role) VALUES (?,?,?) "
        "ON CONFLICT (experiment_id, metric_id) DO UPDATE SET role = excluded.role",
        (exp_id, metric_id, role))
    return RedirectResponse(f"/experiments/{exp_id}", status_code=303)


@router.post("/{exp_id}/metrics/{em_id}/delete")
def detach_metric(exp_id: int, em_id: int):
    db.execute("DELETE FROM experiment_metrics WHERE id = ? AND experiment_id = ?",
               (em_id, exp_id))
    return RedirectResponse(f"/experiments/{exp_id}", status_code=303)


@router.post("/{exp_id}/design")
def design(exp_id: int, mde: float = Form(), baseline: float = Form(),
           sd: str = Form("")):
    """Compute and store target n per arm and the expected duration."""
    exp = _exp(exp_id)
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (exp["flag_id"],)))
    primary = db.query_one(
        "SELECT m.* FROM experiment_metrics em JOIN metrics m ON m.id = em.metric_id "
        "WHERE em.experiment_id = ? AND em.role = 'success' LIMIT 1", (exp_id,))
    if not primary:
        return RedirectResponse(f"/experiments/{exp_id}", status_code=303)

    n = sample_size.n_per_arm(
        primary["type"], baseline, mde, exp["alpha"], exp["power"],
        sd=float(sd) if sd else None,
        planned_looks=exp["planned_looks"], spending=exp["spending_function"])
    duration = sample_size.expected_duration_days(2 * n, flag)
    db.execute("UPDATE experiments SET mde = ?, target_n_per_arm = ?, "
               "expected_duration_days = ? WHERE id = ?", (mde, n, duration, exp_id))
    return RedirectResponse(f"/experiments/{exp_id}", status_code=303)


@router.post("/{exp_id}/start")
def start(exp_id: int):
    """Start the experiment: flips the flag to 'experiment' state so analysis
    entry begins now. Analysis population = units entering from this day on."""
    exp = _exp(exp_id)
    if exp["state"] != "draft" or not exp["target_n_per_arm"]:
        return RedirectResponse(f"/experiments/{exp_id}", status_code=303)
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (exp["flag_id"],)))
    if flag["state"] in ("qa", "rollout"):
        db.execute("UPDATE flags SET state = 'experiment' WHERE id = ?", (flag["id"],))
        db.execute("INSERT INTO rollout_history (flag_id, action, sim_day, note) "
                   "VALUES (?,?,?,?)", (flag["id"], "set_state", _sim_day(),
                                        f"experiment '{exp['name']}' started"))
    db.execute("UPDATE experiments SET state = 'running', start_sim_day = ? WHERE id = ?",
               (_sim_day(), exp_id))
    return RedirectResponse(f"/experiments/{exp_id}/analysis", status_code=303)


@router.post("/{exp_id}/stop")
def stop(exp_id: int):
    db.execute("UPDATE experiments SET state = 'aborted', end_sim_day = ? "
               "WHERE id = ? AND state = 'running'", (_sim_day(), exp_id))
    return RedirectResponse(f"/experiments/{exp_id}", status_code=303)
