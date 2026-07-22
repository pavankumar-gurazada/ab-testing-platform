"""Analysis service: interim looks, the sample-size calculator, monitoring
and release analytics.

run_look() is the orchestrator where everything converges:

  SRM check -> per-metric frames (metrics engine) -> CUPED (if enabled)
  -> difference / non-inferiority / delta-method test -> compare the primary
  z against the look's alpha-spending boundary -> conditional power (futility)
  -> guardrail one-sided harm checks -> persist -> update experiment state.

Simplifications (all deliberate, all visible in the UI):
  * analysis assumes exactly two variants (control + one treatment)
  * guardrails are one-sided tests at fixed alpha=.05, no multiplicity
    adjustment across guardrails
  * non-inferiority experiments reuse the two-sided efficacy boundary for
    their one-sided z (slightly conservative)
  * CUPED is applied to proportion/continuous metrics with a covariate;
    ratio metrics are analyzed unadjusted
"""

import numpy as np
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, metrics_engine
from ..stats import cuped, sequential, srm as srm_mod
from ..stats import sample_size as ss
from ..stats.difference import diff_means, diff_proportions
from ..stats.intervals import repeated_ci, wald_ci
from ..stats.noninferiority import noninferiority_test
from ..stats.ratio import diff_ratios

router = APIRouter(tags=["analysis"])

GUARDRAIL_ALPHA = 0.05     # one-sided harm test per guardrail


# ---------------------------------------------------------------------------
# Core: one interim look
# ---------------------------------------------------------------------------

def _entry_counts(flag: dict, start_day: int, as_of_day: int) -> dict[str, int]:
    table = "exposures" if flag["exposure_trigger"] == "exposure" else "assignments"
    rows = db.query(
        f"SELECT variant_key, COUNT(DISTINCT unit_id) AS n FROM {table} "
        f"WHERE flag_id = ? AND sim_day BETWEEN ? AND ? GROUP BY variant_key",
        (flag["id"], start_day, as_of_day))
    return {r["variant_key"]: r["n"] for r in rows}


def _test_metric(metric: dict, frame, use_cuped: bool):
    """Run the right test for one metric; returns (EffectResult, cuped_info)."""
    ctrl = frame[frame["variant"] == "control"]
    treat = frame[frame["variant"] != "control"]
    cuped_info = {"applied": 0, "theta": None, "reduction": None}

    if metric["type"] == "ratio":
        res = diff_ratios(ctrl["num"].values, ctrl["den"].values,
                          treat["num"].values, treat["den"].values)
        return res, cuped_info

    y_c, y_t = ctrl["y"].astype(float).values, treat["y"].astype(float).values
    x_all = frame["x_pre"].astype(float).values
    has_covariate = np.isfinite(x_all).sum() > len(frame) * 0.5

    if use_cuped and has_covariate and len(frame) > 10:
        y_all = frame["y"].astype(float).values
        adj = cuped.cuped_adjust(y_all, x_all)
        is_ctrl = (frame["variant"] == "control").values
        res = diff_means(adj.y_adj[is_ctrl], adj.y_adj[~is_ctrl])
        cuped_info = {"applied": 1, "theta": adj.theta,
                      "reduction": adj.variance_reduction}
        return res, cuped_info

    if metric["type"] == "proportion":
        return diff_proportions(int(y_c.sum()), len(y_c), int(y_t.sum()), len(y_t)), cuped_info
    return diff_means(y_c, y_t), cuped_info


def run_look(exp_id: int) -> dict:
    exp = dict(db.query_one("SELECT * FROM experiments WHERE id = ?", (exp_id,)))
    if exp["state"] != "running":
        return {"error": "experiment is not running"}
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (exp["flag_id"],)))
    as_of = db.query_one("SELECT current_sim_day FROM sim_state")["current_sim_day"] - 1
    if as_of < exp["start_sim_day"]:
        return {"error": "no completed days since the experiment started — advance the simulator"}

    past = db.query("SELECT * FROM analysis_looks WHERE experiment_id = ? ORDER BY look_number",
                    (exp_id,))
    k, big_k = len(past) + 1, exp["planned_looks"]
    if k > big_k:
        return {"error": f"all {big_k} planned looks already taken"}

    # --- SRM first: if units are leaking, nothing downstream is trustworthy --
    counts = _entry_counts(flag, exp["start_sim_day"], as_of)
    weights = {v["key"]: v["weight"] for v in
               db.query("SELECT * FROM variants WHERE flag_id = ?", (flag["id"],))}
    n_total = sum(counts.values())
    srm_res = (srm_mod.srm_test(counts, weights)
               if len(counts) == len(weights) and n_total else None)

    # --- information fraction & this look's boundary -------------------------
    t_k = min(1.0, n_total / (2 * exp["target_n_per_arm"]))
    fractions = [lk["information_fraction"] for lk in past]
    if fractions and t_k <= fractions[-1]:
        t_k = min(1.0, fractions[-1] + 1e-3)      # no new information: nudge
    if k == big_k:
        t_k = 1.0                                  # final look closes the design
    fractions = fractions + [t_k]
    looks = sequential.boundaries(fractions, exp["alpha"], exp["spending_function"])
    this = looks[-1]

    # final-look boundary of the PLANNED schedule (for conditional power)
    planned = sequential.boundaries([i / big_k for i in range(1, big_k + 1)],
                                    exp["alpha"], exp["spending_function"])
    z_final = planned[-1].z_boundary

    # --- metrics --------------------------------------------------------------
    attached = db.query(
        "SELECT em.role, m.* FROM experiment_metrics em JOIN metrics m ON m.id = em.metric_id "
        "WHERE em.experiment_id = ? "
        "ORDER BY CASE em.role WHEN 'success' THEN 0 WHEN 'guardrail' THEN 1 ELSE 2 END",
        (exp_id,))

    results, primary_z, guardrail_breached = [], None, False
    for row in attached:
        metric, role = dict(row), row["role"]
        frame = metrics_engine.compute_metric(metric, flag, exp["start_sim_day"], as_of)
        if frame.empty or frame["variant"].nunique() < 2:
            continue
        res, cu = _test_metric(metric, frame, bool(exp["use_cuped"]) and role == "success")

        crossed = 0
        if role == "success":
            if exp["test_type"] == "noninferiority" and exp["noninferiority_margin"]:
                ni = noninferiority_test(res, exp["noninferiority_margin"],
                                         metric["direction"], exp["alpha"])
                z_for_boundary = ni.z          # one-sided vs the (conservative) boundary
            else:
                z_for_boundary = abs(res.z)
            crossed = int(z_for_boundary >= this.z_boundary)
            if primary_z is None:
                # signed toward 'good' for conditional power
                primary_z = res.z if metric["direction"] == "increase" else -res.z
                if exp["test_type"] == "noninferiority" and exp["noninferiority_margin"]:
                    primary_z = z_for_boundary
            ci_lo, ci_hi = repeated_ci(res.effect, res.se, this.z_boundary)
        else:
            ci_lo, ci_hi = wald_ci(res.effect, res.se, GUARDRAIL_ALPHA)
            if role == "guardrail":
                # one-sided harm check: is the metric significantly moving the BAD way?
                from scipy.stats import norm
                p_harm = (norm.cdf(res.z) if metric["direction"] == "increase"
                          else norm.sf(res.z))
                if p_harm < GUARDRAIL_ALPHA:
                    guardrail_breached = True
                    crossed = 1

        results.append({"metric": metric, "role": role, "res": res, "cuped": cu,
                        "crossed": crossed, "ci": (ci_lo, ci_hi)})

    # A look at full information is final even if fewer than K looks were taken
    # (all the alpha budget is spent at t = 1).
    is_final = (k == big_k) or (t_k >= 1.0)

    # --- futility (current-trend conditional power on the primary metric) ----
    cp = None
    if primary_z is not None and not is_final:
        cp = sequential.conditional_power(primary_z, t_k, z_final)

    # --- decision -------------------------------------------------------------
    success_crossed = any(r["crossed"] for r in results if r["role"] == "success")
    if guardrail_breached:
        decision = "guardrail_breach"          # recorded loudly; stopping is the human's call
    elif success_crossed:
        decision = "stop_efficacy"
    elif cp is not None and cp < sequential.FUTILITY_CP_THRESHOLD:
        decision = "stop_futility"
    elif is_final:
        decision = "complete"
    else:
        decision = "continue"

    # --- persist ----------------------------------------------------------------
    look_id = db.execute(
        "INSERT INTO analysis_looks (experiment_id, look_number, sim_day, "
        " information_fraction, alpha_spent_cumulative, alpha_spent_this_look, "
        " efficacy_z_boundary, futility_conditional_power, srm_chi2, srm_p, srm_flag, decision) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (exp_id, k, as_of, t_k, this.alpha_spent_cumulative, this.alpha_spent_this_look,
         this.z_boundary, cp,
         srm_res.chi2 if srm_res else None, srm_res.p if srm_res else None,
         int(srm_res.flagged) if srm_res else 0, decision))
    for r in results:
        e = r["res"]
        db.execute(
            "INSERT INTO look_results (look_id, metric_id, role, control_n, treatment_n, "
            " control_mean, treatment_mean, effect, se, z_stat, p_value, ci_low, ci_high, "
            " cuped_applied, cuped_theta, variance_reduction_pct, crossed_boundary) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (look_id, r["metric"]["id"], r["role"], e.control_n, e.treatment_n,
             e.control_mean, e.treatment_mean, e.effect, e.se, e.z, e.p,
             r["ci"][0], r["ci"][1], r["cuped"]["applied"], r["cuped"]["theta"],
             r["cuped"]["reduction"] * 100 if r["cuped"]["reduction"] is not None else None,
             r["crossed"]))

    new_state = {"stop_efficacy": "stopped_efficacy", "stop_futility": "stopped_futility",
                 "complete": "completed"}.get(decision)
    if new_state is None and is_final:
        # e.g. a guardrail breach on the final look: the design is over either way
        new_state = "stopped_efficacy" if success_crossed else "completed"
    if new_state:
        db.execute("UPDATE experiments SET state = ?, end_sim_day = ? WHERE id = ?",
                   (new_state, as_of, exp_id))
    return {"look_id": look_id, "decision": decision}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/experiments/{exp_id}/looks")
def take_look(exp_id: int):
    run_look(exp_id)
    return RedirectResponse(f"/experiments/{exp_id}/analysis", status_code=303)


@router.get("/experiments/{exp_id}/analysis", response_class=HTMLResponse)
def analysis_page(request: Request, exp_id: int):
    from ..main import render
    exp = dict(db.query_one("SELECT * FROM experiments WHERE id = ?", (exp_id,)))
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (exp["flag_id"],)))
    looks = [dict(lk) for lk in db.query(
        "SELECT * FROM analysis_looks WHERE experiment_id = ? ORDER BY look_number", (exp_id,))]
    look_results = {}
    for lk in looks:
        look_results[lk["id"]] = [dict(r) for r in db.query(
            "SELECT lr.*, m.name AS metric_name, m.direction, m.type AS metric_type "
            "FROM look_results lr JOIN metrics m ON m.id = lr.metric_id "
            "WHERE lr.look_id = ? ORDER BY CASE lr.role WHEN 'success' THEN 0 "
            "WHEN 'guardrail' THEN 1 ELSE 2 END", (lk["id"],))]
    effects = db.query("SELECT * FROM sim_effects WHERE flag_id = ?", (exp["flag_id"],))
    planned = sequential.boundaries(
        [i / exp["planned_looks"] for i in range(1, exp["planned_looks"] + 1)],
        exp["alpha"], exp["spending_function"])

    # data for the SVG boundary plot: realized looks + planned schedule
    latest = looks[-1] if looks else None
    latest_results = look_results.get(latest["id"], []) if latest else []
    primary = next((r for r in latest_results if r["role"] == "success"), None)
    return render(request, "experiments/analysis.html", exp=exp, flag=flag,
                  looks=looks, look_results=look_results, effects=effects,
                  planned=planned, latest=latest, latest_results=latest_results,
                  primary=primary)


@router.get("/flags/{flag_id}/monitoring", response_class=HTMLResponse)
def monitoring_page(request: Request, flag_id: int):
    """Continuous rollout monitoring: daily metric trends by variant, rolling
    SRM over the enrollment stream, guardrail status, rollout-stage timeline."""
    from ..main import render
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag_id,)))
    metrics = [dict(m) for m in db.query("SELECT * FROM metrics ORDER BY id")]

    # Daily per-variant trend per metric. Deliberately event-level (not the
    # full attribution-window engine): monitoring watches for drift and
    # breakage day by day; the experiment analysis owns rigorous inference.
    series = []
    for m in metrics:
        if m["numerator_agg"] in ("sum", "mean"):
            value_sql = "AVG(e.value)"
            label = f"daily avg {m['numerator_event']} value"
        else:
            value_sql = "COUNT(*) * 1.0 / COUNT(DISTINCT e.user_id)"
            label = f"daily {m['numerator_event']} per active user"
        rows = db.query(f"""
            SELECT e.sim_day AS day, a.variant_key AS variant, {value_sql} AS v
            FROM events e JOIN assignments a
              ON a.user_id = e.user_id AND a.flag_id = ?
            WHERE e.event_name = ? AND e.sim_day >= 0
            GROUP BY e.sim_day, a.variant_key ORDER BY e.sim_day
        """, (flag_id, m["numerator_event"]))
        if rows:
            by_variant: dict[str, list] = {}
            for r in rows:
                by_variant.setdefault(r["variant"], []).append((r["day"], r["v"]))
            vmax = max(r["v"] for r in rows)
            series.append({"metric": m, "label": label, "by_variant": by_variant,
                           "vmax": vmax or 1.0,
                           "dmin": min(r["day"] for r in rows),
                           "dmax": max(r["day"] for r in rows)})

    # Rolling SRM: cumulative entry counts per day -> chi-square p over time
    table = "exposures" if flag["exposure_trigger"] == "exposure" else "assignments"
    weights = {v["key"]: v["weight"] for v in
               db.query("SELECT * FROM variants WHERE flag_id = ?", (flag_id,))}
    daily_entries = db.query(f"""
        SELECT sim_day, variant_key, COUNT(*) AS n FROM {table}
        WHERE flag_id = ? AND sim_day >= 0 GROUP BY sim_day, variant_key ORDER BY sim_day
    """, (flag_id,))
    srm_series, cum = [], {k: 0 for k in weights}
    for day in sorted({r["sim_day"] for r in daily_entries}):
        for r in daily_entries:
            if r["sim_day"] == day:
                cum[r["variant_key"]] = cum.get(r["variant_key"], 0) + r["n"]
        if len([v for v in cum.values() if v]) == len(weights) and sum(cum.values()) > 50:
            res = srm_mod.srm_test(dict(cum), weights)
            srm_series.append({"day": day, "p": res.p, "flagged": res.flagged})

    history = db.query("SELECT * FROM rollout_history WHERE flag_id = ? ORDER BY sim_day",
                       (flag_id,))
    # latest guardrail verdicts from the most recent look of any experiment on the flag
    guardrails = db.query("""
        SELECT lr.*, m.name AS metric_name, m.direction FROM look_results lr
        JOIN metrics m ON m.id = lr.metric_id
        WHERE lr.role = 'guardrail' AND lr.look_id = (
            SELECT al.id FROM analysis_looks al
            JOIN experiments e ON e.id = al.experiment_id
            WHERE e.flag_id = ? ORDER BY al.sim_day DESC, al.id DESC LIMIT 1)
    """, (flag_id,))
    return render(request, "monitoring.html", flag=flag, series=series,
                  srm_series=srm_series, history=history, guardrails=guardrails)


@router.get("/experiments/{exp_id}/release", response_class=HTMLResponse)
def release_page(request: Request, exp_id: int):
    """Release analytics: the experiment's verdict plus before/after-launch
    population deltas once the flag has shipped."""
    from ..main import render
    exp = dict(db.query_one("SELECT * FROM experiments WHERE id = ?", (exp_id,)))
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (exp["flag_id"],)))
    final_look = db.query_one(
        "SELECT * FROM analysis_looks WHERE experiment_id = ? ORDER BY look_number DESC LIMIT 1",
        (exp_id,))
    final_results = db.query(
        "SELECT lr.*, m.name AS metric_name, m.direction FROM look_results lr "
        "JOIN metrics m ON m.id = lr.metric_id WHERE lr.look_id = ? "
        "ORDER BY CASE lr.role WHEN 'success' THEN 0 WHEN 'guardrail' THEN 1 ELSE 2 END",
        (final_look["id"],)) if final_look else []
    looks_summary = db.query(
        "SELECT COUNT(*) AS n, SUM(alpha_spent_this_look) AS spent FROM analysis_looks "
        "WHERE experiment_id = ?", (exp_id,))[0]

    # before/after launch deltas (whole population, event-level daily means)
    launch = db.query_one(
        "SELECT sim_day FROM rollout_history WHERE flag_id = ? AND action = 'launch' "
        "ORDER BY id DESC LIMIT 1", (flag["id"],))
    launch_delta = []
    if launch:
        d = launch["sim_day"]
        for m in db.query("SELECT * FROM metrics ORDER BY id"):
            agg = "AVG(value)" if m["numerator_agg"] in ("sum", "mean") else \
                  "COUNT(*) * 1.0 / COUNT(DISTINCT user_id)"
            row = db.query_one(f"""
                SELECT
                  (SELECT {agg} FROM events WHERE event_name = :ev
                    AND sim_day BETWEEN :d - 7 AND :d - 1) AS before,
                  (SELECT {agg} FROM events WHERE event_name = :ev
                    AND sim_day >= :d) AS after
            """, {"ev": m["numerator_event"], "d": d})
            if row["before"] is not None and row["after"] is not None:
                launch_delta.append({"metric": m, "before": row["before"],
                                     "after": row["after"],
                                     "pct": (row["after"] / row["before"] - 1) * 100
                                            if row["before"] else None})
    return render(request, "release.html", exp=exp, flag=flag, final_look=final_look,
                  final_results=final_results, looks_summary=looks_summary,
                  launch=launch, launch_delta=launch_delta)


@router.get("/calculator", response_class=HTMLResponse)
@router.post("/calculator", response_class=HTMLResponse)
async def calculator(request: Request):
    from ..main import render
    result = None
    form = {"metric_type": "proportion", "baseline": 0.10, "sd": "", "mde": 0.02,
            "alpha": 0.05, "power": 0.8, "planned_looks": 1,
            "spending": "obrien_fleming", "flag_id": ""}
    if request.method == "POST":
        data = await request.form()
        form.update({k: data.get(k, v) for k, v in form.items()})
        try:
            n = ss.n_per_arm(form["metric_type"], float(form["baseline"]),
                             float(form["mde"]), float(form["alpha"]), float(form["power"]),
                             sd=float(form["sd"]) if form["sd"] else None,
                             planned_looks=int(form["planned_looks"]),
                             spending=form["spending"])
            duration = None
            if form["flag_id"]:
                flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (form["flag_id"],)))
                duration = ss.expected_duration_days(2 * n, flag)
            preview = sequential.boundaries(
                [i / int(form["planned_looks"]) for i in range(1, int(form["planned_looks"]) + 1)],
                float(form["alpha"]), form["spending"])
            result = {"n_per_arm": n, "n_total": 2 * n, "duration": duration,
                      "inflation": sequential.inflation_factor(int(form["planned_looks"]),
                                                               form["spending"]),
                      "preview": preview}
        except (ValueError, TypeError) as e:
            result = {"error": str(e)}
    flags = db.query("SELECT * FROM flags ORDER BY id")
    return render(request, "calculator.html", form=form, result=result, flags=flags)
