"""End-to-end: init population -> flag lifecycle -> true effect -> experiment
with sequential looks -> the analysis recovers the configured truth."""

import pytest

from app import db


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "e2e.db")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def _setup_running_experiment(client, effects, planned_looks=3, use_cuped="1"):
    client.post("/simulator/init", data={"population_size": "6000", "seed": "42",
                                         "pre_period_days": "7"}, follow_redirects=False)
    client.post("/flags/1/state", data={"state": "qa"}, follow_redirects=False)
    client.post("/flags/1/state", data={"state": "rollout"}, follow_redirects=False)
    client.post("/flags/1/rollout", data={"percent": "100"}, follow_redirects=False)
    for param, value in effects:
        client.post("/simulator/effects",
                    data={"flag_id": "1", "variant_key": "treatment",
                          "parameter": param, "value": value}, follow_redirects=False)
    client.post("/experiments", data={
        "flag_id": "1", "name": "e2e", "test_type": "difference", "alpha": "0.05",
        "power": "0.8", "planned_looks": str(planned_looks),
        "spending_function": "obrien_fleming", "use_cuped": use_cuped},
        follow_redirects=False)
    client.post("/experiments/1/metrics", data={"metric_id": "2", "role": "success"},
                follow_redirects=False)          # watch_time (continuous)
    client.post("/experiments/1/metrics", data={"metric_id": "5", "role": "guardrail"},
                follow_redirects=False)          # page_latency
    client.post("/experiments/1/design",
                data={"mde": "300", "baseline": "2500", "sd": "3000"},
                follow_redirects=False)
    client.post("/experiments/1/start", follow_redirects=False)


def _looks(client, n_rounds, days_per_round=3):
    out = []
    for _ in range(n_rounds):
        client.post("/simulator/advance", data={"days": str(days_per_round)},
                    follow_redirects=False)
        client.post("/experiments/1/looks", follow_redirects=False)
        out = [dict(r) for r in db.query(
            "SELECT * FROM analysis_looks WHERE experiment_id = 1 ORDER BY look_number")]
        state = db.query_one("SELECT state FROM experiments WHERE id = 1")["state"]
        if state != "running":
            break
    return out, state


def test_large_effect_recovered_and_experiment_concludes(client):
    _setup_running_experiment(client, [("watch_time_multiplier", "1.25"),
                                       ("latency_add_ms", "120")])
    looks, state = _looks(client, 3)
    assert looks, "at least one look ran"
    # alpha ledger is coherent
    assert looks[-1]["alpha_spent_cumulative"] <= 0.05 + 1e-9
    assert all(lk["srm_flag"] == 0 for lk in looks)

    primary = dict(db.query_one(
        "SELECT * FROM look_results WHERE look_id = ? AND role = 'success'",
        (looks[-1]["id"],)))
    assert primary["effect"] > 0, "positive watch-time effect recovered"
    assert primary["cuped_applied"] == 1 and primary["variance_reduction_pct"] > 0

    guardrail = dict(db.query_one(
        "SELECT * FROM look_results WHERE look_id = ? AND role = 'guardrail'",
        (looks[-1]["id"],)))
    assert guardrail["crossed_boundary"] == 1, "latency guardrail flags the +120ms truth"
    assert looks[-1]["decision"] == "guardrail_breach"

    for url in ("/experiments/1/analysis", "/flags/1/monitoring",
                "/experiments/1/release", "/experiments/1"):
        assert client.get(url).status_code == 200


def test_aa_experiment_stays_quiet(client):
    """No true effects: no look may cross the efficacy boundary and SRM must
    stay silent (a single fixed-seed A/A; distributional FPR checks live in
    scripts/calibration.py)."""
    _setup_running_experiment(client, [], planned_looks=3, use_cuped="")
    looks, state = _looks(client, 3)
    success = [dict(r) for lk in looks for r in db.query(
        "SELECT * FROM look_results WHERE look_id = ? AND role = 'success'", (lk["id"],))]
    assert all(not r["crossed_boundary"] for r in success)
    assert all(lk["srm_flag"] == 0 for lk in looks)
    assert state in ("running", "completed", "stopped_futility")


def test_srm_bug_is_detected(client):
    _setup_running_experiment(client, [])
    client.post("/simulator/anomaly", data={"mode": "srm_bug"}, follow_redirects=False)
    looks, _ = _looks(client, 1, days_per_round=4)
    assert looks[-1]["srm_flag"] == 1, "chi-square catches the dropped treatment records"
