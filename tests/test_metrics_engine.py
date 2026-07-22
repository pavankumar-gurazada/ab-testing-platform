"""Metrics engine edge cases on a small hand-built fixture DB:
attribution windows, missing-value policies, winsorization, ratio columns,
exposure- vs assignment-triggered populations."""

import pytest

from app import db, metrics_engine


@pytest.fixture()
def fixture_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "engine.db")
    db.init_db()
    app_id = db.execute("INSERT INTO applications (name) VALUES ('t')")
    flag_id = db.execute(
        "INSERT INTO flags (application_id, key, name, salt, state, rollout_percent, "
        " exposure_trigger) VALUES (?,?,?,?,?,?,?)",
        (app_id, "f", "F", "s", "experiment", 100, "exposure"))
    for key, ctrl in (("control", 1), ("treatment", 0)):
        db.execute("INSERT INTO variants (flag_id, key, name, is_control, weight) "
                   "VALUES (?,?,?,?,1)", (flag_id, key, key, ctrl))

    # three users; u1/u2 exposed on day 0, u3 on day 5
    for uid, variant, day in (("u1", "control", 0), ("u2", "treatment", 0),
                              ("u3", "treatment", 5)):
        db.execute("INSERT INTO users (user_id, device_id, platform, client, app_version, "
                   " activity, engagement, skill, pre_lessons_completed) "
                   "VALUES (?,?,?,?,?,0.5,0.5,70,2)", (uid, "d" + uid, "web", "browser", "4.0.0"))
        db.execute("INSERT INTO exposures (flag_id, unit_id, user_id, variant_key, sim_day, "
                   " sim_time) VALUES (?,?,?,?,?,?)", (flag_id, uid, uid, variant, day, float(day)))

    # events: u1 completes on days 1 and 9 (values 100, 900); u2 never completes;
    # u3 completes on day 6 (value 50) and has an extreme outlier on day 7 (10000)
    rows = [("u1", "lesson_complete", 100.0, 1), ("u1", "lesson_complete", 900.0, 9),
            ("u3", "lesson_complete", 50.0, 6), ("u3", "lesson_complete", 10000.0, 7),
            ("u1", "enrollment", None, 1), ("u3", "enrollment", None, 6)]
    for uid, ev, val, day in rows:
        db.execute("INSERT INTO events (user_id, session_id, event_name, value, sim_day, "
                   " sim_time) VALUES (?,?,?,?,?,?)", (uid, uid + ":s", ev, val, day, day + 0.5))
    return flag_id


def _metric(**over):
    base = dict(id=1, type="continuous", numerator_event="lesson_complete",
                numerator_agg="sum", denominator_event=None, denominator_agg=None,
                attribution_window_days=None, missing_value_policy="zero",
                winsorize_pct=None, direction="increase")
    base.update(over)
    return base


def _flag(flag_id):
    return dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag_id,)))


def test_zero_fill_includes_non_converters(fixture_db):
    frame = metrics_engine.compute_metric(_metric(), _flag(fixture_db), 0, 10)
    assert len(frame) == 3
    assert frame.set_index("unit_id").loc["u2", "y"] == 0.0


def test_exclude_drops_non_converters(fixture_db):
    frame = metrics_engine.compute_metric(_metric(missing_value_policy="exclude"),
                                          _flag(fixture_db), 0, 10)
    assert set(frame["unit_id"]) == {"u1", "u3"}


def test_attribution_window_caps_events(fixture_db):
    """u1 exposed day 0: a 5-day window keeps the day-1 event (100), drops day 9."""
    frame = metrics_engine.compute_metric(_metric(attribution_window_days=5),
                                          _flag(fixture_db), 0, 10)
    assert frame.set_index("unit_id").loc["u1", "y"] == 100.0
    # u3 exposed day 5: day-6 and day-7 events both inside its window
    assert frame.set_index("unit_id").loc["u3", "y"] == 10050.0


def test_as_of_day_truncates(fixture_db):
    """Analyzing as of day 4: u1 has only the day-1 event; u3 not yet exposed."""
    frame = metrics_engine.compute_metric(_metric(), _flag(fixture_db), 0, 4)
    assert set(frame["unit_id"]) == {"u1", "u2"}
    assert frame.set_index("unit_id").loc["u1", "y"] == 100.0


def test_winsorize_caps_outlier(fixture_db):
    frame = metrics_engine.compute_metric(_metric(winsorize_pct=0.5),
                                          _flag(fixture_db), 0, 10)
    assert frame["y"].max() < 10050.0
    assert frame.attrs["n_capped"] >= 1


def test_proportion_semantics(fixture_db):
    frame = metrics_engine.compute_metric(
        _metric(type="proportion", numerator_agg="any"), _flag(fixture_db), 0, 10)
    got = frame.set_index("unit_id")["y"].to_dict()
    assert got == {"u1": 1.0, "u2": 0.0, "u3": 1.0}


def test_ratio_columns(fixture_db):
    frame = metrics_engine.compute_metric(
        _metric(type="ratio", numerator_agg="count",
                denominator_event="enrollment", denominator_agg="count"),
        _flag(fixture_db), 0, 10)
    row = frame.set_index("unit_id")
    assert row.loc["u1", "num"] == 2 and row.loc["u1", "den"] == 1
    assert row.loc["u2", "num"] == 0 and row.loc["u2", "den"] == 0


def test_cuped_covariate_attached(fixture_db):
    frame = metrics_engine.compute_metric(_metric(numerator_agg="count"),
                                          _flag(fixture_db), 0, 10)
    assert (frame["x_pre"] == 2).all()      # pre_lessons_completed = 2 for all users


def test_assignment_trigger_uses_assignments_table(fixture_db):
    """With exposure_trigger='assignment' and an empty assignments table the
    population is empty — proving the trigger switches the entry table."""
    flag = _flag(fixture_db)
    flag["exposure_trigger"] = "assignment"
    frame = metrics_engine.compute_metric(_metric(), flag, 0, 10)
    assert frame.empty
