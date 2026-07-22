"""Assignment logic tests: hash determinism/uniformity, targeting operators,
rollout gating, stickiness, pause/rollback semantics, variant splits."""

import pytest
from scipy import stats as sps

from app import assignment, db


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    """Point the app at a throwaway SQLite file."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    yield


@pytest.fixture()
def flag(fresh_db):
    app_id = db.execute("INSERT INTO applications (name) VALUES ('test-app')")
    flag_id = db.execute(
        "INSERT INTO flags (application_id, key, name, salt, state, rollout_percent) "
        "VALUES (?,?,?,?,?,?)", (app_id, "test-flag", "Test", "fixedsalt", "rollout", 100))
    db.execute("INSERT INTO variants (flag_id, key, name, is_control, weight) "
               "VALUES (?,?,?,?,?)", (flag_id, "control", "Control", 1, 1))
    db.execute("INSERT INTO variants (flag_id, key, name, is_control, weight) "
               "VALUES (?,?,?,?,?)", (flag_id, "treatment", "Treatment", 0, 1))
    return dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag_id,)))


ATTRS = {"user_id": "u1", "platform": "web", "client": "browser",
         "app_version": "4.0.0", "is_qa": False}


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def test_bucket_deterministic():
    assert assignment.bucket("s", "variant", "u42") == assignment.bucket("s", "variant", "u42")


def test_bucket_purposes_independent():
    """Rollout and variant buckets must be uncorrelated for the same unit."""
    same = sum(assignment.bucket("s", "rollout", f"u{i}") == assignment.bucket("s", "variant", f"u{i}")
               for i in range(2000))
    assert same < 10  # collisions only by chance (~2000/10000)


def test_bucket_uniformity():
    """Chi-square on 20 equal-width bins over 10k units."""
    buckets = [assignment.bucket("salt", "variant", f"user{i}") for i in range(10_000)]
    counts = [0] * 20
    for b in buckets:
        counts[b // 500] += 1
    _, p = sps.chisquare(counts)
    assert p > 0.01


# ---------------------------------------------------------------------------
# Targeting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op,value,attr_value,expected", [
    ("eq", '"ios"', "ios", True),
    ("eq", '"ios"', "web", False),
    ("in", '["ios","android"]', "android", True),
    ("in", '["ios","android"]', "web", False),
    ("semver_gte", '"4.1.0"', "4.1.0", True),
    ("semver_gte", '"4.1.0"', "4.10.2", True),   # numeric, not lexicographic
    ("semver_gte", '"4.1.0"', "4.0.9", False),
    ("semver_lt", '"5.0.0"', "4.9.9", True),
])
def test_evaluate_rules_operators(op, value, attr_value, expected):
    rules = [{"attribute": "platform" if "semver" not in op else "app_version",
              "operator": op, "value": value}]
    attr_key = "app_version" if "semver" in op else "platform"
    ok, _ = assignment.evaluate_rules(rules, {attr_key: attr_value})
    assert ok is expected


def test_rules_are_anded():
    rules = [{"attribute": "platform", "operator": "eq", "value": '"ios"'},
             {"attribute": "is_qa", "operator": "eq", "value": "true"}]
    ok, _ = assignment.evaluate_rules(rules, {"platform": "ios", "is_qa": True})
    assert ok
    ok, why = assignment.evaluate_rules(rules, {"platform": "ios", "is_qa": False})
    assert not ok and "is_qa" in why


# ---------------------------------------------------------------------------
# Resolution semantics
# ---------------------------------------------------------------------------

def test_split_roughly_even(flag):
    n_treat = sum(
        assignment.get_variant(flag, f"u{i}", ATTRS, persist=False).variant_key == "treatment"
        for i in range(4000))
    assert 1800 < n_treat < 2200  # 50/50 ± ~5 sigma


def test_rollout_gate_admits_expected_fraction(flag):
    db.execute("UPDATE flags SET rollout_percent = 20 WHERE id = ?", (flag["id"],))
    flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag["id"],)))
    enrolled = sum(
        assignment.get_variant(flag, f"u{i}", ATTRS, persist=False).reason == "new_assignment"
        for i in range(5000))
    assert 850 < enrolled < 1150  # 20% of 5000 = 1000


def test_ramp_up_does_not_reshuffle(flag):
    """The core two-hash property: units enrolled at 10% keep their variant at 50%."""
    db.execute("UPDATE flags SET rollout_percent = 10 WHERE id = ?", (flag["id"],))
    f10 = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag["id"],)))
    at_10 = {f"u{i}": assignment.get_variant(f10, f"u{i}", ATTRS, persist=False)
             for i in range(3000)}
    enrolled_at_10 = {u: r.variant_key for u, r in at_10.items() if r.reason == "new_assignment"}

    db.execute("UPDATE flags SET rollout_percent = 50 WHERE id = ?", (flag["id"],))
    f50 = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag["id"],)))
    for u, variant_at_10 in enrolled_at_10.items():
        assert assignment.get_variant(f50, u, ATTRS, persist=False).variant_key == variant_at_10


def test_stickiness(flag):
    first = assignment.get_variant(flag, "sticky-unit", ATTRS)
    assert first.reason == "new_assignment"
    # Even if the split flips to 100% control, the unit keeps its variant.
    db.execute("UPDATE variants SET weight = 0 WHERE flag_id = ? AND is_control = 0", (flag["id"],))
    again = assignment.get_variant(flag, "sticky-unit", ATTRS)
    assert again.reason == "sticky"
    assert again.variant_key == first.variant_key


def test_pause_blocks_new_but_keeps_existing(flag):
    before = assignment.get_variant(flag, "early-bird", ATTRS)
    db.execute("UPDATE flags SET paused = 1 WHERE id = ?", (flag["id"],))
    paused_flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag["id"],)))
    assert assignment.get_variant(paused_flag, "early-bird", ATTRS).variant_key == before.variant_key
    fresh = assignment.get_variant(paused_flag, "latecomer", ATTRS)
    assert fresh.reason == "paused" and fresh.variant_key == "control"


def test_rollback_overrides_stickiness(flag):
    assigned = assignment.get_variant(flag, "victim", ATTRS)
    db.execute("UPDATE flags SET state = 'rolled_back' WHERE id = ?", (flag["id"],))
    rb_flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag["id"],)))
    res = assignment.get_variant(rb_flag, "victim", ATTRS)
    assert res.variant_key == "control" and res.reason == "rolled_back"
    assert assigned.variant_key in ("control", "treatment")  # sanity


def test_qa_state_gates_on_is_qa(flag):
    db.execute("UPDATE flags SET state = 'qa' WHERE id = ?", (flag["id"],))
    qa_flag = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag["id"],)))
    civilian = assignment.get_variant(qa_flag, "u-civ", {**ATTRS, "is_qa": False}, persist=False)
    assert civilian.reason == "targeting_fail"
    tester = assignment.get_variant(qa_flag, "u-qa", {**ATTRS, "is_qa": True}, persist=False)
    assert tester.reason == "new_assignment"


def test_launched_serves_treatment_to_all(flag):
    db.execute("UPDATE flags SET state = 'launched' WHERE id = ?", (flag["id"],))
    launched = dict(db.query_one("SELECT * FROM flags WHERE id = ?", (flag["id"],)))
    for u in ("a", "b", "c"):
        assert assignment.get_variant(launched, u, ATTRS).variant_key == "treatment"


def test_weighted_split(flag):
    db.execute("UPDATE variants SET weight = 9 WHERE flag_id = ? AND is_control = 1", (flag["id"],))
    n_treat = sum(
        assignment.get_variant(flag, f"w{i}", ATTRS, persist=False).variant_key == "treatment"
        for i in range(5000))
    assert 400 < n_treat < 600  # 10% of 5000 = 500
