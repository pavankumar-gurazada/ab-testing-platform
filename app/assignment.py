"""Core assignment logic: deterministic hashing, targeting, rollout gating,
variant splitting, stickiness, and exposure recording.

This module is what a client SDK would embed. It is deliberately written as
pure-ish functions over plain dicts so each step of variant resolution is
individually testable and the debug UI can explain every decision.

Why TWO hashes per flag?
    rollout_bucket decides WHETHER a unit is in the rollout at all;
    variant_bucket decides WHICH variant an enrolled unit gets.
    If one hash did both jobs, ramping rollout_percent from 10% to 20% would
    shuffle some units between variants (their bucket now falls in a different
    slice of a rescaled split). With independent hashes, ramping up only
    ADMITS new units — everyone already enrolled keeps their variant.
"""

import hashlib
import json
from dataclasses import dataclass, field

from . import db

BUCKETS = 10_000  # bucket space granularity: 0.01% resolution


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def bucket(salt: str, purpose: str, unit_id: str) -> int:
    """Deterministic bucket in [0, BUCKETS). Same inputs -> same bucket, on any
    machine, forever. sha256 output is uniform, so buckets are uniform."""
    digest = hashlib.sha256(f"{salt}:{purpose}:{unit_id}".encode()).hexdigest()
    return int(digest[:8], 16) % BUCKETS


# ---------------------------------------------------------------------------
# Targeting
# ---------------------------------------------------------------------------

def _semver_tuple(v: str) -> tuple:
    """'4.1.2' -> (4, 1, 2). Non-numeric parts compare as 0."""
    parts = []
    for p in str(v).split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def evaluate_rules(rules: list[dict], attrs: dict) -> tuple[bool, str]:
    """AND all rules. Returns (eligible, reason). attrs holds the unit's
    attributes: platform, client, app_version, is_qa."""
    for rule in rules:
        actual = attrs.get(rule["attribute"])
        expected = json.loads(rule["value"])
        op = rule["operator"]
        if op == "eq":
            ok = actual == expected
        elif op == "in":
            ok = actual in expected
        elif op == "semver_gte":
            ok = _semver_tuple(actual) >= _semver_tuple(expected)
        elif op == "semver_lt":
            ok = _semver_tuple(actual) < _semver_tuple(expected)
        else:
            raise ValueError(f"unknown operator {op}")
        if not ok:
            return False, f"failed rule: {rule['attribute']} {op} {expected} (was {actual!r})"
    return True, "all rules passed"


# ---------------------------------------------------------------------------
# Variant resolution
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    """The full story of one assignment decision, for the debug UI."""
    variant_key: str
    reason: str                    # 'sticky'|'rolled_back'|'targeting_fail'|'rollout_gate'|'paused'|'new_assignment'|'launched'
    detail: str = ""
    rollout_bucket: int | None = None
    variant_bucket: int | None = None
    enrolled: bool = False         # True only when a NEW assignment row was written


def _control_key(variants: list[dict]) -> str:
    for v in variants:
        if v["is_control"]:
            return v["key"]
    return variants[0]["key"] if variants else "control"


def split_variants(variants: list[dict], variant_bucket: int) -> str:
    """Partition [0, BUCKETS) contiguously by weight; return the variant whose
    slice contains variant_bucket."""
    total = sum(v["weight"] for v in variants)
    edge = 0.0
    for v in variants:
        edge += v["weight"] / total * BUCKETS
        if variant_bucket < edge:
            return v["key"]
    return variants[-1]["key"]  # float-rounding fallback


def resolve(flag: dict, variants: list[dict], rules: list[dict],
            unit_id: str, attrs: dict,
            sticky: dict | None = None) -> Resolution:
    """PURE variant resolution — no database access. Resolution order:

      1. rolled_back  -> control (kill switch overrides everything, even stickiness)
      2. launched     -> the non-control variant is now the default experience
      3. sticky       -> previously assigned units keep their variant
      4. targeting    -> ineligible units get the default (control) experience
      5. rollout gate -> units outside rollout_percent get control
         (paused flags additionally admit no NEW units)
      6. variant split -> hash into weighted slices

    `sticky`, if given, is the unit's existing assignment
    {variant_key, rollout_bucket, variant_bucket}. The caller (get_variant for
    one-off calls, simulator/runner.py for bulk) supplies it; this keeps a
    single code path for both.
    """
    control = _control_key(variants)

    # 1. Kill switch: everyone sees control, including previously-assigned units.
    if flag["state"] == "rolled_back":
        return Resolution(control, "rolled_back", "flag rolled back; serving control to all")

    # 2. Launched: the winning treatment is the new default for everyone.
    if flag["state"] == "launched":
        winner = next((v["key"] for v in variants if not v["is_control"]), control)
        return Resolution(winner, "launched", "flag launched; treatment is the default")

    # 3. Stickiness: first-touch wins.
    if sticky:
        return Resolution(sticky["variant_key"], "sticky",
                          "unit already assigned (first-touch stickiness)",
                          sticky["rollout_bucket"], sticky["variant_bucket"])

    # 4. Targeting. In 'qa' state an implicit is_qa==true rule applies, so a
    #    flag can be exercised by the QA allowlist before any rollout.
    if flag["state"] == "qa":
        rules = rules + [{"attribute": "is_qa", "operator": "eq", "value": "true"}]
    eligible, why = evaluate_rules(rules, attrs)
    if not eligible:
        return Resolution(control, "targeting_fail", why)

    # Draft flags serve control to everyone (nothing is live yet).
    if flag["state"] == "draft":
        return Resolution(control, "targeting_fail", "flag is in draft state")

    rollout_b = bucket(flag["salt"], "rollout", unit_id)
    variant_b = bucket(flag["salt"], "variant", unit_id)

    # 5. Rollout gate. QA state ignores the percentage (QA users always in).
    if flag["state"] != "qa":
        if rollout_b >= flag["rollout_percent"] / 100 * BUCKETS:
            return Resolution(control, "rollout_gate",
                              f"rollout_bucket {rollout_b} >= cutoff "
                              f"{flag['rollout_percent']}% of {BUCKETS}",
                              rollout_b, variant_b)
        if flag["paused"]:
            # Pause stops NEW enrollment only; sticky units returned at step 3.
            return Resolution(control, "paused",
                              "flag paused; not enrolling new units", rollout_b, variant_b)

    # 6. Variant split among enrolled units.
    variant_key = split_variants(variants, variant_b)
    return Resolution(variant_key, "new_assignment",
                      f"enrolled; variant_bucket {variant_b} -> {variant_key}",
                      rollout_b, variant_b, enrolled=True)


def load_flag_config(flag_id: int) -> tuple[list[dict], list[dict]]:
    """Variants and targeting rules for a flag, as plain dicts."""
    variants = [dict(v) for v in db.query(
        "SELECT * FROM variants WHERE flag_id = ? ORDER BY is_control DESC, id", (flag_id,))]
    rules = [dict(r) for r in db.query(
        "SELECT * FROM targeting_rules WHERE flag_id = ?", (flag_id,))]
    return variants, rules


def get_variant(flag: dict, unit_id: str, attrs: dict,
                sim_day: int = 0, sim_time: float = 0.0,
                persist: bool = True) -> Resolution:
    """DB-backed wrapper around resolve(): loads config + sticky assignment,
    persists a new assignment row when one is created.

    persist=False resolves hypothetically (debug UI) without writing rows.
    """
    variants, rules = load_flag_config(flag["id"])
    existing = db.query_one(
        "SELECT variant_key, rollout_bucket, variant_bucket FROM assignments "
        "WHERE flag_id = ? AND unit_id = ?", (flag["id"], unit_id))
    res = resolve(flag, variants, rules, unit_id, attrs,
                  sticky=dict(existing) if existing else None)
    if res.enrolled and persist:
        db.execute(
            "INSERT OR IGNORE INTO assignments "
            "(flag_id, unit_type, unit_id, user_id, variant_key, rollout_bucket, "
            " variant_bucket, sim_day, sim_time) VALUES (?,?,?,?,?,?,?,?,?)",
            (flag["id"], flag["randomization_unit"], unit_id, attrs.get("user_id"),
             res.variant_key, res.rollout_bucket, res.variant_bucket, sim_day, sim_time))
    elif res.enrolled and not persist:
        res.enrolled = False
    return res


def record_exposure(flag: dict, unit_id: str, user_id: str | None,
                    variant_key: str, sim_day: int, sim_time: float) -> None:
    """Log that the unit actually reached the feature surface. Only the first
    exposure per (flag, unit) matters for analysis entry; we keep it cheap by
    only inserting if none exists yet."""
    exists = db.query_one(
        "SELECT 1 FROM exposures WHERE flag_id = ? AND unit_id = ?",
        (flag["id"], unit_id))
    if not exists:
        db.execute(
            "INSERT INTO exposures (flag_id, unit_id, user_id, variant_key, sim_day, sim_time) "
            "VALUES (?,?,?,?,?,?)",
            (flag["id"], unit_id, user_id, variant_key, sim_day, sim_time))
