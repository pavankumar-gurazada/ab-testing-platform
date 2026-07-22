"""Seed the demo application, example flags and the metric catalog.

Idempotent: runs on every startup, inserts only if the DB is empty.
The demo domain is an ed-tech learning platform ("LearnHub"): learners
enroll -> watch lessons -> complete courses -> take quizzes.
"""

import secrets

from . import db


def seed_if_empty() -> None:
    if db.query_one("SELECT 1 FROM applications LIMIT 1"):
        return

    app_id = db.execute(
        "INSERT INTO applications (name, description) VALUES (?, ?)",
        ("LearnHub", "Demo ed-tech learning platform: learners enroll, watch lessons, "
                     "complete courses and take quizzes."))

    # --- Example flag: a redesigned lesson player -------------------------
    flag_id = db.execute(
        "INSERT INTO flags (application_id, key, name, description, salt, "
        " randomization_unit, exposure_trigger, state, rollout_percent) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (app_id, "new-lesson-player", "New Lesson Player",
         "Redesigned video player with inline quizzes and auto-resume.",
         secrets.token_hex(8), "user", "exposure", "draft", 0))
    db.execute("INSERT INTO variants (flag_id, key, name, is_control, weight) VALUES (?,?,?,?,?)",
               (flag_id, "control", "Current player", 1, 1))
    db.execute("INSERT INTO variants (flag_id, key, name, is_control, weight) VALUES (?,?,?,?,?)",
               (flag_id, "treatment", "New player", 0, 1))

    # --- Metric catalog ----------------------------------------------------
    metric_rows = [
        # key, name, type, num_event, num_agg, den_event, den_agg, window, missing, winsor, direction, description
        ("lesson_completion_rate", "Lesson completion rate", "proportion",
         "lesson_complete", "any", None, None, 14, "zero", None, "increase",
         "Share of learners completing at least one lesson within 14 days of exposure."),
        ("watch_time", "Watch time per learner (s)", "continuous",
         "lesson_complete", "sum", None, None, 14, "zero", 0.99, "increase",
         "Total seconds of lesson watch time per learner; winsorized at p99."),
        ("avg_quiz_score", "Average quiz score", "continuous",
         "quiz_submit", "mean", None, None, 14, "exclude", None, "increase",
         "Mean quiz score among learners who submitted a quiz (missing = excluded)."),
        ("completions_per_enrollment", "Completions per enrollment", "ratio",
         "lesson_complete", "count", "enrollment", "count", 14, "zero", None, "increase",
         "Lesson completions divided by enrollments (delta-method analysis)."),
        ("page_latency", "Page load latency (ms)", "continuous",
         "page_view", "mean", None, None, 14, "exclude", 0.99, "decrease",
         "Mean page-view latency per learner. Guardrail: lower is better."),
        ("course_completion_rate", "Course completion rate", "proportion",
         "course_complete", "any", None, None, None, "zero", None, "increase",
         "Share of learners finishing a full course (no attribution window cap)."),
    ]
    for row in metric_rows:
        db.execute(
            "INSERT INTO metrics (application_id, key, name, type, numerator_event, "
            " numerator_agg, denominator_event, denominator_agg, attribution_window_days, "
            " missing_value_policy, winsorize_pct, direction, description) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (app_id, *row))
