"""The TRUE data-generating process: one simulated user-day.

This file is the ground truth that the statistics are later asked to recover.
A user's behavior is driven by their latent traits, perturbed by the effect
parameters of whatever treatment they are exposed to:

    activity_multiplier        x P(visit today)           (applied by runner)
    lesson_complete_uplift_pp  + P(complete | started)    (percentage points)
    watch_time_multiplier      x watch seconds per completed lesson
    quiz_score_shift           + mean quiz score
    latency_add_ms             + page-view latency

Funnel per session: page_view -> enroll (once ever) -> 1-3 lesson_start,
each completing w.p. engagement -> quiz after completion -> course_complete
after 8 cumulative lesson completions.
"""

import numpy as np

# Baseline page latency (ms, lognormal median) by platform.
BASE_LATENCY = {"web": 800.0, "ios": 600.0, "android": 1000.0}
COURSE_LENGTH = 8          # lesson completions needed to finish a course
QUIZ_PROB = 0.7            # P(quiz after a completed lesson)
SESSIONS_MAX = 2


def simulate_user_day(user: dict, day: int, effects: dict, rng: np.random.Generator,
                      state: dict) -> list[tuple]:
    """Generate one active day of events for a user.

    `effects` holds the treatment parameters in force for this user today
    (empty dict = control / not exposed).
    `state` is the user's cumulative funnel state {enrolled: bool,
    completions: int}; mutated in place so course_complete fires exactly once
    per COURSE_LENGTH completions across days.

    Returns rows shaped for the events table:
    (user_id, session_id, event_name, value, sim_day, sim_time).
    """
    events = []
    uid = user["user_id"]
    n_sessions = int(rng.integers(1, SESSIONS_MAX + 1))

    p_complete = min(1.0, max(0.0, user["engagement"]
                              + effects.get("lesson_complete_uplift_pp", 0.0)))
    watch_mult = effects.get("watch_time_multiplier", 1.0)
    score_shift = effects.get("quiz_score_shift", 0.0)
    latency_add = effects.get("latency_add_ms", 0.0)

    for s in range(n_sessions):
        sid = f"{uid}:s{day}:{s}"
        # spread sessions across the day: sim_time = day + fraction
        t = day + float(rng.uniform(0, 1))

        latency = rng.lognormal(np.log(BASE_LATENCY[user["platform"]]), 0.35) + latency_add
        events.append((uid, sid, "page_view", float(latency), day, t))

        if not state["enrolled"]:
            # Enroll in a (new) course; recurs after each course completion.
            events.append((uid, sid, "enrollment", None, day, t))
            state["enrolled"] = True

        for _ in range(int(rng.integers(1, 4))):          # 1-3 lessons started
            events.append((uid, sid, "lesson_start", None, day, t))
            if rng.random() < p_complete:
                watch = rng.lognormal(np.log(300 + 600 * user["engagement"]), 0.4) * watch_mult
                events.append((uid, sid, "lesson_complete", float(watch), day, t))
                state["completions"] += 1
                if state["completions"] % COURSE_LENGTH == 0:
                    events.append((uid, sid, "course_complete", None, day, t))
                    state["enrolled"] = False   # next session enrolls in a new course
                if rng.random() < QUIZ_PROB:
                    score = float(np.clip(rng.normal(user["skill"] + score_shift, 8), 0, 100))
                    events.append((uid, sid, "quiz_submit", score, day, t))
    return events
