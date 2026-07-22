"""Synthetic population generation.

Every user gets observable attributes (platform, client, app_version, is_qa)
used by targeting, and LATENT TRAITS (activity, engagement, skill) that drive
the behavior model. The traits are stored in the users table on purpose:
ground truth should always be inspectable in a learning tool.

Heterogeneous traits are also what makes CUPED work here: the same trait
drives a user's pre-period and experiment-period behavior, so pre-period
metrics correlate with experiment metrics (rho ~ 0.5-0.7).
"""

import numpy as np

from .. import db

PLATFORMS = ["web", "ios", "android"]
PLATFORM_P = [0.45, 0.30, 0.25]
APP_VERSIONS = ["3.9.1", "4.0.2", "4.1.0"]
APP_VERSION_P = [0.2, 0.5, 0.3]
N_QA_USERS = 25


def generate_population(n: int, seed: int) -> int:
    """Create n users; returns the number created."""
    rng = np.random.default_rng(seed)

    platform = rng.choice(PLATFORMS, size=n, p=PLATFORM_P)
    app_version = rng.choice(APP_VERSIONS, size=n, p=APP_VERSION_P)
    activity = rng.beta(2, 5, size=n)            # mean ~0.29 visits/day
    engagement = rng.beta(4, 3, size=n)          # mean ~0.57 completion propensity
    skill = np.clip(rng.normal(70, 12, size=n), 0, 100)

    rows = []
    for i in range(n):
        uid = f"u{i:05d}"
        rows.append((
            uid, f"d{i:05d}", platform[i],
            "browser" if platform[i] == "web" else "mobile_app",
            app_version[i],
            1 if i < N_QA_USERS else 0,           # first 25 users are the QA allowlist
            float(activity[i]), float(engagement[i]), float(skill[i]),
        ))

    with db.get_conn() as conn:
        conn.executemany(
            "INSERT INTO users (user_id, device_id, platform, client, app_version, "
            " is_qa, activity, engagement, skill) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return n
