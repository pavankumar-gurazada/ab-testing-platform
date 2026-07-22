-- ============================================================================
-- A/B Testing Platform — SQLite schema
--
-- Design notes:
--  * This is a learning prototype: every internal (hash buckets, alpha spent,
--    CUPED theta, true simulated effects) is STORED so it can be rendered.
--  * Times come in two flavors:
--      - created_at: real wall-clock time, informational only
--      - sim_day / sim_time: the simulated clock. sim_day is an integer day
--        counter (day 0 = start of the simulation, pre-period days are
--        negative). sim_time = sim_day + fraction-of-day, a REAL, so events
--        within a day are ordered.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Config entities
-- ---------------------------------------------------------------------------

-- An application is the top-level container (e.g. "LearnHub"). Flags and
-- metrics belong to an application; targeting can filter by it.
CREATE TABLE IF NOT EXISTS applications (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- The feature flag is the base unit of the whole platform. Its lifecycle:
--   draft -> qa -> rollout -> experiment -> launched
-- with rolled_back reachable from qa/rollout/experiment (kill switch).
-- An experiment (see experiments table) is an analysis layer attached to a
-- flag while the flag is in 'experiment' state.
CREATE TABLE IF NOT EXISTS flags (
    id                 INTEGER PRIMARY KEY,
    application_id     INTEGER NOT NULL REFERENCES applications(id),
    key                TEXT NOT NULL UNIQUE,          -- e.g. 'new-lesson-player'
    name               TEXT NOT NULL,
    description        TEXT NOT NULL DEFAULT '',
    salt               TEXT NOT NULL,                 -- random hex; seeds the assignment hashes
    randomization_unit TEXT NOT NULL DEFAULT 'user'
                       CHECK (randomization_unit IN ('user','device','session')),
    exposure_trigger   TEXT NOT NULL DEFAULT 'exposure'
                       CHECK (exposure_trigger IN ('assignment','exposure')),
    state              TEXT NOT NULL DEFAULT 'draft'
                       CHECK (state IN ('draft','qa','rollout','experiment','launched','rolled_back')),
    rollout_percent    REAL NOT NULL DEFAULT 0,       -- 0..100, gate on top of targeting
    paused             INTEGER NOT NULL DEFAULT 0,    -- 1 = no NEW units enrolled; existing keep variant
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Variants of a flag. Exactly one is_control per flag (enforced in app code).
-- weight is a relative split among ENROLLED units (e.g. 50/50).
CREATE TABLE IF NOT EXISTS variants (
    id         INTEGER PRIMARY KEY,
    flag_id    INTEGER NOT NULL REFERENCES flags(id),
    key        TEXT NOT NULL,                          -- 'control', 'treatment'
    name       TEXT NOT NULL,
    is_control INTEGER NOT NULL DEFAULT 0,
    weight     REAL NOT NULL DEFAULT 1,
    UNIQUE (flag_id, key)
);

-- Targeting rules restrict WHO is eligible for a flag. All rules on a flag
-- are AND-ed together (deliberate simplification; real platforms have rule
-- sets with OR groups). 'value' is JSON: a scalar for eq/semver ops, a list
-- for 'in'.
CREATE TABLE IF NOT EXISTS targeting_rules (
    id        INTEGER PRIMARY KEY,
    flag_id   INTEGER NOT NULL REFERENCES flags(id),
    attribute TEXT NOT NULL CHECK (attribute IN ('platform','client','app_version','is_qa')),
    operator  TEXT NOT NULL CHECK (operator IN ('eq','in','semver_gte','semver_lt')),
    value     TEXT NOT NULL                            -- JSON-encoded
);

-- Audit trail of every rollout action; drives the state timeline on the flag
-- page and stage annotations on the monitoring page.
CREATE TABLE IF NOT EXISTS rollout_history (
    id         INTEGER PRIMARY KEY,
    flag_id    INTEGER NOT NULL REFERENCES flags(id),
    action     TEXT NOT NULL,                          -- 'set_state','set_percent','pause','resume','rollback','launch'
    percent    REAL,
    sim_day    INTEGER NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Metric definitions over the raw event stream.
--   proportion: did the unit do numerator_event at least once? (0/1)
--   continuous: aggregate of numerator_event's value (or count) per unit
--   ratio:      numerator agg / denominator agg, analyzed with the delta method
CREATE TABLE IF NOT EXISTS metrics (
    id                       INTEGER PRIMARY KEY,
    application_id           INTEGER NOT NULL REFERENCES applications(id),
    key                      TEXT NOT NULL UNIQUE,
    name                     TEXT NOT NULL,
    type                     TEXT NOT NULL CHECK (type IN ('proportion','continuous','ratio')),
    numerator_event          TEXT NOT NULL,
    numerator_agg            TEXT NOT NULL DEFAULT 'count'
                             CHECK (numerator_agg IN ('any','count','sum','mean')),
    denominator_event        TEXT,                     -- ratio metrics only
    denominator_agg          TEXT CHECK (denominator_agg IN ('count','sum')),
    attribution_window_days  INTEGER,                  -- NULL = from exposure until analysis time
    missing_value_policy     TEXT NOT NULL DEFAULT 'zero'
                             CHECK (missing_value_policy IN ('zero','exclude')),
    winsorize_pct            REAL,                     -- e.g. 0.99 caps at 99th percentile; NULL = off
    direction                TEXT NOT NULL DEFAULT 'increase'
                             CHECK (direction IN ('increase','decrease')),
    description              TEXT NOT NULL DEFAULT ''
);

-- An experiment is the ANALYSIS configuration attached to a flag: hypothesis,
-- statistical design, and its lifecycle. The flag handles delivery; the
-- experiment handles inference.
CREATE TABLE IF NOT EXISTS experiments (
    id                     INTEGER PRIMARY KEY,
    flag_id                INTEGER NOT NULL REFERENCES flags(id),
    name                   TEXT NOT NULL,
    hypothesis             TEXT NOT NULL DEFAULT '',
    test_type              TEXT NOT NULL DEFAULT 'difference'
                           CHECK (test_type IN ('difference','noninferiority')),
    noninferiority_margin  REAL,                       -- absolute margin, same units as the metric
    alpha                  REAL NOT NULL DEFAULT 0.05,
    power                  REAL NOT NULL DEFAULT 0.8,
    mde                    REAL,                       -- minimum detectable effect (absolute)
    planned_looks          INTEGER NOT NULL DEFAULT 1, -- 1 = fixed-horizon
    spending_function      TEXT NOT NULL DEFAULT 'obrien_fleming'
                           CHECK (spending_function IN ('obrien_fleming','pocock')),
    use_cuped              INTEGER NOT NULL DEFAULT 0,
    target_n_per_arm       INTEGER,                    -- filled by the design step
    expected_duration_days REAL,                       -- filled by the design step
    state                  TEXT NOT NULL DEFAULT 'draft'
                           CHECK (state IN ('draft','running','stopped_efficacy','stopped_futility',
                                            'completed','aborted')),
    start_sim_day          INTEGER,
    end_sim_day            INTEGER,
    created_at             TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Metrics attached to an experiment with a role:
--   success:    drives the stop/ship decision (sequential boundaries apply)
--   guardrail:  must not regress; one-sided check against metric.direction
--   supporting: informational only
CREATE TABLE IF NOT EXISTS experiment_metrics (
    id            INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    metric_id     INTEGER NOT NULL REFERENCES metrics(id),
    role          TEXT NOT NULL CHECK (role IN ('success','guardrail','supporting')),
    UNIQUE (experiment_id, metric_id)
);

-- ---------------------------------------------------------------------------
-- Population & events (filled by the simulator)
-- ---------------------------------------------------------------------------

-- Simulated users. Latent traits (activity/engagement/skill) are the TRUE
-- data-generating parameters — stored openly so ground truth is inspectable.
-- pre_* columns are per-user pre-experiment aggregates used by CUPED.
CREATE TABLE IF NOT EXISTS users (
    id                    INTEGER PRIMARY KEY,
    user_id               TEXT NOT NULL UNIQUE,        -- 'u00042'
    device_id             TEXT NOT NULL,               -- 'd00042' (one device per user, simplification)
    platform              TEXT NOT NULL CHECK (platform IN ('web','ios','android')),
    client                TEXT NOT NULL CHECK (client IN ('browser','mobile_app')),
    app_version           TEXT NOT NULL,
    is_qa                 INTEGER NOT NULL DEFAULT 0,
    activity              REAL NOT NULL,               -- P(visits on a given day)
    engagement            REAL NOT NULL,               -- P(completes a started lesson)
    skill                 REAL NOT NULL,               -- mean quiz score
    pre_lessons_completed REAL,                        -- filled after pre-period
    pre_watch_time        REAL,
    pre_quiz_score        REAL
);

-- Assignment log. First-touch sticky: the first resolved variant for a
-- (flag, unit) pair is looked up before hashing on subsequent calls.
-- Both hash buckets are stored so the debug UI can show WHY a unit landed
-- where it did.
CREATE TABLE IF NOT EXISTS assignments (
    id             INTEGER PRIMARY KEY,
    flag_id        INTEGER NOT NULL REFERENCES flags(id),
    unit_type      TEXT NOT NULL,                      -- 'user'|'device'|'session'
    unit_id        TEXT NOT NULL,
    user_id        TEXT,                               -- owning user (for joining to metrics)
    variant_key    TEXT NOT NULL,
    rollout_bucket INTEGER NOT NULL,
    variant_bucket INTEGER NOT NULL,
    sim_day        INTEGER NOT NULL,
    sim_time       REAL NOT NULL,
    UNIQUE (flag_id, unit_id)
);

-- Exposure log: the unit actually reached the feature surface (saw the
-- change), as opposed to merely being assigned. Analysis can use either as
-- the population entry point, per flags.exposure_trigger.
CREATE TABLE IF NOT EXISTS exposures (
    id          INTEGER PRIMARY KEY,
    flag_id     INTEGER NOT NULL REFERENCES flags(id),
    unit_id     TEXT NOT NULL,
    user_id     TEXT,
    variant_key TEXT NOT NULL,
    sim_day     INTEGER NOT NULL,
    sim_time    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exposures_flag_unit ON exposures (flag_id, unit_id);

-- The raw event stream. One narrow table; 'value' is event-specific:
--   lesson_complete -> watch time (seconds)
--   quiz_submit     -> score (0-100)
--   page_view       -> latency (ms)
--   others          -> NULL
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_name TEXT NOT NULL,
    value      REAL,
    sim_day    INTEGER NOT NULL,
    sim_time   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_user_time ON events (user_id, sim_time);
CREATE INDEX IF NOT EXISTS idx_events_name_time ON events (event_name, sim_time);

-- ---------------------------------------------------------------------------
-- Analysis results & simulation state
-- ---------------------------------------------------------------------------

-- One row per interim look at an experiment. This is where sequential testing
-- becomes visible: information fraction, alpha spent, and the boundary that
-- the observed z was compared against.
CREATE TABLE IF NOT EXISTS analysis_looks (
    id                        INTEGER PRIMARY KEY,
    experiment_id             INTEGER NOT NULL REFERENCES experiments(id),
    look_number               INTEGER NOT NULL,
    sim_day                   INTEGER NOT NULL,
    information_fraction      REAL NOT NULL,           -- n_observed / n_target
    alpha_spent_cumulative    REAL NOT NULL,
    alpha_spent_this_look     REAL NOT NULL,
    efficacy_z_boundary       REAL NOT NULL,
    futility_conditional_power REAL,                   -- NULL on final look
    srm_chi2                  REAL,
    srm_p                     REAL,
    srm_flag                  INTEGER NOT NULL DEFAULT 0,
    decision                  TEXT NOT NULL
                              CHECK (decision IN ('continue','stop_efficacy','stop_futility',
                                                  'guardrail_breach','complete')),
    created_at                TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (experiment_id, look_number)
);

-- Per-metric results within a look.
CREATE TABLE IF NOT EXISTS look_results (
    id                     INTEGER PRIMARY KEY,
    look_id                INTEGER NOT NULL REFERENCES analysis_looks(id),
    metric_id              INTEGER NOT NULL REFERENCES metrics(id),
    role                   TEXT NOT NULL,
    control_n              INTEGER NOT NULL,
    treatment_n            INTEGER NOT NULL,
    control_mean           REAL NOT NULL,
    treatment_mean         REAL NOT NULL,
    effect                 REAL NOT NULL,              -- treatment - control (absolute)
    se                     REAL NOT NULL,
    z_stat                 REAL NOT NULL,
    p_value                REAL NOT NULL,
    ci_low                 REAL NOT NULL,
    ci_high                REAL NOT NULL,
    cuped_applied          INTEGER NOT NULL DEFAULT 0,
    cuped_theta            REAL,
    variance_reduction_pct REAL,
    crossed_boundary       INTEGER NOT NULL DEFAULT 0
);

-- Single-row simulation state (id = 1 always).
-- current_sim_day is the NEXT day to simulate; pre-period runs on negative days.
CREATE TABLE IF NOT EXISTS sim_state (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    current_sim_day INTEGER NOT NULL DEFAULT 0,
    rng_seed        INTEGER NOT NULL DEFAULT 42,
    anomaly_mode    TEXT NOT NULL DEFAULT 'none'
                    CHECK (anomaly_mode IN ('none','srm_bug','guardrail_degrade')),
    pre_period_days INTEGER NOT NULL DEFAULT 0,
    population_size INTEGER NOT NULL DEFAULT 0
);

-- The configured GROUND TRUTH effects the simulator applies to treated users.
-- Analysis pages display these beside the estimates so "did the stats recover
-- the truth?" is always answerable.
CREATE TABLE IF NOT EXISTS sim_effects (
    id          INTEGER PRIMARY KEY,
    flag_id     INTEGER NOT NULL REFERENCES flags(id),
    variant_key TEXT NOT NULL,
    parameter   TEXT NOT NULL
                CHECK (parameter IN ('lesson_complete_uplift_pp','watch_time_multiplier',
                                     'quiz_score_shift','latency_add_ms','activity_multiplier')),
    value       REAL NOT NULL,
    UNIQUE (flag_id, variant_key, parameter)
);
