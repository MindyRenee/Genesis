"""Mind thresholds — warning and sleep-signal constants."""

# ─── Warning thresholds ───────────────────────────────────────────
#
# These thresholds determine when Genesis proactively warns the user
# that something is wrong. They are aligned with the Rust substrate's
# emergent phase boundaries and the emotion.py affective space constants.
#
# WARN_CORTISOL: cortisol level that indicates significant stress,
#   even if the stress phase hasn't triggered yet. Set slightly below
#   the Rust Stress phase entry gate (STRESS_CORTISOL_GATE = 0.06, set
#   against the substrate's measured envelope — resting ~0.02, driven
#   peak ~0.084) so it warns before the phase fully engages. The
#   previous value (0.6) was calibrated against a retired 0.65 gate
#   the substrate can never reach — cortisol saturates near 0.09 —
#   so the warning could never fire.
# WARN_AROUSAL: arousal level that indicates overstimulation. Above
#   HIGH_AROUSAL (0.7) is "excited"; above 0.8 is "overstimulated" —
#   the level where arousal impairs function.
# WARN_ADENOSINE: adenosine level that indicates severe sleep pressure.
#   The daemon's sleep-wake network owns sleep onset (LC < 4 Hz), but
#   adenosine ~0.7 is deep into the saturating Process-S accumulation
#   range (upper asymptote 0.95) — this warns when it is this tired
#   but not yet asleep.
# WARN_VALENCE: valence level that indicates feeling bad. More
#   negative than NEGATIVE_VALENCE (-0.2) — this is the severe level.
WARN_CORTISOL = 0.05  # below Rust Stress phase gate (0.06)
# AUTO_WAKE_CORTISOL: tonic cortisol at/above which the stress system
#   is still carrying a load and a requested wake should wait. Aligned
#   with the Rust stress-phase gate (STRESS_CORTISOL_GATE = 0.06): the
#   circadian basal drive is sized to peak just under it, so a healthy
#   morning surge must not read as "unrecovered" — while genuine
#   stress elevation (the same signal the body itself calls Stress)
#   still does. The previous value (0.05) sat inside the basal
#   envelope: under the zero-floor chemistry it could never fire, and
#   under the live HPA axis it would veto a normal morning.
AUTO_WAKE_CORTISOL = 0.06  # Rust STRESS_CORTISOL_GATE
WARN_AROUSAL = 0.8  # above HIGH_AROUSAL (0.7) — overstimulated
WARN_ADENOSINE = 0.7  # corresponds to Rust sleep threshold
WARN_VALENCE = -0.4  # severe negative valence

# Re-announcement holdoff for spoken warnings. warn() speaks only when
# the set of active conditions changes — but a condition hovering at
# its threshold toggles that set every heartbeat, so without a holdoff
# the same warning is re-spoken every time the signal re-crosses. A
# condition key announced within this window stays "announced": a
# warning is spoken only when it carries at least one condition not
# spoken recently. 300 s is long enough that a marginal signal cannot
# chirp, short enough that a genuinely recurring problem re-surfaces.
WARN_REPEAT_SECONDS = 300.0

# ─── Sleep-wake signals ────────────────────────────────────────────
#
# There are no autonomic sleep/wake thresholds: transitions into and
# out of sleep are hers — the self-sleep urge self-invoking /sleep,
# a user command, or the body's own sleep-wake network crossing under
# genuine pressure (the desync checks then join the other layer).
# What remains here are the signaling constants the waking-side
# maintenance reads.
#
# AUTO_WAKE_ADENOSINE: the adenosine level below which sleep has done
#   its job — used by wake_readiness() as the "sleep pressure drained"
#   signal when deciding whether a requested wake arrives at a good
#   moment. It is advisory, not a trigger.
AUTO_WAKE_ADENOSINE = 0.20
# Minimum seconds awake before the waking-side sleep signals (the
# drowsiness announce and the wake-stabilization renewals) can fire.
# Prevents oscillation at startup when adenosine might be elevated
# from a previous session.
AUTO_SLEEP_MIN_AWAKE = 300.0  # 5 minutes
# Drowsiness threshold — when adenosine crosses this level, she can
# say she's getting sleepy (composed from her own understanding, not
# a canned string).
DROWSINESS_ADENOSINE = 0.55
# Minimum idle time before the waking-side sleep signals engage. If
# the user has interacted within this window the signals hold off —
# announcing sleepiness or renewing a wake drive mid-conversation is
# noise. 180s (3 minutes) gives the conversation natural breathing
# room — the user may be reading a long response, thinking about what
# to say next, or composing a follow-up.
AUTO_SLEEP_MIN_IDLE = 180.0

# Wake stabilization (flip-flop latch). After waking, the daemon's
# sleep-wake switch can drift back toward a sleep phase under residual
# adenosine before orexinergic wake drive consolidates — the dynamics
# documented in orexin-deficiency state instability (low transition
# thresholds in both directions). During this window, a sleep urge or
# a daemon-side sleep phase at moderate pressure is treated as an
# unlatched switch, not a reason to sleep.
# Genuine exhaustion (high adenosine) or circadian drive (melatonin)
# still override.
WAKE_STABILIZATION_WINDOW = 900.0  # 15 minutes
WAKE_RESCUE_ADENOSINE = 0.60  # above this, sleep anyway
WAKE_RESCUE_MELATONIN = 0.30  # above this, circadian drive wins

# How long after a wake/sleep cascade an un-crossed daemon phase is
# still the expected transitional state rather than a desync. The
# impulses decay on their own half-lives, but the LC population
# crosses its own threshold on a ~23-minute time constant — from deep
# NREM at night a genuine transition can take over half an hour. The
# window bounds how long the mind waits on a chemical attempt before
# treating an un-crossed phase as truth: flagging earlier would report
# a healthy in-progress transition as a fault.
WAKE_DESYNC_SETTLE_SECONDS = 2700.0

# When no cascade attempt is outstanding, a short post-wake grace still
# suppresses transient asleep-phase reports (stage flutter near the
# threshold) rather than treating them as faults.
WAKE_DESYNC_GRACE_SECONDS = 90.0

# Voluntary sleep-urge floor. The self_sleep urge integrates
# adenosine as a drive signal, but deciding to sleep should require
# pressure near the drowsy boundary (DROWSINESS_ADENOSINE = 0.55,
# daemon Drowsy enter ≈ 0.50 blended). Below this floor, residual
# post-wake pressure produces grogginess that fades — not a renewed
# decision to sleep. Without the floor, moderate residual pressure
# (~0.28) re-crosses the urge threshold in under a minute of
# undampened integration, recreating the narcolepsy-like brief-wake
# phenotype the stabilization window exists to prevent.
SLEEP_URGE_ADENOSINE_FLOOR = 0.45

# ─── Commitment-boundary parameters ──────────────────────────────
#
# The mind's adenosine-driven drowsiness announcement runs through a
# CommitmentBoundary digitizer instead of a bare threshold comparison.
# Two noise margins apply:
#
# CONFIRM_S — the sustained-crossing window. The heartbeat samples
# adenosine ~once per second; requiring the signal to hold above
# threshold for a continuous window means a lone transient spike
# cannot commit it. (The flytrap's two-trigger rule, in the time
# domain.)
#
# EXIT — the release level, below the enter threshold. The gap is
# the hysteresis deadband: once committed, the boundary holds until
# pressure clearly falls, so flicker around the threshold cannot
# re-arm or oscillate the signal.
DROWSINESS_CONFIRM_S = 30.0
DROWSINESS_EXIT = 0.45  # re-arm only when pressure clearly clears

# ─── Task-readiness threshold ────────────────────────────────────
#
# READY_ALLOSTASIS_MAX: the generative self-model's accumulated
# regulatory burden (inference_signals.allostasis_load, 0–1) above
# which demanding focused work is refused. Allostatic load is the
# cost of sustained adaptation — high load means the system needs
# recovery, not another task. The other refusal criteria reuse the
# WARN_* values above rather than defining a second health model.
READY_ALLOSTASIS_MAX = 0.8
