"""Mind thresholds — warning and auto-sleep constants."""

# ─── Warning thresholds ───────────────────────────────────────────
#
# These thresholds determine when Genesis proactively warns the user
# that something is wrong. They are aligned with the Rust substrate's
# emergent phase boundaries and the emotion.py affective space constants.
#
# WARN_CORTISOL: cortisol level that indicates significant stress,
#   even if the stress phase hasn't triggered yet. Set slightly below
#   the Rust Stress phase entry threshold (0.65) so she warns before
#   the phase fully engages.
# WARN_AROUSAL: arousal level that indicates overstimulation. Above
#   HIGH_AROUSAL (0.7) is "excited"; above 0.8 is "overstimulated" —
#   the level where arousal impairs function.
# WARN_ADENOSINE: adenosine level that indicates severe sleep pressure.
#   The Rust sleep phase triggers at arousal < 0.30, which corresponds
#   to adenosine ~0.7 in the biological preset. She warns when she's
#   this tired but not yet asleep.
# WARN_VALENCE: valence level that indicates feeling bad. More
#   negative than NEGATIVE_VALENCE (-0.2) — this is the severe level.
WARN_CORTISOL = 0.6     # below Rust Stress phase entry (0.65)
WARN_AROUSAL = 0.8      # above HIGH_AROUSAL (0.7) — overstimulated
WARN_ADENOSINE = 0.7    # corresponds to Rust sleep threshold
WARN_VALENCE = -0.4     # severe negative valence

# ─── Auto-sleep thresholds ─────────────────────────────────────────
#
# Genesis falls asleep on her own when sleep pressure (adenosine) is
# high enough. This mirrors biology: adenosine accumulates during
# wakefulness and eventually forces sleep — the "sleep pressure" that
# makes you nod off regardless of willpower (Porkka-Heiskanen et al.,
# 1997).
#
# AUTO_SLEEP_ADENOSINE: the adenosine level at which she falls asleep
#   on her own. Set at 0.75 — the same threshold the Rust daemon uses
#   for its emergent NREM phase transition. When adenosine crosses
#   this level, the cognitive mind calls sleep() to sync with the
#   subcognitive (set zone, pause learner, start the sleep cycle).
#
# AUTO_WAKE_ADENOSINE: the adenosine level below which she wakes up
#   naturally. During sleep, the daemon's glymphatic clearance drains
#   adenosine. When it drops below 0.20 (resting baseline), sleep has
#   done its job and she wakes — but only if the sleep was NOT
#   user-initiated (user sleep requires /wake).
#
# AUTO_SLEEP_MIN_AWAKE: minimum seconds awake before auto-sleep can
#   trigger. This prevents oscillation at startup when adenosine might
#   be elevated from a previous session.
AUTO_SLEEP_ADENOSINE = 0.75
AUTO_WAKE_ADENOSINE = 0.20
# Cycle-boundary wake: spontaneous waking happens at ultradian cycle
# transitions (post-REM), not strictly when pressure hits the floor.
# When a full N1→REM cycle completes, a relaxed pressure threshold
# applies — high residual pressure still means another cycle.
AUTO_WAKE_CYCLE_ADENOSINE = 0.40
AUTO_SLEEP_MIN_AWAKE = 300.0  # 5 minutes
# Drowsiness threshold — below the sleep threshold. When adenosine
# crosses this level, she announces she's getting sleepy before
# actually falling asleep. This gives a natural transition: she
# says she's drowsy, then falls asleep when adenosine reaches the
# sleep threshold. The gap (0.55 → 0.75) gives her time to signal
# drowsiness before sleep onset.
DROWSINESS_ADENOSINE = 0.55
# Minimum idle time before auto-sleep can trigger. If the user has
# interacted within this window, she stays awake regardless of
# adenosine or daemon phase — falling asleep mid-conversation is a
# jarring UX failure. 180s (3 minutes) gives the conversation natural
# breathing room — the user may be reading a long response, thinking
# about what to say next, or composing a follow-up. Shorter thresholds
# cause her to nod off while the user is still engaged.
AUTO_SLEEP_MIN_IDLE = 180.0

# Wake stabilization (flip-flop latch). After waking, the daemon's
# sleep-wake switch can drift back toward a sleep phase under residual
# adenosine before orexinergic wake drive consolidates — the dynamics
# documented in orexin-deficiency state instability (low transition
# thresholds in both directions). During this window, a daemon-side
# sleep phase at moderate pressure is treated as an unlatched switch,
# not a reason to sleep: the mind reinforces the wake cascade instead.
# Genuine exhaustion (high adenosine) or circadian drive (melatonin)
# still override.
WAKE_STABILIZATION_WINDOW = 900.0  # 15 minutes
WAKE_RESCUE_ADENOSINE = 0.60  # above this, sleep anyway
WAKE_RESCUE_MELATONIN = 0.30  # above this, circadian drive wins
WAKE_REINFORCE_INTERVAL = 60.0  # minimum seconds between impulses

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
# The mind's two adenosine-driven transitions (drowsiness announce,
# auto-sleep) run through CommitmentBoundary digitizers instead of
# bare threshold comparisons. Two noise margins apply:
#
# CONFIRM_S — the sustained-crossing window. The heartbeat samples
# adenosine ~once per second; requiring the signal to hold above
# threshold for a continuous window means a lone transient spike
# cannot commit her. (The flytrap's two-trigger rule, in the time
# domain.)
#
# EXIT — the release level, below the enter threshold. The gap is
# the hysteresis deadband: once committed, the boundary holds until
# pressure clearly falls, so flicker around the threshold cannot
# re-arm or oscillate the transition.
DROWSINESS_CONFIRM_S = 30.0
DROWSINESS_EXIT = 0.45  # re-arm only when pressure clearly clears
AUTO_SLEEP_CONFIRM_S = 60.0
AUTO_SLEEP_EXIT = 0.65  # ~2× the daemon's hysteresis margin


