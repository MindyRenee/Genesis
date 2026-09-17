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


