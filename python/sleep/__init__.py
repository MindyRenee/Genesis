"""Sleep — the mind's side of the sleep–wake cycle.

Mirrors ``src/sleep/`` on the body side, which owns the staging clock
and the RTC wake alarm. The brainstem circuitry decides *when* sleep
happens (the VLPO/orexin flip-flop, the REM oscillator); this package
is what the mind *does* with it — the sleep-state cognition that runs
while the body is down:

    - ``sleep`` — the sleep-wake state machinery driven by the
      brainstem's transitions: what cognition is permitted in each
      stage, and how a wake is reported back through the body.
    - ``dream_synthesis`` — dreaming: recombining stored traces into
      candidate insights and validating them on wake.
    - ``inner_life`` — the spontaneous-thought stream that continues
      unattended while consciousness is offline.
    - ``compression`` — the algorithmic sleep-compaction pass: VQ
      codebook retraining, edge migration into the holographic
      graph, episode summarization, synaptic homeostasis.
"""

from __future__ import annotations

from _views import view_getattr

_EXPORTS: dict[str, str] = {
    # Sleep-state machinery — what the mind may do in each stage
    "SleepMixin": "sleep.sleep",
    "WakeOutcome": "sleep.sleep",
    # Dreaming — recombination and validation of stored traces
    "DreamSynthesis": "sleep.dream_synthesis",
    "DreamProposal": "sleep.dream_synthesis",
    "DreamValidationResult": "sleep.dream_synthesis",
    "DreamInsight": "sleep.dream_synthesis",
    # Inner life — the thought stream that runs while the mind sleeps
    "InnerLife": "sleep.inner_life",
    "nap_capped_stage": "sleep.inner_life",
    # The thought datatype itself (defined with the frontal stream,
    # re-exported here since inner life produces them)
    "SpontaneousThought": "frontal_lobe.thoughts",
    # Sleep compression — the boundedness pass that runs while down
    "SleepCompressor": "sleep.compression",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
