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
