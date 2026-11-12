"""One cognitive turn, handed between regions.

A turn used to be a method chain inside FrontalLobe: one class
perceived, felt, remembered, deliberated, and spoke. The regions
already own those functions. This module is the handoff — each stage
names the region that performs it and calls that region's method on
the engine. Adding a stage means adding a region call here, not
growing another method on the engine.

It lives at the repository root beside ``mind.py`` and
``commitment.py`` — the level of the whole organism, not inside
any region and not inside the composition entry (the regions can't
import ``mind.py`` without a cycle): the organism is the mind, and
one pass of input through its regions is orchestration of the
whole, not a part of it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Turn:
    """A single pass of input through the regions that process it."""

    def __init__(self, engine: Any, user_input: str) -> None:
        self.engine = engine
        self.user_input = user_input

    def run(self) -> tuple[str, Any]:
        import time as _time

        engine = self.engine
        timing, t0 = engine._think_setup()
        raw_input = self.user_input

        # Temporal: resolve reference, then comprehend.
        resolved = engine._resolve_anaphora(raw_input)
        comprehension, learning_events, prediction, recent_turns = (
            engine._think_comprehend_and_predict(resolved)
        )
        timing["comprehend"] = _time.perf_counter() - t0

        # Occipital/parietal: perceive. Perception is run on the raw
        # words — anaphora resolution rewrites "you" and would mark
        # every second-person question as being about Genesis.
        t1 = _time.perf_counter()
        perception, prediction_error = engine._think_perceive_and_error(raw_input, prediction)
        timing["perceive"] = _time.perf_counter() - t1
        logger.debug(
            "perceive done in %.3fs, topics=%s",
            timing["perceive"],
            perception.topics[:5],
        )

        engine._think_learn_emotion_label(raw_input)

        # Frontal: route before the full pipeline. An early route is
        # a complete answer and ends the turn here.
        t1 = _time.perf_counter()
        early = engine._think_early_routes(resolved, perception, raw_input)
        timing["early_routes"] = _time.perf_counter() - t1
        if early is not None:
            timing["total"] = _time.perf_counter() - t0
            logger.debug("think() timing (early route): %s", timing)
            return early

        engine._think_early_broadcast(perception)
        engine._think_cognitive_surprise()

        (
            emotion,
            _brain_waves,
            _user_model,
            _damasio_feeling,
            _memory_mode,
            _memory_context,
            _reasoning_results,
            goal,
            thought,
            _ddm_result,
            _workspace_item_summaries,
            response,
            _repairs,
            answer_assessment,
            _semantic_facts,
            state,
        ) = engine._think_run_pipeline(
            resolved,
            perception,
            prediction_error,
            recent_turns,
            learning_events,
            comprehension,
            timing,
            t0,
        )
        timing["total"] = _time.perf_counter() - t0
        logger.debug("think() timing: %s", timing)
        return engine._think_post_response(
            state,
            response,
            goal,
            perception,
            emotion,
            thought,
            prediction,
            prediction_error,
            insights=None,
            answer_assessment=answer_assessment,
            learning_events=learning_events,
            comprehension_result=comprehension,
        )
