"""Integration tests for the live cognitive-topos adapter."""

from __future__ import annotations

import numpy as np
from self.cognitive_topos_runtime import CognitiveToposRuntime


def test_live_signals_update_all_four_domains_and_engine() -> None:
    runtime = CognitiveToposRuntime(latent_dim=8)
    self_state = {"agency_fiber_states": np.linspace(-1.0, 1.0, 40).tolist()}
    outer_state = {"state": np.linspace(0.0, 1.0, 22).tolist()}

    runtime.observe(
        "outer", kind="perception", content="observed grid",
        outer_snapshot=outer_state, self_snapshot=self_state,
    )
    runtime.observe(
        "inner", kind="reflection", content="compare prior prediction",
        self_snapshot=self_state,
    )
    runtime.observe(
        "self", kind="agency", content="selected action",
        self_snapshot=self_state,
    )
    output = runtime.observe(
        "language", kind="utterance", content="the object moved",
        self_snapshot=self_state,
    )

    assert output["updates"] == 4
    assert output["output"]["last_domain"] == "language"
    assert output["output"]["transported_macro_state"]
    assert len(output["output"]["synchronized_phases"]) == 1
    assert all(len(vector) == 8 for vector in output["states"].values())
    assert np.isfinite(output["output"]["master_free_energy"])


def test_runtime_rejects_unknown_domain_without_advancing() -> None:
    runtime = CognitiveToposRuntime(latent_dim=4)
    try:
        runtime.observe("motor", kind="action", content="move")
    except ValueError as exc:
        assert "unknown cognitive domain" in str(exc)
    else:
        raise AssertionError("unknown domain should be rejected")
    assert runtime.updates == 0



def test_runtime_consumes_real_inner_and_language_latent_snapshots() -> None:
    runtime = CognitiveToposRuntime(latent_dim=8)
    inner_state = np.linspace(-0.8, 0.8, 32)
    language_state = np.linspace(0.9, -0.9, 32)
    output = runtime.observe(
        "inner",
        kind="thought",
        content="reflecting",
        self_snapshot={"agency_fiber_states": np.zeros(40).tolist()},
        inner_snapshot={
            "recursive_meta_states": inner_state.tolist(),
            "affective_fiber_states": np.zeros(32).tolist(),
        },
        language_snapshot={"latent_state": language_state.tolist()},
    )
    # The inner domain consumes the real recursive state, not the text hash.
    expected_inner = np.tanh(np.interp(
        np.linspace(0.0, 1.0, 8),
        np.linspace(0.0, 1.0, inner_state.size),
        inner_state,
    ))
    assert np.allclose(output["states"]["inner"], expected_inner)

    runtime.observe(
        "language",
        kind="utterance",
        content="the parsed language",
        self_snapshot={"agency_fiber_states": np.zeros(40).tolist()},
        language_snapshot={"latent_state": language_state.tolist()},
    )
    expected_language = np.tanh(np.interp(
        np.linspace(0.0, 1.0, 8),
        np.linspace(0.0, 1.0, language_state.size),
        language_state,
    ))
    assert np.allclose(runtime.snapshot()["states"]["language"], expected_language)


def test_runtime_rejects_invalid_latent_dimension() -> None:
    for dimension in (0, -1, True, 2.5):
        try:
            CognitiveToposRuntime(latent_dim=dimension)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid latent dimension accepted: {dimension!r}")
