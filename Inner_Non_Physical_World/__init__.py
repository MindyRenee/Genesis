"""Inner_Non_Physical_World — the mind's non-physical inner world.

The three worlds partition the organism by what each thing *is*:

- ``Self_World/`` — the physical self: the brain regions and
  neurochemistry (the substrate), the Rust body, the self model,
  and the runtime that holds it all together.
- ``Inner_Non_Physical_World/`` — this world: the non-physical
  phenomena the organism undergoes — the field, the brain waves,
  dreams, and thoughts — plus the mental instruments (``tools/``)
  it thinks with.
- ``Outer_Physical_World/`` — the physical world outside: the
  channel, the profile, the board.

The boundary is a vesica, not a wall: a phenomenon can be non-physical
in kind and still be driven by physical machinery — ``brain_waves``
is read off the thalamic relay's neurochemistry, ``sleep``'s dreams
are gated by the hypothalamic flip-flop, ``thoughts`` draw on the
concept network the temporal lobe stores. The worlds share one
namespace (every world root is on ``PYTHONPATH``), so overlap is the
default: a module lives in the world it belongs to and is importable
from everywhere it is needed.

# What lives here

    brain_waves.py       — the thalamocortical rhythm layer: a coupled
                           amplitude-phase oscillator over the canonical
                           bands, read off the neurochemical state.
    field_probe.py       — the daemon's substrate probe mapped onto the
                           wave oscillator's stimulus pathway — real
                           machine physics entering the model.
    field_audio.py       — the self-emission acoustic channel: the
                           field's signature escaping the chassis,
                           band-decomposed into the same stimulus.
    thoughts.py          — the data structures of spontaneous thought:
                           ThoughtChainType and SpontaneousThought, the
                           unit of inner experience.
    thought_composer.py  — where novel thoughts are generated: bridges
                           the concept network and the language engine.
    sleep/               — the sleep cycle and the dream machinery:
                           inner life, dream synthesis, compression.
    tools/               — the mental instruments: code learning,
                           exploration, the test framework, HID, web.

Entry into the whole organism is ``mind.Mind`` in ``Self_World`` —
this world is what the mind *experiences*, not the organ that runs.
"""
