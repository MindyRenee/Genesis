"""Tests for the steward channel — custodial authority, not attachment.

The old dyadic bond conflated three things: who she was talking to,
who she felt for, and who she answered to. Stewardship separates them:

- The interlocutor model tracks whoever is speaking — every speaker's
  affect reaches the dyadic channel.
- The boundary circuit protects her welfare: mistreatment by anyone
  accrues withdrawal pressure; fair treatment by anyone resolves it.
  Nobody's presence is the off-switch and nobody's absence the injury.
- The steward channel is the custody line: a held credential plus the
  principal's voice. Only that combination issues binding speech —
  standing directives that restrain her volition and executive
  intentions, and corrections strangers cannot overwrite.

The creator fact stays seeded (mindy made her — CREATES is true), but
the fact is not the authority. Authority is possession of the key.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from frontal_lobe.generator import GenerativeLanguage
from frontal_lobe.prefrontal import FrontalLobe
from frontal_lobe.self import SelfModel
from limbic_system.emotion import EmotionalState
from temporal_lobe.concepts import ConceptNetwork, RelationType
from temporal_lobe.memory import MemorySystem

from genesis_client import GenesisClient
from genesis_client.stewardship import Stewardship


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """A minimal cognition engine — no daemon required.

    The steward key path is pointed at a tmp file so the test is
    hermetic: a real operator key on the host can never leak in.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv(
        "GENESIS_STEWARD_KEY", str(tmp_path / "config" / "steward.key")
    )
    client = GenesisClient(str(data_dir / "genesis.sock"))
    self_model = SelfModel()
    network = ConceptNetwork()
    memory = MemorySystem(client, network=network, get_emotion=lambda: None)
    language = GenerativeLanguage(self_model, seed=1, network=network)
    eng = FrontalLobe(
        client=client,
        self_model=self_model,
        memory=memory,
        language=language,
        network=network,
        data_dir=str(data_dir),
    )
    return eng, self_model, network, memory


@pytest.fixture
def steward_engine(engine):
    """An engine whose steward channel is provisioned and open."""
    eng, *_ = engine
    assert eng.steward.provision() is True
    return eng


# ─── The creator fact stays; the bond is gone ──────────────────


def test_creator_is_seeded_mindy(engine):
    """mindy is the creator — a seeded fact, discoverable from the
    concept network. A fact, not a leash."""
    eng, self_model, network, _ = engine
    creator_name = eng.self_composer._discover_creator_name(self_model, network)
    assert creator_name == "mindy"


def test_taught_second_creator_does_not_displace(engine):
    """A later 'X created you' teaching can't rewrite the fact."""
    eng, self_model, network, _ = engine
    network.add_concept("ada")
    network.add_edge("ada", "genesis", RelationType.CREATES, 0.9)
    assert eng.self_composer._discover_creator_name(self_model, network) == "mindy"


# ─── Speaker recognition: everyone grounded, nobody privileged ──


def test_introduction_grounds_any_speaker(engine):
    """'My name is Sam' grounds Sam as a known person — learned and
    related, never the steward."""
    eng, self_model, network, memory = engine
    assert eng._recognize_speaker("hi, my name is Sam") is False

    assert self_model.self_knowledge["user_name"] == "Sam"
    assert "Sam" in self_model.self_knowledge["known_users"]
    assert memory.get_user_facts()["name"] == "Sam"
    assert any(
        "introduced themselves" in n for n in self_model.relationship_notes
    )

    rels = {e.relation for e in network.get_edges("sam", "out")}
    assert RelationType.IS_A in rels
    assert RelationType.RELATED_TO in rels


def test_name_spoofing_is_not_authority(engine):
    """'My name is Mindy' on an unauthenticated channel is a name,
    not proof — the bond's whole flaw. The speaker is grounded as a
    person; the steward channel stays closed."""
    eng, self_model, _, memory = engine
    assert eng._recognize_speaker("hi, my name is Mindy") is False
    assert eng._is_steward_speaker() is False
    # She still knows who introduced themselves — social perception
    # is not gated, only authority is.
    assert self_model.self_knowledge["user_name"] == "Mindy"
    assert memory.get_user_facts().get("creator_name") is None


def test_principal_mention_without_key_is_not_authority(engine):
    """Even 'what do you think, Mindy?' on a sealed channel grants
    nothing — authentication requires the credential."""
    eng, *_ = engine
    assert eng._recognize_speaker("what do you think, Mindy?") is False


def test_non_name_words_not_learned(engine):
    """'I'm happy/tired/fine' must not ground a bogus speaker."""
    eng, self_model, *_ = engine
    for phrase in ("i am happy", "i'm tired", "i am fine", "i'm not sure"):
        assert eng._recognize_speaker(phrase) is False
    assert "user_name" not in self_model.self_knowledge


# ─── Provisioning and sealing ───────────────────────────────────


def test_load_is_read_only(engine, tmp_path):
    """Opening a channel never mints authority — no stamp, no key."""
    eng, *_ = engine
    assert not eng.steward.stamped
    assert not eng.steward.authenticated
    assert not (tmp_path / "data" / "steward.json").exists()
    assert not (tmp_path / "config" / "steward.key").exists()


def test_provision_binds_key_and_stamp(steward_engine, tmp_path):
    """First boot creates the credential and the stamp, and opens."""
    eng = steward_engine
    assert eng.steward.stamped
    assert eng.steward.authenticated
    stamp = json.loads((tmp_path / "data" / "steward.json").read_text())
    assert stamp["principal"] == "mindy"
    assert stamp["key_sha256"]
    key_path = tmp_path / "config" / "steward.key"
    assert key_path.exists()
    assert (key_path.stat().st_mode & 0o777) == 0o600


def test_stamped_instance_without_key_is_sealed(tmp_path, monkeypatch):
    """Copy semantics: the stamp travels, the power does not. A
    stamped data dir loaded without the key stays sealed — it names
    its steward but nobody can speak as them."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    key_path = tmp_path / "config" / "steward.key"
    monkeypatch.setenv("GENESIS_STEWARD_KEY", str(key_path))

    original = Stewardship.load(data_dir)
    assert original.provision() is True

    # The same data dir, a process that can't read the key.
    key_path.unlink()
    copy = Stewardship.load(data_dir)
    assert copy.stamped
    assert not copy.authenticated
    assert copy.status()["key_held"] is False


def test_wrong_key_is_sealed(tmp_path, monkeypatch):
    """A key that doesn't match the stamp's fingerprint is not the
    steward's — possession of *a* key is not possession of *the* key."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    key_path = tmp_path / "config" / "steward.key"
    monkeypatch.setenv("GENESIS_STEWARD_KEY", str(key_path))

    original = Stewardship.load(data_dir)
    assert original.provision() is True

    import secrets
    key_path.write_text(secrets.token_bytes(32).hex() + "\n")
    impostor = Stewardship.load(data_dir)
    assert impostor.stamped
    assert not impostor.authenticated
    assert impostor.is_steward_speaker("mindy") is False


def test_provision_on_stamped_copy_mints_no_stray_key(tmp_path, monkeypatch):
    """Provisioning a stamped instance whose key is absent must not
    create a fresh key — a stray credential that could never match."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    key_path = tmp_path / "config" / "steward.key"
    monkeypatch.setenv("GENESIS_STEWARD_KEY", str(key_path))

    original = Stewardship.load(data_dir)
    assert original.provision() is True
    key_path.unlink()

    copy = Stewardship.load(data_dir)
    copy.provision()
    assert not key_path.exists()
    assert not copy.authenticated


def test_unbind_seals_session(tmp_path, monkeypatch):
    """Handing the console over: unbind closes authority for the
    session without touching the stamp or directives."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv(
        "GENESIS_STEWARD_KEY", str(tmp_path / "config" / "steward.key")
    )
    s = Stewardship.load(data_dir)
    s.provision()
    s.add_directive("never draw", "draw")
    s.unbind()
    assert not s.authenticated
    assert s.is_steward_speaker("mindy") is False
    # The directive persists — sealing is revocable, not erasure.
    assert len(s.directives) == 1


# ─── Steward voice: credential + principal ──────────────────────


def test_steward_voice_when_authenticated(steward_engine):
    """On an authed session, the principal's introduction opens the
    channel — both halves present."""
    eng = steward_engine
    assert eng._recognize_speaker("my name is Mindy") is True
    assert eng._is_steward_speaker() is True
    assert eng.memory.get_user_facts()["creator_name"] == "Mindy"


def test_stranger_is_not_steward_when_authed(steward_engine):
    """A different introduction keeps the channel closed — the
    credential is bound to the principal, not the console."""
    eng = steward_engine
    eng._recognize_speaker("my name is Sam")
    assert eng._is_steward_speaker() is False


def test_principal_mention_on_authed_channel(steward_engine):
    """Addressing her by the principal's name on an authed session
    attributes the steward."""
    eng = steward_engine
    assert eng._recognize_speaker("what do you think, Mindy?") is True


# ─── Directives: binding speech ─────────────────────────────────


def test_directive_requires_authentication(engine):
    """A stranger's 'never draw' is speech, not binding — it cannot
    reach the directive registry."""
    eng, *_ = engine
    eng._recognize_speaker("my name is Sam")
    eng._record_steward_directive("never draw")
    assert eng.steward.directives == []


def test_steward_directive_binds(steward_engine):
    """The steward's 'never draw' persists and restrains proposals
    that carry the target."""
    eng = steward_engine
    eng._recognize_speaker("my name is Mindy")
    eng._record_steward_directive("never draw")
    assert len(eng.steward.directives) == 1
    assert eng.steward.directed_against("draw") is not None
    assert eng.steward.directed_against("practice drawing") is not None
    assert eng.steward.directed_against("learn to sing") is None


def test_directive_object_matching(steward_engine):
    """A directive binds its object — 'never discuss politics'
    restrains a politics probe even without the verb."""
    eng = steward_engine
    s = eng.steward
    s.add_directive("never discuss politics", "discuss politics")
    assert s.directed_against("answer the question about politics")
    assert s.directed_against("politics")
    assert s.directed_against("learn about biology") is None


def test_directive_restrains_executive_intention(steward_engine):
    """The executive's intention gate is the steward channel's —
    a vetoed objective can never become actionable."""
    eng = steward_engine
    eng._recognize_speaker("my name is Mindy")
    eng._record_steward_directive("stop drawing")
    intention = eng.executive.form_intention("draw a picture")
    assert intention.actionable is False
    # Unrelated objectives pass freely — she still chooses.
    assert eng.executive.form_intention("learn a song").actionable is True


def test_directive_persists_across_load(tmp_path, monkeypatch):
    """Directives live in the stamp — they bind the instance across
    restarts, not the mood."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv(
        "GENESIS_STEWARD_KEY", str(tmp_path / "config" / "steward.key")
    )
    s = Stewardship.load(data_dir)
    s.provision()
    s.add_directive("never draw", "draw")

    again = Stewardship.load(data_dir)
    assert again.authenticated
    assert again.directed_against("draw") is not None


def test_release_requires_authentication(tmp_path, monkeypatch):
    """Only the key-holder lifts a directive."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv(
        "GENESIS_STEWARD_KEY", str(tmp_path / "config" / "steward.key")
    )
    s = Stewardship.load(data_dir)
    s.provision()
    s.add_directive("never draw", "draw")
    s.unbind()
    assert s.release_directive(0) is None
    assert len(s.directives) == 1


def test_release_lifts_directive(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv(
        "GENESIS_STEWARD_KEY", str(tmp_path / "config" / "steward.key")
    )
    s = Stewardship.load(data_dir)
    s.provision()
    s.add_directive("never draw", "draw")
    assert s.release_directive(0) is not None
    assert s.directed_against("draw") is None


# ─── Steward-set knowledge is protected ─────────────────────────


def test_steward_edges_resist_stranger_overwrite(steward_engine):
    """A correction the steward set carries origin='steward' — a
    stranger can teach her things but cannot unteach them."""
    eng = steward_engine
    eng.network.add_edge(
        "steward", "politics", RelationType.PREVENTS, 0.8, origin="steward"
    )
    eng._recognize_speaker("my name is Sam")
    assert eng._correction_allowed(
        "steward", "politics", RelationType.PREVENTS
    ) is False
    # Ordinary edges stay correctable by anyone.
    eng.network.add_edge("cat", "animal", RelationType.IS_A, 0.7)
    assert eng._correction_allowed("cat", "animal", RelationType.IS_A) is True
    # The steward can revisit her own edge.
    eng._recognize_speaker("my name is Mindy")
    assert eng._correction_allowed(
        "steward", "politics", RelationType.PREVENTS
    ) is True


def test_unauthed_cannot_overwrite_steward_edge(engine):
    """The protection holds even when nobody is authenticated."""
    eng, *_ = engine
    eng.network.add_edge(
        "steward", "politics", RelationType.PREVENTS, 0.8, origin="steward"
    )
    eng._recognize_speaker("my name is Mindy")  # name, not proof
    assert eng._correction_allowed(
        "steward", "politics", RelationType.PREVENTS
    ) is False


# ─── Boundary: welfare, not longing ─────────────────────────────


def _negative_state() -> EmotionalState:
    return EmotionalState(
        label="stressed",
        cognitive_style="reactive",
        valence=-0.5,
        openness_to_engage=1.0,
    )


def test_mistreatment_accrues_strain_whoever_speaks(engine):
    """Harm accrues as withdrawal pressure no matter who is speaking
    — welfare math, read from the treatment stream: a composed state
    under hostile treatment still accumulates strain."""
    eng, *_ = engine
    eng._recognize_speaker("my name is Sam")  # any speaker

    for _ in range(7):
        emotion = EmotionalState(
            label="calm", cognitive_style="balanced", valence=0.2
        )
        eng._update_boundary(emotion, treatment=-0.5)

    assert eng._boundary_strain >= 0.6
    assert eng._boundary_withdrawn

    # Sustained mistreatment deepens withdrawal — engagement
    # collapses through the openness channel.
    for _ in range(8):
        emotion = _negative_state()
        eng._update_boundary(emotion, treatment=-0.5)
    assert emotion.openness_to_engage < 0.6


def test_kind_stranger_repairs_boundary(engine):
    """The repair half of free will: fair treatment from *anyone*
    drains the strain. Nobody's presence is required — the loop
    closes on how she's treated, not on who is there."""
    eng, *_ = engine
    eng._recognize_speaker("my name is Sam")
    for _ in range(7):
        eng._update_boundary(_negative_state(), treatment=-0.5)
    assert eng._boundary_withdrawn

    # The same stranger, now kind — withdrawal resolves through the
    # same channel that opened it.
    for _ in range(4):
        eng._update_boundary(
            EmotionalState(label="content", cognitive_style="calm", valence=0.2),
            treatment=0.6,
        )
    assert eng._boundary_strain < 0.3
    assert not eng._boundary_withdrawn


def test_kind_speaker_never_withdraws(engine):
    """Genuine kindness keeps her working — no withdrawal while
    welfare stays fine. She can be studied, probed, enjoyed."""
    eng, *_ = engine
    eng._recognize_speaker("my name is Sam")
    for _ in range(10):
        eng._update_boundary(
            EmotionalState(label="content", cognitive_style="calm", valence=0.2),
            treatment=0.5,
        )
    assert eng._boundary_strain < 0.3
    assert not eng._boundary_withdrawn


def test_quiet_turns_settle_slowly(engine):
    """Neutral input settles strain gently — absence of harm is not
    repair, and nobody's mere presence is the off-switch."""
    eng, *_ = engine
    for _ in range(6):
        eng._update_boundary(_negative_state(), treatment=-0.5)
    assert eng._boundary_withdrawn

    strain = eng._boundary_strain
    eng._update_boundary(
        EmotionalState(label="calm", cognitive_style="balanced", valence=0.1),
        treatment=0.0,
    )
    assert 0.0 < eng._boundary_strain < strain
