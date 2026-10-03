"""Tests for GET_SUBSYSTEM_TELEMETRY parsing and the activity tiers.

Two gaps motivated this file. The subsystem/module wire format had no
Python-side coverage at all — only ``tests/ipc.rs`` exercised it, from
the Rust side, which is how a real divergence between the two parsers
went unnoticed (see the truncation tests below). And the activity tiers
the format feeds had no coverage for the seed projection the language
engine reads.

Layout under test (little-endian), from ``serialize_subsystem_telemetry``
in ``src/daemon/ipc.rs``:

    [u8 count] then per subsystem (21 bytes):
      [u8 subsystem][u32 pid][f32 cpu][f32 io]
      [f32 cache_miss_rate][f32 branch_miss_rate]
    [u8 module_count] then per module (6 bytes):
      [u8 module_id][u8 status][f32 cpu_share]
"""

import struct
from dataclasses import FrozenInstanceError

import pytest

from genesis_client.protocol import (
    MODULE_DREAMING,
    MODULE_EMOTION,
    MODULE_LANGUAGE,
    MODULE_MEMORY,
    MODULE_METACOGNITION,
    MODULE_REASONING,
    PHASE_NAMES,
    SENSOR_BATTERY,
    SENSOR_FAN,
    SENSOR_FREQUENCY,
    SENSOR_MAX_FREQUENCY,
    SENSOR_MEMORY,
    SENSOR_PSI,
    SENSOR_TEMPERATURE,
    SUBSYSTEM_COGNITIVE,
    SUBSYSTEM_DAEMON,
    SUBSYSTEM_RETINA,
    ZONE_NAMES,
)
from genesis_client.types import (
    MODULE_REC_V2,
    MODULE_REC_V3,
    BodyState,
    ModuleTelemetry,
    ProcessHealth,
    SensorPresence,
    SubsystemReport,
    SubsystemTelemetry,
    ZoneTransitions,
)
from genesis_cognitive.self.model import SelfModel

TRACKED_STATES = 8  # must match zones::TRACKED_STATES
SUBSYSTEM_REC = 21
MODULE_REC = MODULE_REC_V2


def _subsystem_rec(
    tag: int, pid: int, cpu: float, io: float,
    cache: float, branch: float,
) -> bytes:
    return (
        bytes([tag])
        + struct.pack("<I", pid)
        + struct.pack("<ffff", cpu, io, cache, branch)
    )


def _module_rec(module_id: int, status: int, cpu_share: float) -> bytes:
    """The original 6-byte v2 record: no task or error tail."""
    return bytes([module_id, status]) + struct.pack("<f", cpu_share)


def _module_rec_v3(
    module_id: int,
    status: int,
    cpu_share: float,
    task_id: int = 0,
    error_code: int = 0,
) -> bytes:
    """The 18-byte v3 record, with the optional tail."""
    return (
        bytes([module_id, status])
        + struct.pack("<f", cpu_share)
        + struct.pack("<Q", task_id)
        + struct.pack("<I", error_code)
    )


# ─── subsystem section ────────────────────────────────────────


def test_unpack_three_subsystems() -> None:
    """A full subsystem section round-trips field-for-field."""
    data = bytes([3])
    data += _subsystem_rec(0, 100, 0.05, 0.02, 0.01, 0.005)
    data += _subsystem_rec(1, 200, 0.40, 0.10, 0.08, 0.03)
    data += _subsystem_rec(2, 300, 0.15, 0.30, 0.00, 0.00)
    data += bytes([0])  # no module section

    report = SubsystemReport.unpack(data)

    assert len(report.subsystems) == 3
    daemon, cognitive, retina = report.subsystems
    assert daemon.subsystem_name == "daemon"
    assert daemon.pid == 100
    assert daemon.cpu == pytest.approx(0.05)
    assert cognitive.subsystem_name == "cognitive"
    assert cognitive.pid == 200
    assert cognitive.cache_miss_rate == pytest.approx(0.08)
    assert cognitive.branch_miss_rate == pytest.approx(0.03)
    assert retina.subsystem_name == "retina"
    assert retina.io == pytest.approx(0.30)
    assert report.modules == ()


def test_unpack_empty_report() -> None:
    """The empty report serializes as [0][0] and unpacks to nothing."""
    report = SubsystemReport.unpack(bytes([0, 0]))
    assert report.subsystems == ()
    assert report.modules == ()


def test_unknown_subsystem_tag_is_named_not_dropped() -> None:
    """An unrecognized tag is labelled, never silently discarded."""
    data = bytes([1]) + _subsystem_rec(9, 42, 0.1, 0.1, 0.0, 0.0) + bytes([0])
    report = SubsystemReport.unpack(data)
    assert report.subsystems[0].subsystem_name == "unknown:9"


def test_truncated_subsystem_section_raises() -> None:
    """A short subsystem section is an error, not a silent zero-fill."""
    data = bytes([2]) + _subsystem_rec(0, 1, 0.1, 0.1, 0.1, 0.1)
    with pytest.raises(ValueError):
        SubsystemReport.unpack(data)


# ─── module section ───────────────────────────────────────────


def test_unpack_module_section() -> None:
    """The module section carries the self-reported brain parts."""
    data = bytes([0])
    data += bytes([2])
    data += _module_rec(MODULE_LANGUAGE, 2, 0.42)
    data += _module_rec(MODULE_DREAMING, 2, 0.0)

    report = SubsystemReport.unpack(data)

    assert report.subsystems == ()
    assert len(report.modules) == 2
    lang = report.modules[0]
    assert lang.module_id == MODULE_LANGUAGE
    assert lang.module_name == "language"
    assert lang.status == 2
    assert lang.cpu_share == pytest.approx(0.42)
    # A module at zero load must survive the round trip as zero, not
    # be dropped or confused with "absent".
    assert report.modules[1].cpu_share == pytest.approx(0.0)
    assert report.modules[1].module_name == "dreaming"


def test_unpack_both_sections() -> None:
    """Both sections coexist in one response."""
    data = bytes([1]) + _subsystem_rec(1, 200, 0.4, 0.1, 0.08, 0.03)
    data += bytes([1]) + _module_rec(MODULE_REASONING, 2, 1.0)

    report = SubsystemReport.unpack(data)

    assert len(report.subsystems) == 1
    assert len(report.modules) == 1
    assert report.modules[0].module_name == "reasoning"


def test_unpack_missing_module_section_is_empty() -> None:
    """A daemon predating the module section parses with no modules."""
    data = bytes([1]) + _subsystem_rec(1, 200, 0.4, 0.1, 0.08, 0.03)
    report = SubsystemReport.unpack(data)
    assert report.subsystems
    assert report.modules == ()


def test_unknown_module_id_is_named_not_dropped() -> None:
    """An unrecognized module id is labelled, never silently discarded."""
    data = bytes([0]) + bytes([1]) + _module_rec(200, 2, 0.5)
    report = SubsystemReport.unpack(data)
    assert report.modules[0].module_name == "unknown:200"


def test_truncated_module_section_yields_no_modules() -> None:
    """A module section too short to hold its records is treated as empty.

    Matching the Rust parser's tolerance is deliberate. The heartbeat's
    sensor block catches ValueError around the *whole* sensor cycle, so
    raising here would discard the hardware reading it had already
    taken — over the malformed module tail, which is the least
    informative part of the response. Losing the module section instead
    leaves the activity tiers stale, which is visible, and keeps the
    body state, which is not.
    """
    data = bytes([0]) + bytes([5])
    report = SubsystemReport.unpack(data)
    assert report.modules == ()


def test_record_sizes_match_the_documented_layout() -> None:
    """The 21-byte and 6-byte record sizes are what the docs claim."""
    assert len(_subsystem_rec(0, 1, 0.0, 0.0, 0.0, 0.0)) == SUBSYSTEM_REC
    assert len(_module_rec(0, 0, 0.0)) == MODULE_REC
    assert len(_module_rec_v3(0, 0, 0.0)) == MODULE_REC_V3


# ─── activity tiers → seeds ───────────────────────────────────


class _FakeTelemetry:
    """Minimal stand-in for a wire subsystem record."""

    def __init__(self, name: str, cpu: float) -> None:
        self.subsystem_name = name
        self.cpu = cpu


class _FakeModuleTelemetry:
    """Minimal stand-in for a wire module record."""

    def __init__(self, name: str, cpu_share: float) -> None:
        self.module_name = name
        self.cpu_share = cpu_share


def _report(subsystems, modules):
    class R:
        pass

    r = R()
    r.subsystems = subsystems
    r.modules = modules
    return r


def test_update_brain_activity_records_both_tiers() -> None:
    """The report populates the hardware and self-measured tiers."""
    model = SelfModel()
    model.update_brain_activity(_report(
        [_FakeTelemetry("cognitive", 0.4), _FakeTelemetry("daemon", 0.05)],
        [_FakeModuleTelemetry("reasoning", 0.6),
         _FakeModuleTelemetry("language", 0.2)],
    ))

    assert model.body_model.subsystem_activity == {"cognitive": 0.4, "daemon": 0.05}
    assert model.body_model.module_activity == {"reasoning": 0.6, "language": 0.2}


def test_update_brain_activity_ignores_none() -> None:
    """A missing report leaves both tiers untouched."""
    model = SelfModel()
    model.body_model.module_activity = {"reasoning": 0.5}
    model.update_brain_activity(None)
    assert model.body_model.module_activity == {"reasoning": 0.5}


def test_activity_seeds_orders_by_measured_share() -> None:
    """The busiest region leads, so the composer tries it first."""
    model = SelfModel()
    model.body_model.module_activity = {
        "reasoning": 0.1, "language": 0.5, "memory": 0.3,
    }
    assert model.activity_seeds() == ["language", "memory", "reasoning"]


def test_activity_seeds_excludes_idle_regions() -> None:
    """A region at zero load is not offered.

    Seeding it would bias the composer toward whatever sorted first
    rather than toward what is actually firing.
    """
    model = SelfModel()
    model.body_model.module_activity = {"reasoning": 0.7, "language": 0.0}
    assert model.activity_seeds() == ["reasoning"]


def test_activity_seeds_empty_when_nothing_is_firing() -> None:
    """With no activity there is nothing to seed from."""
    model = SelfModel()
    assert model.activity_seeds() == []


def test_activity_seeds_maps_subsystems_to_their_regions() -> None:
    """Process names map onto the concept names for the regions they host."""
    model = SelfModel()
    model.body_model.subsystem_activity = {"daemon": 0.4}
    assert model.activity_seeds() == ["subcognitive"]


def test_activity_seeds_passes_unmapped_names_through() -> None:
    """An unrecognized process name is still usable as a seed."""
    model = SelfModel()
    model.body_model.subsystem_activity = {"newprocess": 0.3}
    assert model.activity_seeds() == ["newprocess"]


def test_activity_seeds_deduplicates_across_tiers() -> None:
    """A name reaching both tiers appears once."""
    model = SelfModel()
    model.body_model.module_activity = {"cognition": 0.2}
    model.body_model.subsystem_activity = {"cognitive": 0.3}
    seeds = model.activity_seeds()
    assert seeds.count("cognition") == 1


def test_activity_seeds_are_concept_names_not_prose() -> None:
    """Seeds are bare region names — it composes the words itself.

    Guards the project rule that Genesis never recites pre-written
    responses: the projection from telemetry to language must yield
    building blocks, never sentences.
    """
    model = SelfModel()
    model.body_model.module_activity = {"reasoning": 0.9, "dreaming": 0.1}
    for seed in model.activity_seeds():
        assert " " not in seed
        assert not seed.endswith((".", "!", "?"))


def test_module_activity_at_zero_is_still_served() -> None:
    """An idle module stays in the report carrying an explicit zero.

    The daemon's manifest cpu_share is write-only, so an omitted field
    leaves the previous value in place. This is the case that used to
    pin a quiet module at its last busy-window load forever.
    """
    data = bytes([0]) + bytes([1]) + _module_rec(MODULE_EMOTION, 2, 0.0)
    report = SubsystemReport.unpack(data)
    assert len(report.modules) == 1
    assert report.modules[0].cpu_share == 0.0


def test_module_record_reads_task_and_error() -> None:
    """The 18-byte v3 record carries task and error alongside the share."""
    data = bytes([0]) + bytes([1])
    data += _module_rec_v3(MODULE_LANGUAGE, 2, 0.42, task_id=77, error_code=0)
    mod = SubsystemReport.unpack(data).modules[0]
    assert mod.module_name == "language"
    assert mod.status_name == "running"
    assert mod.cpu_share == pytest.approx(0.42)
    assert mod.task_id == 77
    assert mod.error_code == 0
    assert not mod.is_failed
    assert mod.is_busy
    assert mod.is_alive


def test_module_record_reports_failure() -> None:
    """A module that reported an error is distinguishable from a stopped one."""
    data = bytes([0]) + bytes([1])
    data += _module_rec_v3(MODULE_REASONING, 5, 0.0, task_id=0, error_code=3)
    mod = SubsystemReport.unpack(data).modules[0]
    assert mod.is_failed
    assert mod.status_name == "error"
    assert not mod.is_busy
    # Still "alive": the failure is what it is reporting, not absence.
    assert mod.is_alive


def test_module_record_v2_layout_reads_without_task_or_error() -> None:
    """A 6-byte v2-era record parses, with the optional tail left at zero."""
    data = bytes([0]) + bytes([1]) + _module_rec(MODULE_MEMORY, 2, 0.3)
    mod = SubsystemReport.unpack(data).modules[0]
    assert mod.module_name == "memory"
    assert mod.cpu_share == pytest.approx(0.3)
    assert mod.task_id == 0
    assert mod.error_code == 0


def test_module_record_idle_state() -> None:
    """An idle module is alive, not busy, and not failed."""
    data = bytes([0]) + bytes([1]) + _module_rec_v3(MODULE_METACOGNITION, 3, 0.0)
    mod = SubsystemReport.unpack(data).modules[0]
    assert mod.status_name == "idle"
    assert mod.is_alive
    assert not mod.is_busy
    assert not mod.is_failed


def test_module_record_stopped_is_not_alive() -> None:
    """Stopped and stopping both mean "not running"."""
    for status in (0, 4):
        data = bytes([0]) + bytes([1]) + _module_rec_v3(MODULE_DREAMING, status, 0.0)
        mod = SubsystemReport.unpack(data).modules[0]
        assert not mod.is_alive, f"status {status} should not be alive"


def test_types_are_frozen() -> None:
    """Wire records are immutable, so a consumer cannot corrupt them."""
    rec = SubsystemTelemetry(1, 2, 0.1, 0.1, 0.1, 0.1)
    with pytest.raises(FrozenInstanceError):
        rec.cpu = 0.9  # type: ignore[misc]
    mod = ModuleTelemetry(MODULE_MEMORY, 2, 0.1)
    with pytest.raises(FrozenInstanceError):
        mod.cpu_share = 0.9  # type: ignore[misc]


# ─── sensor presence ──────────────────────────────────────────


def test_presence_unpack_reads_the_mask() -> None:
    """The 2-byte response unpacks into a mask."""
    mask = SENSOR_TEMPERATURE | SENSOR_BATTERY
    presence = SensorPresence.unpack(struct.pack("<H", mask))
    assert presence.mask == mask
    assert presence.has(SENSOR_TEMPERATURE)
    assert presence.has(SENSOR_BATTERY)
    assert not presence.has(SENSOR_FAN)


def test_presence_short_response_is_empty_not_an_error() -> None:
    """A truncated response reads as "nothing verified"."""
    assert SensorPresence.unpack(b"").mask == 0
    assert SensorPresence.unpack(b"\x01").mask == 0


def test_presence_empty_mask_reports_everything_missing() -> None:
    """A zero mask means no sensor is confirmed present."""
    presence = SensorPresence(mask=0)
    assert presence.present == frozenset()
    assert len(presence.missing) == 16
    assert "battery" in presence.missing


def test_channel_verified_returns_bool_for_unknown_channel() -> None:
    """`channel_verified` yields a bool, never a truthy int.

    `has()` takes a *bit*; `channel_verified()` takes a *name* and must
    return a real bool. Returning `mask & bit` would leak 0/2/4/8… into
    a condition, which reads as truthy for any set bit.
    """
    presence = SensorPresence(mask=SENSOR_TEMPERATURE)
    result = presence.channel_verified("temperature")
    assert result is True
    assert presence.channel_verified("battery") is False
    assert presence.channel_verified("no such channel") is False


def test_presence_distinguishes_desktop_from_laptop() -> None:
    """A machine with no battery reports it missing, not as zero energy."""
    desktop = SensorPresence(
        mask=SENSOR_TEMPERATURE | SENSOR_FREQUENCY | SENSOR_MEMORY
    )
    assert "battery" in desktop.missing
    assert "fan PWM" in desktop.missing
    assert "temperature" in desktop.present
    assert not desktop.channel_verified("battery")
    assert desktop.channel_verified("temperature")


def test_presence_unknown_channel_is_not_verified() -> None:
    """An unrecognised channel cannot be claimed to be measured."""
    presence = SensorPresence(mask=0xFFFF)
    assert not presence.channel_verified("no such channel")


# ─── hardware_sensors_present reflects real hardware ──────────


def _body_state():
    return BodyState(
        cpu_temp_c=50.0, temperature=0.5, arousal_freq=0.6,
        cognitive_load=0.2, io_activity=0.1, stress_load=0.2,
        energy_reserve=1.0, on_ac_power=True, num_cores=8,
        distressed=False, autonomic_rate=3.0, thermoregulatory_effort=0.3,
        metabolic_rate=0.1, core_voltage=1.2, supply_voltage=12.0,
        description="",
    )


def test_sensors_present_counts_real_hardware_not_fields() -> None:
    """A desktop without battery/fan/RAPL counts fewer channels.

    This is the regression that made the guarantee vacuous: `hasattr`
    on a dataclass whose every field has a default is always true, so
    all 28 channels claimed to be present regardless of hardware.
    """
    desktop = SensorPresence(
        mask=SENSOR_TEMPERATURE | SENSOR_FREQUENCY | SENSOR_MAX_FREQUENCY
        | SENSOR_MEMORY | SENSOR_PSI
    )
    model = SelfModel()
    model.update_hardware_body(_body_state(), desktop)
    desktop_count = model.body_model.hardware_sensors_present

    full = SensorPresence(mask=0xFFFF)
    model_full = SelfModel()
    model_full.update_hardware_body(_body_state(), full)
    full_count = model_full.body_model.hardware_sensors_present

    assert full_count == 28, "all sensors present should count every channel"
    assert desktop_count < full_count, (
        "a machine with no battery, fan or RAPL domains must report fewer "
        "verified channels than one with everything"
    )
    # Self-measured channels hold on any Linux host regardless of sensors.
    assert desktop_count > 0


def test_sensors_present_zero_mask_verifies_only_always_available() -> None:
    """An empty mask still counts the channels that need no hardware."""
    model = SelfModel()
    model.update_hardware_body(_body_state(), SensorPresence(mask=0))
    count = model.body_model.hardware_sensors_present
    assert count > 0, "self-measured channels are real without any sensor"
    assert count < 28


def test_sensors_present_without_mask_keeps_legacy_behaviour() -> None:
    """An older daemon with no mask falls back to counting fields.

    Over-reporting is the safe failure: it claims a channel is present
    rather than pretending an absent sensor reads zero.
    """
    model = SelfModel()
    model.update_hardware_body(_body_state(), None)
    assert model.body_model.hardware_sensors_present == 28


# ─── zone / phase transition telemetry ────────────────────────


def _dwell_section(states: dict[int, tuple[int, int]]) -> bytes:
    """Encode a TRACKED_STATES-wide section, keyed by state index.

    The wire format is positional: record *i* describes state *i*, so
    a test supplies a mapping rather than a list to make that explicit
    and avoid misreading one as the other.
    """
    out = bytes([TRACKED_STATES])
    for idx in range(TRACKED_STATES):
        count, total_ms = states.get(idx, (0, 0))
        out += struct.pack("<IQ", count, total_ms)
    return out


def test_zone_transitions_unpack_both_sections() -> None:
    """Zone and phase tallies arrive as two sections."""
    zones = _dwell_section({1: (1, 5000), 3: (1, 2000)})
    phases = _dwell_section({7: (1, 90000)})
    report = ZoneTransitions.unpack(zones + phases)

    assert report.total_zone_transitions == 2
    assert report.total_phase_transitions == 1
    assert report.zones[1].count == 1
    assert report.zones[1].total_ms == 5000
    assert report.phases[7].count == 1
    assert report.phases[7].total_ms == 90000


def test_zone_transitions_mean_dwell_distinguishes_settled_from_oscillating() -> None:
    """A phase held once reads differently from one re-entered often."""
    phases = _dwell_section({4: (3, 3000), 1: (1, 60000)})
    report = ZoneTransitions.unpack(_dwell_section({}) + phases)

    oscillating = report.phases[4]
    settled = report.phases[1]
    assert oscillating.count == 3
    assert oscillating.mean_dwell_ms == pytest.approx(1000)
    assert settled.count == 1
    assert settled.mean_dwell_ms == pytest.approx(60000)
    # The settled one is the answer to "where does it rest".
    assert report.most_settled_phase() == settled


def test_zone_transitions_busiest_zone() -> None:
    """The most-entered zone is found by count."""
    zones = _dwell_section({0: (3, 30), 2: (1, 999999), 4: (1, 500)})
    report = ZoneTransitions.unpack(zones + _dwell_section({}))
    # State 2 has the most time but state 0 more entries; busiest is
    # about churn, not duration.
    assert report.busiest_zone() == report.zones[0]
    assert report.total_zone_transitions == 5


def test_zone_transitions_empty_is_all_zeroes() -> None:
    """A daemon that has recorded no transitions reports zeroes, not None."""
    report = ZoneTransitions.unpack(
        _dwell_section({}) + _dwell_section({})
    )
    assert report.total_zone_transitions == 0
    assert report.total_phase_transitions == 0
    assert report.busiest_zone() is None
    assert report.most_settled_phase() is None


def test_zone_transitions_truncated_section_raises() -> None:
    """A section claiming more records than it carries is rejected."""
    bad = bytes([TRACKED_STATES]) + struct.pack("<IQ", 1, 1000)  # claims 8, carries 1
    with pytest.raises(ValueError):
        ZoneTransitions.unpack(bad)


def test_zone_transitions_state_names_resolve() -> None:
    """Indices resolve to both zone and phase names."""
    report = ZoneTransitions.unpack(
        _dwell_section({0: (1, 1)}) + _dwell_section({1: (1, 1)})
    )
    assert report.zones[0].zone_name in ZONE_NAMES.values()
    assert report.phases[1].phase_name in PHASE_NAMES.values()


# ─── body condition seeds ─────────────────────────────────────


def _loaded_body():
    return BodyState(
        cpu_temp_c=90.0, temperature=0.9, arousal_freq=0.1,
        cognitive_load=0.2, io_activity=0.1, stress_load=0.2,
        energy_reserve=0.05, on_ac_power=False, num_cores=8,
        distressed=True, autonomic_rate=3.0, thermoregulatory_effort=0.9,
        metabolic_rate=0.1, core_voltage=1.2, supply_voltage=11.0,
        description="", throttle_state=0.8, psi_cpu=0.4, psi_io=0.2,
        psi_mem=0.1, battery_cycles=500.0, entropy_level=0.3,
        core_activity=0.9, uncore_activity=0.85, dram_activity=0.88,
    )


def test_body_conditions_surface_throttling_and_stalls() -> None:
    """The involuntary channels reach a consumer for the first time."""
    model = SelfModel()
    model.update_hardware_body(_loaded_body(), SensorPresence(mask=0xFFFF))
    names = dict(model.body_condition_seeds())
    assert "throttle" in names
    assert "stall" in names
    assert "hunger" in names
    # Ordered most urgent first.
    ranked = [s for s, _u in model.body_condition_seeds()]
    assert ranked.index("throttle") < ranked.index("hunger")


def test_body_conditions_require_verified_sensors() -> None:
    """A machine with no battery, fan or RAPL reports none of those."""
    desktop = SensorPresence(
        mask=SENSOR_TEMPERATURE | SENSOR_FREQUENCY | SENSOR_MEMORY | SENSOR_PSI
    )
    model = SelfModel()
    model.update_hardware_body(_loaded_body(), desktop)
    names = dict(model.body_condition_seeds())
    for absent in ("hunger", "heat", "power", "wear"):
        assert absent not in names, f"{absent} needs a sensor this machine lacks"
    # Throttling and stalls are real here and must still appear.
    assert "throttle" in names
    assert "stall" in names


def test_body_conditions_ignore_absent_sensors_entirely() -> None:
    """An empty mask verifies nothing, so nothing sensor-backed fires."""
    model = SelfModel()
    model.update_hardware_body(_loaded_body(), SensorPresence(mask=0))
    names = [s for s, _u in model.body_condition_seeds()]
    assert "throttle" not in names
    assert "hunger" not in names


def test_body_conditions_ignore_unreported_fields() -> None:
    """A field the daemon never sent is not read as a measurement."""

    class OldBody:
        cpu_temp_c = 50.0
        temperature = 0.5
        arousal_freq = 0.6
        cognitive_load = 0.2
        io_activity = 0.1
        stress_load = 0.2
        energy_reserve = 1.0
        on_ac_power = True
        num_cores = 8
        distressed = False
        autonomic_rate = 3.0
        thermoregulatory_effort = 0.3
        metabolic_rate = 0.1
        core_voltage = 1.2
        supply_voltage = 12.0
        description = ""

    model = SelfModel()
    model.update_hardware_body(OldBody(), None)
    # entropy_level is structurally 0.0 here, which would read as a
    # totally exhausted pool rather than "not reported".
    assert model.body_model.entropy_level == 0.0
    assert "entropy" not in dict(model.body_condition_seeds())


def test_body_condition_seeds_are_bare_names() -> None:
    """Conditions are building blocks, never pre-written sentences."""
    model = SelfModel()
    model.update_hardware_body(_loaded_body(), SensorPresence(mask=0xFFFF))
    for seed, urgency in model.body_condition_seeds():
        assert " " not in seed
        assert not seed.endswith((".", "!", "?"))
        assert urgency > 0.0


# ─── process health (GET_PROCESS_HEALTH) ──────────────────────


def test_process_health_reports_liveness() -> None:
    """A running tree reports each subsystem alive."""
    h = ProcessHealth.unpack(struct.pack("<HHB", 0b0111, 0, 0xFF))
    assert h.is_alive(SUBSYSTEM_DAEMON)
    assert h.is_alive(SUBSYSTEM_COGNITIVE)
    assert h.is_alive(SUBSYSTEM_RETINA)
    assert h.alive == frozenset({"daemon", "cognitive", "retina"})
    assert h.missing == frozenset()
    assert not h.has_lost_a_subsystem
    assert h.last_death_name == "none"


def test_process_health_reports_a_lost_subsystem() -> None:
    """A dead retina is countable, not merely absent.

    It used to vanish from the telemetry entirely, so a crashed retina
    looked exactly like one that was never started.
    """
    # daemon + cognitive alive, retina gone, one death recorded.
    h = ProcessHealth.unpack(struct.pack("<HHB", 0b0011, 1, 2))
    assert h.is_alive(SUBSYSTEM_DAEMON)
    assert not h.is_alive(SUBSYSTEM_RETINA)
    assert h.missing == frozenset({"retina"})
    assert h.has_lost_a_subsystem
    assert h.deaths == 1
    assert h.last_death_name == "retina"


def test_process_health_distinguishes_never_ran_from_died() -> None:
    """An absent subsystem with no deaths is not a crash."""
    h = ProcessHealth.unpack(struct.pack("<HHB", 0b0001, 0, 0xFF))
    assert h.missing == frozenset({"cognitive", "retina"})
    assert not h.has_lost_a_subsystem, "never running is not a death"


def test_process_health_short_response_is_empty() -> None:
    """A truncated record reports nothing confirmed rather than raising."""
    h = ProcessHealth.unpack(b"")
    assert h.alive_mask == 0
    assert h.deaths == 0
    assert h.missing == frozenset({"daemon", "cognitive", "retina"})


def test_process_health_unknown_death_is_named_not_dropped() -> None:
    """An unrecognised subsystem index is labelled."""
    h = ProcessHealth.unpack(struct.pack("<HHB", 0, 2, 99))
    assert h.last_death_name == "unknown:99"
