"""Tests for the optional PyTorch light-cone implementation."""

import pytest

torch = pytest.importorskip("torch")

from temporal_lobe.light_cone_torch import TorchLightConeField  # noqa: E402


def test_torch_light_cone_preserves_causal_boundary_and_axis_convention():
    field = TorchLightConeField(signal_speed=3.0, angular_speed=0.7)
    times = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    angles = torch.zeros_like(times)
    points = field(times, angles)

    assert torch.allclose(points[:, 0], times)
    assert torch.allclose(
        points[:, 1].square() + points[:, 2].square(),
        (field.signal_speed * times.abs()).square(),
        atol=1e-5,
    )
    assert torch.allclose(points[2], torch.zeros(3))


def test_torch_counter_rotation_has_opposite_phase_evolution():
    field = TorchLightConeField(angular_speed=0.5)
    positive = field(torch.tensor([1.0, 2.0]), torch.zeros(2))
    negative = field(torch.tensor([-1.0, -2.0]), torch.zeros(2))

    phase_positive = torch.atan2(positive[:, 2], positive[:, 1])
    phase_negative = torch.atan2(negative[:, 2], negative[:, 1])
    assert phase_positive[1] > phase_positive[0]
    assert phase_negative[1] < phase_negative[0]


def test_torch_geometry_is_differentiable():
    field = TorchLightConeField(signal_speed=2.0, angular_speed=0.2)
    times = torch.tensor([1.0, 2.0])
    points = field(times, torch.zeros_like(times))
    loss = points[:, 1:].square().sum()
    loss.backward()

    assert field.log_signal_speed.grad is not None
    assert field.angular_speed.grad is not None
    assert torch.isfinite(field.log_signal_speed.grad)
    assert torch.isfinite(field.angular_speed.grad)


def test_torch_elastic_response_and_plastic_memory_are_separate():
    field = TorchLightConeField(
        elastic_rate=1.0,
        plastic_rate=0.25,
        initial_dimensions=1,
    )
    response = field.observe(torch.tensor([4.0 + 2.0j]))
    assert torch.allclose(response, torch.tensor([4.0 + 2.0j]))
    assert torch.allclose(field.plastic_state, torch.tensor([1.0 + 0.5j]))
    assert torch.allclose(field.elastic_state, response)

    relaxed = field.relax(1.0)
    assert torch.allclose(relaxed, field.plastic_state)


def test_torch_field_expands_without_silently_shrinking():
    field = TorchLightConeField(plastic_rate=0.0)
    field.observe(torch.tensor([1.0, 2.0, 3.0]))
    field.observe(torch.tensor([5.0]))
    assert field.plastic_state.numel() == 3
    assert torch.allclose(field.plastic_state, torch.zeros(3, dtype=torch.complex64))


def test_torch_spine_and_sunburst_use_separate_axes():
    field = TorchLightConeField()
    spine = field.spine(torch.tensor([-2.0, 0.0, 2.0]))
    rays = field.time_sunburst(8, 5.0)

    assert torch.allclose(spine[:, 0], torch.zeros(3))
    assert torch.allclose(spine[:, 1], torch.zeros(3))
    assert torch.allclose(spine[:, 2], torch.tensor([-2.0, 0.0, 2.0]))
    assert torch.allclose(rays[:, 2], torch.zeros(8))
    assert torch.allclose(
        torch.linalg.vector_norm(rays[:, :2], dim=1),
        torch.full((8,), 5.0),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal_speed": 0.0},
        {"elastic_rate": -0.1},
        {"plastic_rate": 1.1},
        {"angular_speed": float("nan")},
    ],
)
def test_torch_constructor_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        TorchLightConeField(**kwargs)


def test_torch_encoder_propagates_gradients():
    field = TorchLightConeField()
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    encoded = field.encode(values)
    encoded.abs().sum().backward()
    assert values.grad is not None
    assert field.log_amplitude_gain.grad is not None
    assert field.phase_bias.grad is not None
