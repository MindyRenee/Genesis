"""PyTorch implementation of Genesis's adaptive light-cone field.

PyTorch is an optional dependency. This module provides differentiable
light-cone coordinates and a complex-valued adaptive field. The coordinate
embedding is a visualization of a causal boundary, not a physical claim about
cognition. Persistent plastic updates are explicit state changes, not gradient
descent; trainable parameters can separately be optimized with autograd.
"""

from __future__ import annotations

import math

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - depends on optional installation
    raise ImportError(
        "PyTorch support is optional. Install a compatible PyTorch build "
        "before importing temporal_lobe.light_cone_torch."
    ) from exc


class TorchLightConeField(nn.Module):
    """Differentiable causal geometry plus elastic/plastic complex field.

    Coordinates use x as horizontal time and (y, z) as transverse coordinates.
    The future and past sheets counter-rotate when measured outward from the
    shared apex. The radius remains exactly c*abs(t), so rotation cannot alter
    the represented light-cone boundary.

    plastic_state is a persistent buffer; elastic_state is a temporary buffer.
    Call observe to update both explicitly. Use forward for differentiable
    coordinates and encode for differentiable field output.
    """

    def __init__(
        self,
        *,
        signal_speed: float = 1.0,
        angular_speed: float = 0.0,
        elastic_rate: float = 0.5,
        plastic_rate: float = 0.05,
        initial_dimensions: int = 0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        numeric = (signal_speed, angular_speed, elastic_rate, plastic_rate)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("all scalar parameters must be finite")
        if signal_speed <= 0:
            raise ValueError("signal_speed must be positive")
        if not 0.0 <= elastic_rate <= 1.0:
            raise ValueError("elastic_rate must be in [0, 1]")
        if not 0.0 <= plastic_rate <= 1.0:
            raise ValueError("plastic_rate must be in [0, 1]")
        if (
            isinstance(initial_dimensions, bool)
            or not isinstance(initial_dimensions, int)
            or initial_dimensions < 0
        ):
            raise ValueError("initial_dimensions must be a non-negative integer")

        self.log_signal_speed = nn.Parameter(
            torch.tensor(math.log(signal_speed), dtype=dtype, device=device)
        )
        self.angular_speed = nn.Parameter(torch.tensor(angular_speed, dtype=dtype, device=device))
        self.phase_bias = nn.Parameter(torch.zeros((), dtype=dtype, device=device))
        self.log_amplitude_gain = nn.Parameter(torch.zeros((), dtype=dtype, device=device))
        self.elastic_rate = float(elastic_rate)
        self.plastic_rate = float(plastic_rate)
        self.register_buffer(
            "plastic_state",
            torch.zeros(initial_dimensions, dtype=torch.complex64, device=device),
        )
        self.register_buffer(
            "elastic_state",
            torch.zeros(initial_dimensions, dtype=torch.complex64, device=device),
        )

    @property
    def signal_speed(self) -> Tensor:
        """Positive learnable characteristic propagation speed."""
        return self.log_signal_speed.exp()

    def forward(self, time: Tensor, angle: Tensor) -> Tensor:
        """Map times and angles to [..., 3] points on the causal boundary.

        Positive time selects the future sheet, negative time the past sheet.
        Opposite phase signs encode counter-rotation while preserving
        y**2 + z**2 == (signal_speed * abs(time))**2.
        """
        time, angle = torch.broadcast_tensors(time, angle)
        if not torch.is_floating_point(time) or not torch.is_floating_point(angle):
            raise TypeError("time and angle tensors must use floating-point dtypes")
        phase = angle - torch.sign(time) * self.angular_speed * time.abs() + self.phase_bias
        radius = self.signal_speed * time.abs()
        y = radius * torch.cos(phase)
        z = radius * torch.sin(phase)
        return torch.stack((time, y, z), dim=-1)

    def encode(self, observation: Tensor) -> Tensor:
        """Differentiably apply trainable gain and phase to complex features.

        Real-valued feature tensors are accepted and converted to complex
        values. This method does not mutate persistent state.
        """
        if not (observation.is_floating_point() or observation.is_complex()):
            raise TypeError("observation must be floating-point or complex")
        if observation.numel() == 0:
            raise ValueError("observation must not be empty")
        if not torch.isfinite(observation).all():
            raise ValueError("observation must contain only finite values")
        complex_observation = observation.to(torch.complex64)
        gain = self.log_amplitude_gain.exp()
        phase = self.phase_bias
        phasor = torch.complex(torch.cos(phase), torch.sin(phase))
        return gain * phasor * complex_observation

    @torch.no_grad()
    def observe(self, observation: Tensor) -> Tensor:
        """Apply temporary elastic response and persist a plastic update.

        Feature capacity expands to fit new observations and never silently
        shrinks. Newly added dimensions start at zero. Returns the temporary
        response; persistent memory moves toward the observation at plastic_rate.
        """
        if not (observation.is_floating_point() or observation.is_complex()):
            raise TypeError("observation must be floating-point or complex")
        if observation.numel() == 0:
            raise ValueError("observation must not be empty")
        if not torch.isfinite(observation).all():
            raise ValueError("observation must contain only finite values")

        values = (
            observation.detach()
            .to(
                device=self.plastic_state.device,
                dtype=self.plastic_state.dtype,
            )
            .reshape(-1)
        )
        old_size = self.plastic_state.numel()
        size = max(old_size, values.numel())
        if size > old_size:
            extension = torch.zeros(
                size - old_size,
                dtype=self.plastic_state.dtype,
                device=self.plastic_state.device,
            )
            self.plastic_state = torch.cat((self.plastic_state, extension))
            self.elastic_state = torch.cat((self.elastic_state, extension.clone()))

        padded = torch.zeros_like(self.plastic_state)
        padded[: values.numel()] = values
        prior = self.plastic_state.clone()
        response = prior + self.elastic_rate * (padded - prior)
        self.plastic_state.copy_(prior + self.plastic_rate * (padded - prior))
        self.elastic_state.copy_(response)
        return response.clone()

    @torch.no_grad()
    def relax(self, recovery_rate: float = 0.5) -> Tensor:
        """Relax temporary state toward persistent memory without unlearning."""
        if not math.isfinite(recovery_rate) or not 0.0 <= recovery_rate <= 1.0:
            raise ValueError("recovery_rate must be finite and in [0, 1]")
        self.elastic_state.add_(recovery_rate * (self.plastic_state - self.elastic_state))
        return self.elastic_state.clone()

    def spine(self, distance: Tensor) -> Tensor:
        """Return points on the vertical spine through Self, shape [..., 3]."""
        zero = torch.zeros_like(distance)
        return torch.stack((zero, zero, distance), dim=-1)

    def time_sunburst(self, ray_count: int, radius: float) -> Tensor:
        """Return horizontal time-sunburst endpoints, shape [ray_count, 3]."""
        if isinstance(ray_count, bool) or not isinstance(ray_count, int) or ray_count < 2:
            raise ValueError("ray_count must be an integer of at least 2")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        angles = torch.arange(
            ray_count,
            dtype=self.log_signal_speed.dtype,
            device=self.log_signal_speed.device,
        ) * (2.0 * math.pi / ray_count)
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        z = torch.zeros_like(x)
        return torch.stack((x, y, z), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"elastic_rate={self.elastic_rate}, "
            f"plastic_rate={self.plastic_rate}, "
            f"dimensions={self.plastic_state.numel()}"
        )
