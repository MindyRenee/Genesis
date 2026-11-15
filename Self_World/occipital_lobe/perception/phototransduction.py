"""Phototransduction cascade — enzymatic model of cone photoreceptor signaling.

This module implements a mathematical model of the enzymatic cascade that
converts photon absorption into electrical signals in cone photoreceptors.
The model captures the key enzyme kinetics:

    Photon → R* (activated visual pigment)
    R* + Transducin → T* (activated G-protein)
    T* + PDE → PDE* (activated phosphodiesterase)
    PDE* hydrolyzes cGMP → GMP (Michaelis-Menten kinetics)
    cGMP ↓ → CNG channels close → hyperpolarization
    Ca²⁺ ↓ (via exchanger) → disinhibits GC → cGMP synthesis ↑
    Rhodopsin kinase + Arrestin → R* deactivation

The model is based on the Hamer phototransduction model and experimental
data from primate cones (Koch & Lamb, 2020; Nikonov et al., 2006).

Key equations:
    d[R*]/dt = I * k_R - k_R_inact * [R*]
    d[PDE*]/dt = k_act * [R*] - k_deact * [PDE*]
    d[cGMP]/dt = V_GC - V_PDE
    V_PDE = (kcat * [PDE*] * [cGMP]) / (Km + [cGMP])
    V_GC = V_GC_max * (1 / (1 + ([Ca]/K_Ca)^n))  (Ca-inhibited)

Reference:
    "Speed, sensitivity, and stability of the light response in rod and cone
    photoreceptors: Facts and models" (Koch & Lamb, 2020)
    "Mathematical analysis of phototransduction reaction parameters in rods
    and cones" (Scientific Reports, 2022)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class PhototransductionState:
    """The dynamic state of the phototransduction cascade."""

    # Concentrations (arbitrary units, normalized to dark-adapted baseline)
    r_star: float = 0.0  # Activated visual pigment (R*)
    pde_star: float = 0.0  # Activated phosphodiesterase (PDE*)
    cgmp: float = 1.0  # cGMP concentration (normalized: 1.0 = dark level)
    calcium: float = 1.0  # Calcium concentration (normalized: 1.0 = dark level)

    # Membrane response (hyperpolarization). CNG current, normalized:
    # ~0.5 in dark (half-max at cGMP=1.0); light suppresses it toward 0.
    photocurrent: float = 0.0

    # Time since last photon (for adaptation tracking)
    time_since_light: float = 0.0


@dataclass
class PhototransductionParams:
    """Kinetic parameters for the phototransduction cascade.

    Parameters are tuned for primate cones (faster kinetics than rods).
    Values are normalized to dimensionless units for computational stability.
    """

    # Visual pigment activation/inactivation
    k_r: float = 1.0  # Photon absorption rate constant (scales with intensity)
    k_r_inact: float = 50.0  # R* inactivation rate (cones: ~50 s^-1, rods: ~5 s^-1)

    # PDE activation/inactivation
    k_act: float = 30.0  # R* → PDE* activation rate (cones: faster than rods)
    k_deact: float = 18.0  # PDE* deactivation rate (cones: ~18 s^-1, rods: ~1.8 s^-1)

    # PDE enzymatic kinetics (Michaelis-Menten)
    kcat: float = 500.0  # Turnover rate per PDE* molecule (s^-1) - increased for stronger response
    km_cgmp: float = 0.5  # Michaelis constant for cGMP (normalized)

    # Guanylate cyclase (cGMP synthesis)
    v_gc_max: float = 500.0  # Maximum GC synthesis rate (normalized to match PDE kcat)
    k_ca: float = 0.25  # Calcium inhibition constant (normalized)
    n_ca: float = 4.0  # Hill coefficient for Ca inhibition (cooperative)

    # Calcium dynamics
    ca_dark: float = 1.0  # Dark calcium level (normalized)
    ca_exchange_rate: float = 10.0  # Ca exchanger rate (s^-1)

    # CNG channel gating
    cgmp_half: float = 1.0  # cGMP concentration for half-maximal current (set to dark level)
    n_channel: float = 2.0  # Hill coefficient for channel gating

    # Dark-adapted cGMP level (normalized)
    cgmp_dark: float = 1.0

    # Integration time step (seconds)
    dt: float = 0.001  # 1 ms integration step


class PhototransductionCascade:
    """Mathematical model of the cone phototransduction enzyme cascade.

    This model simulates the enzymatic reactions that convert light
    absorption into electrical signals in cone photoreceptors. It captures
    the cascade kinetics, calcium feedback, and adaptation dynamics.

    The model is stateful: it maintains the current concentrations of
    all molecular species and integrates forward in time given light input.
    """

    def __init__(self, params: PhototransductionParams | None = None) -> None:
        """Initialize the phototransduction cascade.

        Args:
            params: Kinetic parameters. If None, uses default cone parameters.
        """
        self.params = params or PhototransductionParams()
        self.state = PhototransductionState()
        self._time = 0.0
        # Compute initial photocurrent from dark-adapted cGMP level
        self._update_photocurrent()

    def _update_photocurrent(self) -> None:
        """Update photocurrent from current cGMP level."""
        p = self.params
        s = self.state
        cgmp_power = s.cgmp**p.n_channel
        half_power = p.cgmp_half**p.n_channel
        s.photocurrent = cgmp_power / (cgmp_power + half_power)

    def reset(self) -> None:
        """Reset the cascade to dark-adapted baseline."""
        self.state = PhototransductionState()
        self._time = 0.0
        self._update_photocurrent()

    def step(self, light_intensity: float) -> PhototransductionState:
        """Advance the cascade by one time step given light input.

        Args:
            light_intensity: Normalized light intensity (0.0 = dark, 1.0 = bright)

        Returns:
            The updated state after integration.
        """
        p = self.params
        s = self.state

        # ── Visual pigment activation (R*) ─────────────────────────────
        # Photon absorption creates R*; inactivation removes it
        dr_star = (p.k_r * light_intensity) - (p.k_r_inact * s.r_star)
        s.r_star += dr_star * p.dt
        s.r_star = max(0.0, s.r_star)

        # ── PDE activation (PDE*) ───────────────────────────────────────
        # R* activates PDE; PDE* deactivates spontaneously
        dpde_star = (p.k_act * s.r_star) - (p.k_deact * s.pde_star)
        s.pde_star += dpde_star * p.dt
        s.pde_star = max(0.0, s.pde_star)

        # ── cGMP hydrolysis by PDE* (Michaelis-Menten) ────────────────
        # V_PDE = (kcat * [PDE*] * [cGMP]) / (Km + [cGMP])
        # Include basal PDE activity (0.1% of kcat) for dark hydrolysis
        basal_pde = 0.001 * p.kcat
        v_pde = ((p.kcat * s.pde_star + basal_pde) * s.cgmp) / (p.km_cgmp + s.cgmp)

        # ── cGMP synthesis by GC (calcium-inhibited) ─────────────────
        # V_GC = V_GC_max / (1 + ([Ca]/K_Ca)^n)
        ca_inhibition = 1.0 + (s.calcium / p.k_ca) ** p.n_ca
        v_gc = p.v_gc_max / ca_inhibition

        # ── cGMP dynamics ─────────────────────────────────────────────
        # Scale rates to concentration units (normalize by kcat for stability)
        dcgmp = (v_gc - v_pde) / p.kcat
        s.cgmp += dcgmp * p.dt
        s.cgmp = max(0.01, min(2.0, s.cgmp))  # Clamp to physiological range

        # ── Calcium dynamics (exchanger-driven) ───────────────────────
        # Calcium follows cGMP: lower cGMP → closed channels → lower Ca
        # Ca exchanger pulls Ca down proportional to current
        target_ca = s.cgmp  # Simplified: Ca tracks cGMP in steady state
        dca = p.ca_exchange_rate * (target_ca - s.calcium)
        s.calcium += dca * p.dt
        s.calcium = max(0.01, min(2.0, s.calcium))

        # ── Photocurrent (CNG channel response) ───────────────────────
        # Current = [cGMP]^n / ([cGMP]^n + K_half^n)
        # In dark: cGMP is high → channels open → high current
        # In light: cGMP drops → channels close → low current (hyperpolarization)
        self._update_photocurrent()

        # ── Track time since light ────────────────────────────────────
        if light_intensity > 0.01:
            s.time_since_light = 0.0
        else:
            s.time_since_light += p.dt

        self._time += p.dt
        return s

    def simulate_flash_response(
        self,
        flash_intensity: float,
        flash_duration: float,
        total_time: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the response to a light flash.

        Args:
            flash_intensity: Intensity of the flash (0-1)
            flash_duration: Duration of the flash (seconds)
            total_time: Total simulation time (seconds)

        Returns:
            (time, photocurrent, cgmp, pde_star) arrays
        """
        self.reset()

        n_steps = int(total_time / self.params.dt)
        time = np.zeros(n_steps)
        photocurrent = np.zeros(n_steps)
        cgmp = np.zeros(n_steps)
        pde_star = np.zeros(n_steps)

        for i in range(n_steps):
            t = i * self.params.dt
            light = flash_intensity if t < flash_duration else 0.0
            state = self.step(light)

            time[i] = t
            photocurrent[i] = state.photocurrent
            cgmp[i] = state.cgmp
            pde_star[i] = state.pde_star

        return time, photocurrent, cgmp, pde_star

    def steady_state_response(self, light_intensity: float) -> PhototransductionState:
        """Compute the steady-state response to constant light.

        Args:
            light_intensity: Constant light intensity (0-1)

        Returns:
            The steady-state after 1 second of integration.
        """
        self.reset()
        for _ in range(int(1.0 / self.params.dt)):
            self.step(light_intensity)
        return self.state

    def sensitivity(self, light_intensity: float = 0.0) -> float:
        """Compute the cascade's incremental light sensitivity.

        Sensitivity is the magnitude of the photocurrent-suppression
        slope at an operating point — |dI/dL| — normalized to the
        dark-adapted slope so the dark value is 1.0. Light suppresses
        the CNG current (hyperpolarization), so the raw derivative is
        negative; the magnitude is what "sensitivity" means here.

        What the model actually does at these input magnitudes: the
        calcium-on-guanylate-cyclase feedback holds the operating
        point nearly fixed, so the slope stays ≈ 1 across the nominal
        range — honest to report; Weber-law desensitization would only
        appear where the response finally compresses. The measurement
        integrates the cascade to steady state under the hood; the
        live state (adaptation included) is restored afterward.

        Args:
            light_intensity: Background level to probe at (0-1).

        Returns:
            Sensitivity relative to the dark-adapted cascade (0-1).
        """
        snapshot = copy.deepcopy(self.state)
        saved_time = self._time
        epsilon = 0.01

        def slope_at(level: float) -> float:
            i_lo = self.steady_state_response(level).photocurrent
            i_hi = self.steady_state_response(level + epsilon).photocurrent
            return (i_lo - i_hi) / epsilon

        try:
            dark_slope = slope_at(0.0)
            if dark_slope <= 0.0:
                return 0.0
            sensitivity = slope_at(light_intensity) / dark_slope
        finally:
            self.state = snapshot
            self._time = saved_time
        return max(0.0, min(1.0, sensitivity))


def cone_vs_rod_comparison() -> dict[str, PhototransductionParams]:
    """Return parameter sets comparing cone and rod phototransduction.

    Cones have faster kinetics (higher inactivation rates) and lower
    sensitivity than rods. This function returns both parameter sets
    for comparative simulation.

    Returns:
        dict with 'cone' and 'rod' parameter sets
    """
    # Cone parameters (default, faster kinetics)
    cone_params = PhototransductionParams(
        k_r_inact=50.0,  # Fast R* inactivation
        k_act=30.0,  # Fast PDE activation
        k_deact=18.0,  # Fast PDE deactivation
    )

    # Rod parameters (slower kinetics, higher sensitivity)
    rod_params = PhototransductionParams(
        k_r_inact=5.0,  # Slow R* inactivation (10x slower than cones)
        k_act=30.0,  # Similar activation
        k_deact=1.8,  # Slow PDE deactivation (10x slower than cones)
    )

    return {"cone": cone_params, "rod": rod_params}
