//! Ion electrochemistry — calcium, chloride, potassium, and sodium.
//!
//! This module is deliberately separate from [`NeurochemicalVector`]:
//! neurochemicals are normalized neuromodulatory control variables, while
//! ions are physical charge carriers with concentrations, electrochemical
//! gradients, reversal potentials, membrane voltage, and transporter
//! fluxes. Conflating the two layers would erase the distinction between
//! "how strongly a modulator is signaling" and "how much charge can move
//! across a membrane."
//!
//! # Model
//!
//! The state represents a whole-brain/population aggregate rather than a
//! single neuron. It uses the standard equivalent-circuit approximation:
//!
//! ```text
//! E_i = (RT / z_i F) ln([ion]_out / [ion]_in)       (Nernst)
//! I_i = g_i (V_m - E_i)                            (chord current)
//! V_m → (Σ g_i E_i - I_pump) / Σ g_i               (chord conductance)
//! ```
//!
//! Conductances are dimensionless relative conductances (K⁺ leak ≈ 1).
//! `current_density` is therefore a normalized `g·mV` quantity, not a
//! calibrated pA/mm² measurement. Concentrations are stored in mM and
//! membrane potential in mV. Calcium's much lower intracellular
//! concentration is represented directly (resting ≈ 0.0001 mM = 100 nM).
//!
//! Activity enters through the existing neurochemical state:
//!
//! - Glutamate opens Na⁺ and voltage-gated Ca²⁺ conductance.
//! - GABA opens Cl⁻ conductance.
//! - Adenosine suppresses Ca²⁺/Na⁺ channel drive and reports ATP load.
//! - K⁺ conductance contains a voltage-gated component plus a leak term.
//!
//! Homeostasis is explicit rather than folded into the channel dynamics:
//!
//! - Na⁺/K⁺-ATPase exports 3 Na⁺ and imports 2 K⁺ per cycle, gated by
//!   intracellular Na⁺, extracellular K⁺, and ATP availability.
//! - A signed K⁺/Cl⁻ cotransport term models net KCC2 extrusion (positive
//!   flux) and NKCC1-like import (negative flux).
//! - A Ca²⁺ extrusion term models PMCA/NCX clearance; its Na⁺-coupled
//!   component can weaken or reverse under intracellular Na⁺ load.
//!
//! This is not a Hodgkin–Huxley spike simulator. It is a bounded,
//! physically interpretable electrochemical layer that lets the abstract
//! neuromodulator system alter membrane excitability, chloride-dependent
//! inhibition, calcium-dependent plasticity, and metabolic pump load.
//!
//! References:
//! - Hodgkin & Huxley (1952) — ionic current formalism.
//! - Goldman (1943); Hodgkin & Katz (1949) — multi-ion membrane potential.
//! - Thomas (1972) — electrogenic Na⁺/K⁺-ATPase current.
//! - Payne et al. (2003) — neuronal chloride transporters.
//! - Berridge (1998) — intracellular calcium signaling.

use super::neurochemical::{NeurochemicalId, NeurochemicalVector};
use super::sanitize::{finite_clamp, finite_or};

/// Number of explicitly modeled ion species.
pub const ION_COUNT: usize = 4;

/// Stable ordering for the four ion arrays.
///
/// The order follows the user-facing naming order (calcium, chloride,
/// potassium, sodium) and is part of the persisted binary schema. Do not
/// reorder these discriminants.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IonId {
    /// Ca²⁺ — second messenger, plasticity, excitotoxic load.
    Calcium = 0,
    /// Cl⁻ — principal fast inhibitory charge carrier.
    Chloride = 1,
    /// K⁺ — dominant resting conductance and repolarizing gradient.
    Potassium = 2,
    /// Na⁺ — depolarizing gradient and Na⁺/K⁺ pump substrate.
    Sodium = 3,
}

impl IonId {
    /// All ion IDs in persisted array order.
    pub const fn all() -> [Self; ION_COUNT] {
        [Self::Calcium, Self::Chloride, Self::Potassium, Self::Sodium]
    }

    /// Convert a persisted discriminant to an ion ID.
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Calcium,
            1 => Self::Chloride,
            2 => Self::Potassium,
            _ => Self::Sodium,
        }
    }

    /// Human-readable ion name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Calcium => "calcium",
            Self::Chloride => "chloride",
            Self::Potassium => "potassium",
            Self::Sodium => "sodium",
        }
    }

    /// Ion valence used by the Nernst equation.
    const fn valence(self) -> f32 {
        match self {
            Self::Calcium => 2.0,
            Self::Chloride => -1.0,
            Self::Potassium | Self::Sodium => 1.0,
        }
    }

    /// Return this ion's index in the fixed-size ion arrays.
    const fn index(self) -> usize {
        self as usize
    }
}

/// Resting intracellular concentrations in mM, in `IonId` order.
///
/// Ca²⁺ is 100 nM (0.0001 mM), Cl⁻ is low intracellularly, K⁺ is the
/// dominant intracellular cation, and Na⁺ is low intracellularly.
pub const REST_INTRACELLULAR_MM: [f32; ION_COUNT] = [0.0001, 6.0, 140.0, 15.0];

/// Resting extracellular concentrations in mM, in `IonId` order.
pub const REST_EXTRACELLULAR_MM: [f32; ION_COUNT] = [1.2, 110.0, 5.0, 145.0];

/// Intracellular concentration safety bounds in mM.
const MIN_INTRACELLULAR_MM: [f32; ION_COUNT] = [0.00002, 1.0, 50.0, 2.0];
const MAX_INTRACELLULAR_MM: [f32; ION_COUNT] = [0.02, 30.0, 220.0, 80.0];

/// Extracellular concentration safety bounds in mM.
const MIN_EXTRACELLULAR_MM: [f32; ION_COUNT] = [0.5, 60.0, 1.0, 80.0];
const MAX_EXTRACELLULAR_MM: [f32; ION_COUNT] = [3.0, 160.0, 25.0, 200.0];

/// RT/F at 310 K expressed in mV.
const NERNST_FACTOR_MV: f32 = 26.713;
/// Resting membrane potential used to normalize excitability.
const RESTING_MEMBRANE_MV: f32 = -75.0;
/// Membrane integration time constant in seconds.
const MEMBRANE_TAU_S: f32 = 0.025;
/// Fraction of extracellular volume affected by a unit intracellular flux.
const EXTRACELLULAR_VOLUME_RATIO: f32 = 0.20;
/// Extracellular reservoir/glial homeostasis time constant in seconds.
const EXTRACELLULAR_TAU_S: f32 = 20.0;
/// Slow intracellular osmotic/metabolic restoration for Na⁺ and K⁺.
const INTRACELLULAR_TAU_S: f32 = 600.0;

/// Relative conductance activation time constants in seconds.
const CONDUCTANCE_TAU_S: [f32; ION_COUNT] = [0.020, 0.050, 0.080, 0.015];
/// Conversion from normalized `g·mV` current to intracellular mM/s.
const CURRENT_TO_MM_PER_S: [f32; ION_COUNT] = [0.00015, 0.008, 0.0024, 0.0035];

/// Maximum Na⁺/K⁺-ATPase cycle flux in mM/s. Each cycle exports 3 Na⁺
/// and imports 2 K⁺, so the Na⁺ rate is `3 × nak_pump_rate`.
const NAK_MAX_RATE_MM_PER_S: f32 = 0.042;
/// Normalized electrogenic pump contribution in the chord equation.
const PUMP_CURRENT_EQUIV_MV: f32 = 1.5;
/// Half-saturation constants for Na⁺i and K⁺o pump activation.
const NAK_KM_NA_I_MM: f32 = 10.0;
const NAK_KM_K_O_MM: f32 = 1.5;

/// Baseline outward Cl⁻ transport at the resting intracellular level.
const CL_TRANSPORT_BASE_MM_PER_S: f32 = 0.010;
/// Sensitivity of net KCC2/NKCC1 transport to intracellular Cl⁻ error.
const CL_TRANSPORT_GAIN_PER_S: f32 = 0.020;
/// K⁺ carried with the net chloride transport flux (KCC2/NKCC1 are
/// cation-chloride cotransporters).
const CL_TRANSPORT_K_STOICH: f32 = 0.5;

/// Baseline Ca²⁺ extrusion balancing resting Ca²⁺ leak/current.
const CA_EXTRUSION_BASE_MM_PER_S: f32 = 0.00016;
/// First-order PMCA/NCX clearance gain for free intracellular Ca²⁺.
const CA_EXTRUSION_GAIN_PER_S: f32 = 2.0;
/// High intracellular Na⁺ weakens/reverses Na⁺/Ca²⁺ exchange.
const NCX_REVERSE_GAIN_MM_PER_S: f32 = 0.0004;
/// Ca²⁺ level treated as a full-strength normalized signal.
const CA_SIGNAL_FULL_MM: f32 = 0.005;

/// Persistent electrochemical ion state.
///
/// # Layout (124 bytes)
///
/// ```text
/// offset  field                     type          size
/// ------  -----                     ----          ----
///   0     intracellular_mm          [f32;4]       16
///  16     extracellular_mm          [f32;4]       16
///  32     reversal_potential_mv     [f32;4]       16
///  48     conductance               [f32;4]       16
///  64     current_density           [f32;4]       16
///  80     membrane_potential_mv     f32            4
///  84     nak_pump_rate             f32            4
///  88     kcl_cotransporter_flux    f32            4
///  92     ncx_flux                  f32            4
///  96     atp_availability          f32            4
/// 100     calcium_signal            f32            4
/// 104     chloride_efficacy         f32            4
/// 108     excitability              f32            4
/// 112     gradient_integrity        f32            4
/// 116     energy_load               f32            4
/// 120     net_membrane_current      f32            4
/// 124     TOTAL                                  124 bytes
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IonState {
    /// Intracellular concentration in mM (`IonId` order).
    pub intracellular_mm: [f32; ION_COUNT],
    /// Extracellular concentration in mM (`IonId` order).
    pub extracellular_mm: [f32; ION_COUNT],
    /// Nernst reversal potential in mV (`IonId` order).
    pub reversal_potential_mv: [f32; ION_COUNT],
    /// Relative membrane conductance (`IonId` order; K⁺ leak ≈ 1).
    pub conductance: [f32; ION_COUNT],
    /// Normalized ionic current `g·(V-E)` (`IonId` order). Positive is
    /// outward conventional current; for Cl⁻ that corresponds to Cl⁻ influx.
    pub current_density: [f32; ION_COUNT],
    /// Aggregate membrane potential in mV.
    pub membrane_potential_mv: f32,
    /// Na⁺/K⁺-ATPase cycle rate in mM/s.
    pub nak_pump_rate: f32,
    /// Net K⁺/Cl⁻ cotransport in mM/s. Positive = outward Cl⁻/K⁺ flux
    /// (KCC2-like); negative = inward flux (NKCC1-like).
    pub kcl_cotransporter_flux: f32,
    /// Na⁺/Ca²⁺ exchange rate in mM/s of Ca²⁺. Positive = Ca²⁺ extrusion
    /// coupled to Na⁺ entry; negative = exchange reversal.
    pub ncx_flux: f32,
    /// ATP availability for ion pumps and transporters [0, 1].
    pub atp_availability: f32,
    /// Log-normalized intracellular Ca²⁺ signal [0, 1].
    pub calcium_signal: f32,
    /// Effective inhibitory Cl⁻ driving force [0, 1].
    pub chloride_efficacy: f32,
    /// Depolarization above the resting potential, normalized [0, 1].
    pub excitability: f32,
    /// Mean preservation of all four resting electrochemical gradients [0, 1].
    pub gradient_integrity: f32,
    /// Normalized channel/transporter workload [0, 1].
    pub energy_load: f32,
    /// Sum of channel currents plus electrogenic pump current.
    pub net_membrane_current: f32,
}

impl IonState {
    /// Fresh physiological resting state.
    pub fn new() -> Self {
        let mut state = Self {
            intracellular_mm: REST_INTRACELLULAR_MM,
            extracellular_mm: REST_EXTRACELLULAR_MM,
            reversal_potential_mv: [0.0; ION_COUNT],
            conductance: [0.0055, 0.475, 0.945, 0.11],
            current_density: [0.0; ION_COUNT],
            membrane_potential_mv: RESTING_MEMBRANE_MV,
            nak_pump_rate: 0.0,
            kcl_cotransporter_flux: 0.0,
            ncx_flux: 0.0,
            atp_availability: 0.9,
            calcium_signal: 0.0,
            chloride_efficacy: 0.5,
            excitability: 0.0,
            gradient_integrity: 1.0,
            energy_load: 0.25,
            net_membrane_current: 0.0,
        };
        state.recompute_derived();
        state
    }

    /// Get an intracellular concentration by ion ID.
    pub fn intracellular(&self, ion: IonId) -> f32 {
        self.intracellular_mm[ion.index()]
    }

    /// Get an extracellular concentration by ion ID.
    pub fn extracellular(&self, ion: IonId) -> f32 {
        self.extracellular_mm[ion.index()]
    }

    /// Get a Nernst reversal potential by ion ID.
    pub fn reversal_potential(&self, ion: IonId) -> f32 {
        self.reversal_potential_mv[ion.index()]
    }

    /// Calcium-dependent plasticity modulation for the memory gating layer.
    ///
    /// Resting Ca²⁺ leaves the neurochemical plasticity gate at ~70% of its
    /// neurotransmitter-derived value; strong but bounded Ca²⁺ transients
    /// restore full plasticity. Extreme Ca²⁺ is clamped rather than allowed
    /// to amplify plasticity without bound.
    pub fn plasticity_modulator(&self) -> f32 {
        finite_clamp(0.70 + 0.60 * self.calcium_signal, 0.0, 1.0)
    }

    /// Advance ion electrochemistry by `dt` seconds, driven by the current
    /// effective neurochemical state.
    ///
    /// The update is stable at the daemon's 100 ms tick: conductance and
    /// voltage approach their targets by exact exponentials rather than
    /// explicit Euler steps, while concentration changes are bounded and
    /// followed by physiological range clamps.
    pub fn tick(&mut self, neuro: &NeurochemicalVector, dt: f32) {
        self.sanitize();
        let dt = finite_clamp(finite_or(dt, super::neurochemical::DT), 0.001, 1.0);

        let glutamate = finite_clamp(neuro.effective(NeurochemicalId::Glutamate), 0.0, 2.0);
        let gaba = finite_clamp(neuro.effective(NeurochemicalId::GABA), 0.0, 2.0);
        let acetylcholine = finite_clamp(neuro.effective(NeurochemicalId::Acetylcholine), 0.0, 2.0);
        let adenosine = finite_clamp(neuro.effective(NeurochemicalId::Adenosine), 0.0, 1.0);
        let cortisol = finite_clamp(neuro.effective(NeurochemicalId::Cortisol), 0.0, 1.0);

        let voltage = self.membrane_potential_mv;
        let sodium_gate = sigmoid((voltage + 35.0) / 6.0);
        let calcium_gate = sigmoid((voltage + 45.0) / 6.0);

        let target = [
            // Voltage-gated Ca²⁺: glutamate opens NMDA/L-type-like influx,
            // while adenosine presynaptically suppresses Ca²⁺ entry.
            0.005 + 0.15 * glutamate * calcium_gate * (1.0 - 0.5 * adenosine),
            // GABA-A-like Cl⁻ conductance plus a resting leak.
            0.20 + 0.55 * gaba,
            // K⁺ leak plus voltage-gated delayed-rectifier-like activation;
            // adenosine/GABA modestly increase outward stabilizing current.
            0.90 + 0.40 * sigmoid((voltage + 35.0) / 10.0) + 0.08 * adenosine + 0.04 * gaba,
            // AMPA-like glutamate drive plus voltage-gated Na⁺ recruitment.
            (0.035 + 0.15 * glutamate + 0.20 * glutamate * sodium_gate + 0.03 * acetylcholine)
                * (1.0 - 0.25 * adenosine),
        ];
        for i in 0..ION_COUNT {
            let alpha = 1.0 - (-dt / CONDUCTANCE_TAU_S[i]).exp();
            self.conductance[i] += (target[i] - self.conductance[i]) * alpha;
            self.conductance[i] = finite_clamp(self.conductance[i], 0.0, 4.0);
        }

        // Na⁺/K⁺-ATPase: 3 Na⁺ out, 2 K⁺ in per ATP. Occupancy terms make
        // the pump self-limiting and responsive to Na⁺ load / K⁺ availability.
        let na_occupancy = self.intracellular_mm[IonId::Sodium as usize]
            / (self.intracellular_mm[IonId::Sodium as usize] + NAK_KM_NA_I_MM);
        let ko_occupancy = self.extracellular_mm[IonId::Potassium as usize]
            / (self.extracellular_mm[IonId::Potassium as usize] + NAK_KM_K_O_MM);
        self.nak_pump_rate = finite_clamp(
            NAK_MAX_RATE_MM_PER_S * na_occupancy * ko_occupancy * self.atp_availability,
            0.0,
            NAK_MAX_RATE_MM_PER_S,
        );

        // Chord-conductance voltage target. The Na⁺/K⁺ pump contributes a
        // small net outward (hyperpolarizing) current due to its 3:2 ratio.
        let pump_current = PUMP_CURRENT_EQUIV_MV * (self.nak_pump_rate / NAK_MAX_RATE_MM_PER_S);
        let mut conductance_sum = 0.0;
        let mut voltage_sum = 0.0;
        for i in 0..ION_COUNT {
            conductance_sum += self.conductance[i];
            voltage_sum += self.conductance[i] * self.reversal_potential_mv[i];
        }
        let voltage_target = if conductance_sum > 0.001 {
            (voltage_sum - pump_current) / conductance_sum
        } else {
            RESTING_MEMBRANE_MV
        };
        let voltage_alpha = 1.0 - (-dt / MEMBRANE_TAU_S).exp();
        self.membrane_potential_mv += (voltage_target - self.membrane_potential_mv) * voltage_alpha;
        self.membrane_potential_mv = finite_clamp(self.membrane_potential_mv, -120.0, 80.0);

        // Channel currents and concentration changes. Positive current is
        // outward conventional current. For cations, outward current removes
        // intracellular ion; for Cl⁻, outward conventional current is an
        // inward anion flux.
        let mut channel_flux = [0.0f32; ION_COUNT];
        let mut channel_load = 0.0;
        for (i, ion) in IonId::all().into_iter().enumerate() {
            self.current_density[i] =
                self.conductance[i] * (self.membrane_potential_mv - self.reversal_potential_mv[i]);
            let inward_flux = if ion == IonId::Chloride {
                self.current_density[i]
            } else {
                -self.current_density[i]
            } * CURRENT_TO_MM_PER_S[i];
            channel_flux[i] = inward_flux;
            channel_load += inward_flux.abs();
            self.intracellular_mm[i] += inward_flux * dt;
            self.extracellular_mm[i] -= inward_flux * EXTRACELLULAR_VOLUME_RATIO * dt;
        }

        // Na⁺/K⁺ pump stoichiometry.
        let nak = self.nak_pump_rate;
        let na_i = IonId::Sodium as usize;
        let k_i = IonId::Potassium as usize;
        self.intracellular_mm[na_i] -= 3.0 * nak * dt;
        self.extracellular_mm[na_i] += 3.0 * nak * EXTRACELLULAR_VOLUME_RATIO * dt;
        self.intracellular_mm[k_i] += 2.0 * nak * dt;
        self.extracellular_mm[k_i] -= 2.0 * nak * EXTRACELLULAR_VOLUME_RATIO * dt;

        // Net cation-chloride transport. Positive flux is KCC2-like outward
        // transport; negative flux is NKCC1-like inward transport.
        let cl_i = IonId::Chloride as usize;
        let cl_transport = finite_clamp(
            CL_TRANSPORT_BASE_MM_PER_S
                + CL_TRANSPORT_GAIN_PER_S
                    * (self.intracellular_mm[cl_i] - REST_INTRACELLULAR_MM[cl_i]),
            -0.08,
            0.12,
        );
        self.kcl_cotransporter_flux = cl_transport;
        self.intracellular_mm[cl_i] -= cl_transport * dt;
        self.extracellular_mm[cl_i] += cl_transport * EXTRACELLULAR_VOLUME_RATIO * dt;
        self.intracellular_mm[k_i] -= CL_TRANSPORT_K_STOICH * cl_transport * dt;
        self.extracellular_mm[k_i] +=
            CL_TRANSPORT_K_STOICH * cl_transport * EXTRACELLULAR_VOLUME_RATIO * dt;

        // PMCA/NCX Ca²⁺ extrusion. Positive ncx_flux extrudes one Ca²⁺ and
        // imports approximately three Na⁺. High intracellular Na⁺ weakens
        // the forward exchange and can reverse it under sustained load.
        let ca_i = IonId::Calcium as usize;
        let na_overload = ((self.intracellular_mm[na_i] - 20.0) / 20.0).max(0.0);
        let ncx = finite_clamp(
            CA_EXTRUSION_BASE_MM_PER_S
                + CA_EXTRUSION_GAIN_PER_S
                    * (self.intracellular_mm[ca_i] - REST_INTRACELLULAR_MM[ca_i])
                - NCX_REVERSE_GAIN_MM_PER_S * na_overload,
            -0.002,
            0.02,
        );
        self.ncx_flux = ncx;
        self.intracellular_mm[ca_i] -= ncx * dt;
        self.extracellular_mm[ca_i] += ncx * EXTRACELLULAR_VOLUME_RATIO * dt;
        self.intracellular_mm[na_i] += 3.0 * ncx * dt;
        self.extracellular_mm[na_i] -= 3.0 * ncx * EXTRACELLULAR_VOLUME_RATIO * dt;

        // Extracellular reservoirs and glial buffering pull concentrations
        // back toward their physiological bath values. Slow intracellular
        // restoration bounds residual Na⁺/K⁺ drift without hiding the
        // explicit pump fluxes above.
        let extracellular_alpha = 1.0 - (-dt / EXTRACELLULAR_TAU_S).exp();
        for (i, &rest) in REST_EXTRACELLULAR_MM.iter().enumerate() {
            self.extracellular_mm[i] += (rest - self.extracellular_mm[i]) * extracellular_alpha;
        }
        let intracellular_alpha = 1.0 - (-dt / INTRACELLULAR_TAU_S).exp();
        for i in [na_i, k_i] {
            self.intracellular_mm[i] +=
                (REST_INTRACELLULAR_MM[i] - self.intracellular_mm[i]) * intracellular_alpha;
        }

        // Metabolic load follows absolute channel and transporter throughput.
        // ATP availability is the bridge back to the ion pumps: adenosine is
        // treated as an ATP-depletion signal, cortisol adds a smaller stress
        // cost, and sustained ionic workload consumes capacity.
        let transporter_load = nak / 0.05 + cl_transport.abs() / 0.05 + ncx.abs() / 0.001;
        let load_target = finite_clamp((channel_load / 0.20 + transporter_load) / 2.5, 0.0, 1.0);
        let load_alpha = 1.0 - (-dt / 5.0).exp();
        self.energy_load += (load_target - self.energy_load) * load_alpha;
        let atp_target = finite_clamp(
            1.0 - 0.35 * adenosine - 0.35 * self.energy_load - 0.10 * cortisol,
            0.25,
            1.0,
        );
        let atp_alpha = 1.0 - (-dt / 2.0).exp();
        self.atp_availability += (atp_target - self.atp_availability) * atp_alpha;

        self.recompute_derived();
        self.sanitize();
    }

    /// Recompute reversal potentials, currents, and normalized summaries.
    fn recompute_derived(&mut self) {
        let mut conductance_sum = 0.0;
        let mut gradient_sum = 0.0;
        let mut net_current = PUMP_CURRENT_EQUIV_MV * (self.nak_pump_rate / NAK_MAX_RATE_MM_PER_S);
        for (i, ion) in IonId::all().into_iter().enumerate() {
            let inside = finite_clamp(
                finite_or(self.intracellular_mm[i], REST_INTRACELLULAR_MM[i]),
                MIN_INTRACELLULAR_MM[i],
                MAX_INTRACELLULAR_MM[i],
            );
            let outside = finite_clamp(
                finite_or(self.extracellular_mm[i], REST_EXTRACELLULAR_MM[i]),
                MIN_EXTRACELLULAR_MM[i],
                MAX_EXTRACELLULAR_MM[i],
            );
            self.reversal_potential_mv[i] =
                (NERNST_FACTOR_MV / ion.valence()) * (outside / inside).ln();
            self.current_density[i] =
                self.conductance[i] * (self.membrane_potential_mv - self.reversal_potential_mv[i]);
            conductance_sum += self.conductance[i];
            net_current += self.current_density[i];

            let gradient_ratio = (inside / outside).ln();
            let rest_ratio = (REST_INTRACELLULAR_MM[i] / REST_EXTRACELLULAR_MM[i]).ln();
            gradient_sum += finite_clamp(gradient_ratio / rest_ratio, 0.0, 1.0);
        }
        self.net_membrane_current = finite_clamp(net_current, -1000.0, 1000.0);
        self.gradient_integrity = finite_clamp(gradient_sum / ION_COUNT as f32, 0.0, 1.0);

        let ca_i = IonId::Calcium as usize;
        let ca_ratio = finite_clamp(
            self.intracellular_mm[ca_i] / REST_INTRACELLULAR_MM[ca_i],
            1.0,
            CA_SIGNAL_FULL_MM / REST_INTRACELLULAR_MM[ca_i],
        );
        self.calcium_signal = finite_clamp(
            ca_ratio.ln() / (CA_SIGNAL_FULL_MM / REST_INTRACELLULAR_MM[ca_i]).ln(),
            0.0,
            1.0,
        );

        let cl_i = IonId::Chloride as usize;
        let cl_drive = self.membrane_potential_mv - self.reversal_potential_mv[cl_i];
        self.chloride_efficacy = finite_clamp(0.5 + cl_drive / 20.0, 0.0, 1.0);
        self.excitability = finite_clamp(
            (self.membrane_potential_mv - RESTING_MEMBRANE_MV) / 60.0,
            0.0,
            1.0,
        );

        if !conductance_sum.is_finite() {
            self.conductance = [0.0055, 0.475, 0.945, 0.11];
        }
    }

    /// Clamp/replace all persisted fields into physiologically bounded,
    /// finite ranges. This is the NaN/∞ circuit breaker for state loaded
    /// from mmap or perturbed by malformed inputs.
    pub fn sanitize(&mut self) {
        for i in 0..ION_COUNT {
            self.intracellular_mm[i] = finite_clamp(
                finite_or(self.intracellular_mm[i], REST_INTRACELLULAR_MM[i]),
                MIN_INTRACELLULAR_MM[i],
                MAX_INTRACELLULAR_MM[i],
            );
            self.extracellular_mm[i] = finite_clamp(
                finite_or(self.extracellular_mm[i], REST_EXTRACELLULAR_MM[i]),
                MIN_EXTRACELLULAR_MM[i],
                MAX_EXTRACELLULAR_MM[i],
            );
            self.reversal_potential_mv[i] =
                finite_clamp(self.reversal_potential_mv[i], -200.0, 250.0);
            self.conductance[i] = finite_clamp(self.conductance[i], 0.0, 4.0);
            self.current_density[i] = finite_clamp(self.current_density[i], -1000.0, 1000.0);
        }
        self.membrane_potential_mv = finite_clamp(
            finite_or(self.membrane_potential_mv, RESTING_MEMBRANE_MV),
            -120.0,
            80.0,
        );
        self.nak_pump_rate = finite_clamp(self.nak_pump_rate, 0.0, NAK_MAX_RATE_MM_PER_S);
        self.kcl_cotransporter_flux = finite_clamp(self.kcl_cotransporter_flux, -0.08, 0.12);
        self.ncx_flux = finite_clamp(self.ncx_flux, -0.002, 0.02);
        self.atp_availability = finite_clamp(self.atp_availability, 0.25, 1.0);
        self.calcium_signal = finite_clamp(self.calcium_signal, 0.0, 1.0);
        self.chloride_efficacy = finite_clamp(self.chloride_efficacy, 0.0, 1.0);
        self.excitability = finite_clamp(self.excitability, 0.0, 1.0);
        self.gradient_integrity = finite_clamp(self.gradient_integrity, 0.0, 1.0);
        self.energy_load = finite_clamp(self.energy_load, 0.0, 1.0);
        self.net_membrane_current = finite_clamp(self.net_membrane_current, -1000.0, 1000.0);
    }
}

impl Default for IonState {
    /// Return the physiological resting ion state.
    fn default() -> Self {
        Self::new()
    }
}

/// Numerically stable logistic function used for bounded pump modulation.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x > 20.0 {
        1.0
    } else if x < -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

// ─── Compile-time layout assertions ──────────────────────────────

const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<IonState>() == 124);
    assert!(offset_of!(IonState, intracellular_mm) == 0);
    assert!(offset_of!(IonState, extracellular_mm) == 16);
    assert!(offset_of!(IonState, reversal_potential_mv) == 32);
    assert!(offset_of!(IonState, conductance) == 48);
    assert!(offset_of!(IonState, current_density) == 64);
    assert!(offset_of!(IonState, membrane_potential_mv) == 80);
    assert!(offset_of!(IonState, nak_pump_rate) == 84);
    assert!(offset_of!(IonState, kcl_cotransporter_flux) == 88);
    assert!(offset_of!(IonState, ncx_flux) == 92);
    assert!(offset_of!(IonState, atp_availability) == 96);
    assert!(offset_of!(IonState, calcium_signal) == 100);
    assert!(offset_of!(IonState, chloride_efficacy) == 104);
    assert!(offset_of!(IonState, excitability) == 108);
    assert!(offset_of!(IonState, gradient_integrity) == 112);
    assert!(offset_of!(IonState, energy_load) == 116);
    assert!(offset_of!(IonState, net_membrane_current) == 120);
};
