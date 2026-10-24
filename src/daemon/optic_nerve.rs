//! Optic nerve — the retinal afferent's subcortical branches into the body.
//!
//! The `retina` capture process writes raw frames into the POSIX
//! shared-memory segment `/genesis_retina` — that segment is the optic
//! nerve. Anatomically the nerve does not reach cortex directly: its
//! ~10⁶ ganglion-cell axons fan out to roughly forty subcortical
//! targets, and only the thalamocortical radiation carries what the
//! thalamus relays onward. The subcortical recipients matter as much
//! as the relay:
//!
//! - **Suprachiasmatic nucleus (SCN)** — retinal illuminance via the
//!   retinohypothalamic tract, carried almost exclusively by
//!   melanopsin ipRGCs (~480 nm peak). This is the dominant circadian
//!   zeitgeber. It feeds `sleep_network::VisualAfference::light`,
//!   replacing the wall-clock photoperiod whenever the eye is live.
//! - **Superior colliculus (SC)** — motion/luminance transients drive
//!   orienting. During sleep the same channel is the sensory-gaining
//!   "wake-up call": salient stimuli are still analysed and can
//!   arouse, so the transient becomes a brief excitatory pulse onto
//!   the LC wake population.
//! - **Pretectum (OPN)** — global illuminance for the light reflex;
//!   published as `illuminance` for whatever consumes it. The reflex
//!   itself runs here: `pupil_mm` is the sphincter/dilator balance —
//!   log-compressive constriction to the photopic channel plus a
//!   sympathetic dilation from effective norepinephrine — published
//!   so the mind's optics layer can scale retinal irradiance and
//!   depth of field by the aperture the body actually set.
//! - **dLGN + thalamic reticular nucleus (TRN)** — the thalamic relay
//!   is *gated*, not a wire: TRN inhibition during NREM hyperpolarizes
//!   relay cells and sensory transmission to cortex is blocked at the
//!   thalamus. `relay_gain` is that gate, computed here in the body
//!   where the sleep network lives. The mind reads it via
//!   `GET_OPTIC_STATE` and honours it — cortex cannot see around a
//!   closed thalamus.
//! - **Corticothalamic feedback** — V1 layer-6 projections back to
//!   the dLGN outnumber retinal input ~10:1 but are modulatory, not
//!   driving. The mind's engagement arrives via `SET_OPTIC_GAIN` and
//!   scales the relay within ±40%, never opening it past what the
//!   sleep stage permits.
//!
//! The body's own ocular state lives here too: the pupil (pretectal
//! light reflex + LC sympathetic dilation), the ciliary lens power
//! (accommodation, set by the mind's `SET_FOCUS`), and the lids —
//! the eye the hardware lacks. The lid is a phase-driven
//! transmission on every afferent channel: closed in NREM/REM (a
//! few percent leak — light through lids still entrains weakly and
//! a flash can still wake), half-mast drowsy, open awake. Sleep
//! miosis and the ciliary's release to rest ride the same closure.
//!
//! What is deliberately NOT here: no pixels leave this module. The
//! frame scan computes a handful of scalar channel signals (the same
//! thing the RHT, tectal, and pretectal fibres carry — there is no
//! image on those axons either). Image-forming content stays on the
//! shared-memory bus for the mind's V1, gated by `relay_gain`.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::state::sanitize::finite_clamp;
use crate::state::zones::MentalPhase;

/// POSIX name of the retinal segment — the same constant the `retina`
/// binary writes and the Python mind reads (`src/bin/retina.rs`,
/// `occipital_lobe/perception/retina.py`).
pub const SHM_NAME: &str = "/genesis_retina";

/// Header layout written by the retina process:
/// `[u64 seqlock][u32 width][u32 height][4B fourcc][u32 size][u32 stride]`.
const HEADER_SIZE: usize = 28;

/// A frame is "live" while the last consumed frame is younger than
/// this. Past it the feed is stale (camera stalled, retina dead) and
/// the circadian input falls back to the photoperiod.
const FRESH_WINDOW_MS: u64 = 5_000;

/// Pixels sampled per frame read. The subcortical channels carry
/// spatially pooled signals — a few thousand points is already far
/// denser than the melanopsin/tectal receptive fields being modelled.
const MAX_SAMPLES: usize = 2048;

/// Melanopic irradiance scaled onto the sleep oscillator's lux axis
/// (`I_0` in `sleep_network`; the scheduled photoperiod uses 500).
/// A webcam has no absolute photometry, so this is a shaped proxy:
/// gamma 1.3 keeps dim evening light visible to the SCN (melanopsin
/// phase-shifts at single-digit lux) while a bright room approaches
/// daylight drive.
const MELANOPIC_LUX_SCALE: f64 = 2500.0;
const MELANOPIC_LUX_GAMMA: f64 = 1.3;

/// TRN relay bases by state — the fraction of retinal drive the
/// thalamus passes to cortex. NREM is the documented block (~zero);
/// REM gates external input almost as hard (its imagery is
/// internally generated). Drowsy is partial transmission; Overwhelmed
/// is sensory shutdown short of sleep.
fn relay_base(phase: MentalPhase) -> f32 {
    match phase {
        MentalPhase::NREM => 0.02,
        MentalPhase::REM => 0.10,
        MentalPhase::Drowsy => 0.6,
        MentalPhase::Overwhelmed => 0.3,
        _ => 1.0,
    }
}

/// Corticothalamic feedback range around the relay base: modulatory,
/// not driving (Sherman & Guillery) — it can amplify or attenuate the
/// relay but cannot open a gate the sleep stage has closed.
const CORTICAL_FLOOR: f32 = 0.6;
const CORTICAL_SPAN: f32 = 0.4;

/// Eyelid transmission by phase — the fraction of scene light that
/// still reaches the retina. The camera hardware has no lid; this is
/// the body's *use* of its afference, the same status as the TRN
/// gate. Lids are not shutters: a closed lid transmits a few percent,
/// so bright light still entrains weakly through it and a strong
/// flash still reaches the colliculus — both are real physiology.
/// Drowsy lids sit half-mast. The mind's voluntary closure
/// (`SET_LID`) can only deepen what the phase sets — a will can
/// shut an open eye, never force a sleeping one open.
fn lid_transmission(phase: MentalPhase) -> f32 {
    match phase {
        MentalPhase::NREM | MentalPhase::REM => 0.03,
        MentalPhase::Drowsy => 0.5,
        _ => 1.0,
    }
}

/// Central miosis with the lids closed — in sleep the sphincter
/// dominates (parasympathetic tone; LC is near-silent in NREM and
/// silent in REM) and the pupil sits miotic ~2–4 mm whatever the
/// room's darkness says. Scaled by lid closure: half-mast lids
/// carry half the drive.
const PUPIL_SLEEP_CONSTRICT_MM: f32 = 4.5;
/// Lid closure at or above this counts the eye as shut — the point
/// where the ciliary muscle releases its request and rests.
const LID_CLOSED: f32 = 0.8;

/// Pupil diameter bounds in mm — the physiological range of the
/// sphincter/dilator antagonists.
const PUPIL_MIN_MM: f32 = 2.0;
const PUPIL_MAX_MM: f32 = 8.0;
/// Diameter the reflex starts from before the first integration —
/// mid-range, neither dark- nor light-adapted.
const PUPIL_REST_MM: f32 = 4.5;
/// The light reflex is log-compressive: diameter tracks log
/// illuminance, not linear luminance (Watson & Yellott's unified
/// field formula has the same shape). This span maps the nerve's
/// [0, 1] photopic channel onto a ~4.5 mm reflex swing: darkness
/// approaches `PUPIL_MAX_MM`, a saturated channel sits near 3.5 mm.
const PUPIL_LIGHT_SPAN_MM: f32 = 4.5;
/// Sympathetic dilation per unit of effective norepinephrine — LC
/// tone pulls the dilator against the sphincter's set point. Tonic
/// NE (~0.3) adds ~0.5 mm; a saturated channel adds ~1.5 mm.
const PUPIL_NE_GAIN_MM: f32 = 1.5;
/// Reflex time constants. Constriction is the fast parasympathetic
/// arc (a light step covers most of its travel in well under a
/// second); dilation is the slower sympathetic mix, several times
/// slower.
const PUPIL_TAU_CONSTRICT_S: f32 = 0.4;
const PUPIL_TAU_DILATE_S: f32 = 1.5;

/// Accommodation bounds in dioptres — the added lens power over the
/// distance refraction. 0 D is focus at infinity; 6 D is a ~17 cm
/// near point, a generous accommodative amplitude.
const ACCOM_MIN_DPT: f32 = 0.0;
const ACCOM_MAX_DPT: f32 = 6.0;
/// Tonic resting position — the ciliary muscle's anatomical rest is
/// midrange, about a metre of focus in the dark.
const ACCOM_REST_DPT: f32 = 1.0;
/// Accommodation time constants. Releasing the lens back to distance
/// (disaccommodation) is the faster direction — the ciliary muscle
/// relaxes onto the choroid's elastic recoil — while pulling power in
/// for near focus takes longer (Schaeffel et al., 1993; Beers & Van
/// Der Heijde, 1994; Vilupuru & Glasser, 2002; Kalsi et al., 2003 —
/// the opposite asymmetry to the pupil's antagonists).
const ACCOM_TAU_NEAR_S: f32 = 1.3;
const ACCOM_TAU_FAR_S: f32 = 0.7;

/// The optic nerve, attached to the body. Held by `TickLoop`; sampled
/// once per physics advance so the afference fed to the integrator
/// describes the same instant as the chemistry it rides beside.
pub struct OpticNerve {
    /// Segment name — a parameter so tests can use private segments.
    shm_name: String,
    /// The open shared-memory attachment; `None` until the retina
    /// process creates the segment, re-tried every sample.
    shm: Option<shared_memory::Shmem>,
    /// Last consumed seqlock value — a frame is consumed once, so an
    /// unchanged counter means the retina has not written since.
    last_seq: u64,
    /// `current_ms` of the last consumed frame (0 = never).
    last_fresh_ms: u64,
    /// Subsampled luma of the previous frame — the motion baseline.
    prev_luma: Vec<u8>,
    /// Photopic illuminance EMA — mean luma in [0, 1]. The pretectal
    /// channel.
    illuminance: f32,
    /// Blue-weighted irradiance EMA in [0, 1] — the ipRGC/melanopsin
    /// proxy (~480 nm peak → blue-weighted; achromatic under YUYV).
    melanopic: f32,
    /// Motion energy EMA in [0, 1] — pooled frame-to-frame luma
    /// change, the tectal channel's slow trace.
    motion: f32,
    /// This-sample orienting burst in [0, 1] — the SC's transient,
    /// not smoothed: a burst that decays is what alerting means.
    orienting: f32,
    /// The mind's corticothalamic feedback in [0, 1], set via
    /// `SET_OPTIC_GAIN`. 0.5 is neutral.
    cortical_gain: f32,
    /// The mind's voluntary lid openness in [0, 1], set via
    /// `SET_LID`. 1.0 is the unrequested state — fully open. The
    /// effective lid is `phase.min(voluntary)`: a will can lower
    /// the lids on an open phase (closing the eyes to rest them or
    /// shut out light) but can never raise them past what the
    /// sleep stage allows.
    voluntary_lid: f32,
    /// The phase the gate was last derived from — kept so a
    /// `SET_OPTIC_GAIN` between advances can re-derive the relay
    /// immediately instead of waiting for the next tick.
    gate_phase: MentalPhase,
    /// Current thalamic relay gain, recomputed each advance from the
    /// sleep stage and the cortical feedback.
    relay_gain: f32,
    /// Pupil diameter in mm — the body's own aperture, integrated on
    /// the advance clock from the light reflex and LC sympathetic
    /// tone. The mind reads it to scale retinal irradiance; the nerve
    /// only publishes it, it does not feed it back into the channels
    /// it was measured from.
    pupil_mm: f32,
    /// `current_ms` of the last pupil integration (0 = never — the
    /// first call seeds the clock).
    pupil_last_ms: u64,
    /// Current lens power in dioptres — the ciliary state, integrated
    /// on the advance clock toward `accom_target_dpt`. 0 D is focus
    /// at infinity, 6 D a ~17 cm near point.
    accommodation_dpt: f32,
    /// The power the ciliary muscle is driving toward, set by the
    /// mind's focus request in meters (`SET_FOCUS`) converted to
    /// dioptres.
    accom_target_dpt: f32,
    /// `current_ms` of the last accommodation integration (0 = never).
    accom_last_ms: u64,
    /// Frames consumed since attach.
    frames_seen: u64,
}

// SAFETY: `OpticNerve` holds a `shared_memory::Shmem`, which is !Send
// only because `MapData` stores the mapping's `*mut u8`. The pointer
// names a process-wide POSIX shared-memory mapping owned by the retina
// process (this attachment is never the owner and never unlinks), and
// every access through it here is a read bounded by the segment length.
// Moving the handle between threads moves no ownership of the mapping
// and enables no aliased mutation, so Send is sound.
unsafe impl Send for OpticNerve {}

/// What one `GET_OPTIC_STATE` returns — the nerve's measured channel
/// signals and the thalamic gate. Process-local, additive on the wire.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpticState {
    /// The segment is attached and frames were ever consumed.
    pub eye_present: bool,
    /// A new frame was consumed within [`FRESH_WINDOW_MS`].
    pub frames_fresh: bool,
    /// Photopic illuminance EMA [0, 1].
    pub illuminance: f32,
    /// Melanopic (blue-weighted) irradiance EMA [0, 1].
    pub melanopic: f32,
    /// Motion energy EMA [0, 1].
    pub motion: f32,
    /// Orienting transient [0, 1] — salient change *now*.
    pub orienting: f32,
    /// Thalamic relay gain [0, 1] — what fraction of retinal drive
    /// cortex may receive (TRN gate × corticothalamic feedback).
    pub relay_gain: f32,
    /// The corticothalamic feedback currently applied.
    pub cortical_gain: f32,
    /// Illuminance on the SCN's lux scale — what the oscillator is
    /// actually being fed (0 when the photoperiod is in charge).
    pub retinal_lux: f32,
    /// Last consumed frame counter — freshness the mind can compare
    /// against the seq it reads from the segment header.
    pub frame_seq: u64,
    /// Milliseconds since the last consumed frame; -1 when none yet.
    pub frame_age_ms: i64,
    /// Pupil diameter in mm — the body's aperture: the light reflex
    /// plus LC sympathetic tone. The mind's optics layer scales
    /// retinal irradiance and depth of field by it.
    pub pupil_mm: f32,
    /// Lens power in dioptres — the ciliary state. 0 D is focus at
    /// infinity (or unreported by an older daemon); the focus
    /// distance is `1/dpt` metres.
    pub accommodation_dpt: f32,
    /// Eyelid transmission [0, 1] — how much scene light the phase's
    /// lid position still passes. 1.0 awake, ~0.5 drowsy, ~0.03 in
    /// NREM/REM (0 also means unreported by an older daemon — the
    /// mind reads it as fully open rather than fully shut, so a
    /// legacy body never looks asleep to the optics layer).
    pub lid: f32,
}

impl OpticState {
    /// Wire layout after the ack byte — fixed-size, additive:
    /// `[u8 flags][u8 _pad][7 × f32][u64 seq][i64 age_ms][f32 pupil_mm]
    /// [f32 accommodation_dpt][f32 lid]` = 58 bytes. flags bit0 =
    /// eye_present, bit1 = frames_fresh. Daemons that predate the
    /// trailing fields send shorter bodies; decoders read pupil/lens
    /// as 0 ("unreported") and lid as "open" (1.0).
    pub fn to_wire(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(58);
        out.push((self.eye_present as u8) | ((self.frames_fresh as u8) << 1));
        out.push(0);
        for v in [
            self.illuminance,
            self.melanopic,
            self.motion,
            self.orienting,
            self.relay_gain,
            self.cortical_gain,
            self.retinal_lux,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.frame_seq.to_le_bytes());
        out.extend_from_slice(&self.frame_age_ms.to_le_bytes());
        out.extend_from_slice(&self.pupil_mm.to_le_bytes());
        out.extend_from_slice(&self.accommodation_dpt.to_le_bytes());
        out.extend_from_slice(&self.lid.to_le_bytes());
        out
    }
}

impl OpticNerve {
    /// Attach to the production retinal segment.
    pub fn new() -> Self {
        Self::with_name(SHM_NAME)
    }

    /// Attach to a named segment (tests use private names).
    pub fn with_name(name: &str) -> Self {
        Self {
            shm_name: name.to_string(),
            shm: None,
            last_seq: 0,
            last_fresh_ms: 0,
            prev_luma: Vec::new(),
            illuminance: 0.0,
            melanopic: 0.0,
            motion: 0.0,
            orienting: 0.0,
            cortical_gain: 0.5,
            voluntary_lid: 1.0,
            gate_phase: MentalPhase::Active,
            relay_gain: 1.0,
            pupil_mm: PUPIL_REST_MM,
            pupil_last_ms: 0,
            accommodation_dpt: ACCOM_REST_DPT,
            accom_target_dpt: ACCOM_REST_DPT,
            accom_last_ms: 0,
            frames_seen: 0,
        }
    }

    /// Sample the nerve once: seqlock-read the header, and when a new
    /// frame has arrived pool it into the channel signals. Absent or
    /// stalled segments are the no-eye condition — the afference just
    /// reports no fresh input, it is never a fault.
    pub fn sample(&mut self, now_ms: u64) {
        // A stale feed can mean the mapping is a corpse: the retina
        // unlinks the segment on exit, and a reincarnated retina
        // creates a *new* object under the same name — this handle
        // still points at the dead one, whose counter is frozen
        // forever. Once frames have been seen, a feed that stays
        // stale past the fresh window is dropped so the open below
        // re-resolves the name to whatever segment exists now. A
        // merely-paused writer re-opens the same object harmlessly.
        if self.shm.is_some()
            && self.frames_seen > 0
            && now_ms.saturating_sub(self.last_fresh_ms) > FRESH_WINDOW_MS
        {
            self.shm = None;
        }
        if self.shm.is_none() {
            match shared_memory::ShmemConf::new().os_id(&self.shm_name).open() {
                Ok(s) => {
                    // The retina owns the segment's lifetime; we are
                    // a recipient, so unlinking is never ours to do.
                    self.shm = Some(s);
                }
                Err(_) => return,
            }
        }
        let shm = self.shm.as_ref().expect("checked above");
        let base = shm.as_ptr();

        // Seqlock read, same protocol as the Python reader and the
        // retina writer: even counter = complete frame, odd = write in
        // progress, changed across the read = torn. The pixel pooling
        // runs inside the bracket too — stats accumulate into locals
        // and only commit after the second counter read verifies.
        for _ in 0..8 {
            // SAFETY: `base` points into a live shm mapping of at
            // least HEADER_SIZE bytes once a frame header has been
            // initialised; the counter is the writer's own AtomicU64
            // and is only touched atomically here.
            let c1 = unsafe { (*(base as *const AtomicU64)).load(Ordering::Acquire) };
            if c1 & 1 != 0 || c1 == 0 {
                // Write in progress, or the segment exists but no
                // frame has ever been published — not an error.
                if c1 == 0 {
                    break;
                }
                std::thread::yield_now();
                continue;
            }
            // SAFETY: volatile reads within the mapped header; bounds
            // fixed by HEADER_SIZE.
            let (width, height, fourcc, size, stride) = unsafe {
                let w = std::ptr::read_volatile(base.add(8) as *const u32);
                let h = std::ptr::read_volatile(base.add(12) as *const u32);
                let mut fcc = [0u8; 4];
                std::ptr::copy_nonoverlapping(base.add(16), fcc.as_mut_ptr(), 4);
                let sz = std::ptr::read_volatile(base.add(20) as *const u32);
                let st = std::ptr::read_volatile(base.add(24) as *const u32);
                (w, h, fcc, sz, st)
            };
            if c1 == self.last_seq {
                // No new frame since the last sample — feed is live
                // but unchanged, or stalled; freshness is decided by
                // last_fresh_ms either way.
                return;
            }
            let total = (size as usize).min(shm.len().saturating_sub(HEADER_SIZE));
            if total == 0 || width == 0 || height == 0 {
                self.last_seq = c1;
                return;
            }
            let pooled = Self::pool_frame(base, width, height, fourcc, total, stride);
            // SAFETY: same live mapping and AtomicU64 counter as the
            // c1 read above — this second load closes the seqlock
            // bracket, so a torn frame shows as c1 != c2 and retries.
            let c2 = unsafe { (*(base as *const AtomicU64)).load(Ordering::Acquire) };
            if c1 != c2 {
                std::thread::yield_now();
                continue;
            }
            self.commit(pooled);
            self.last_seq = c1;
            self.last_fresh_ms = now_ms;
            self.frames_seen += 1;
            return;
        }
    }

    /// Pool one frame's subsampled pixels into (luma, melanopic)
    /// channel sums. Runs inside the seqlock bracket; writes nothing
    /// on `self` until `commit` verifies the counter.
    fn pool_frame(
        base: *const u8,
        width: u32,
        height: u32,
        fourcc: [u8; 4],
        size: usize,
        stride: u32,
    ) -> Vec<u8> {
        let (bpp, rgb) = match &fourcc {
            b"RGB3" | b"BGR3" => (3usize, true),
            b"YUYV" => (2usize, false),
            _ => (1usize, false),
        };
        let bgr = &fourcc == b"BGR3";
        let row_bytes = (width as usize).saturating_mul(bpp);
        let stride = (stride as usize).max(row_bytes).max(1);
        let px = (width as usize).saturating_mul(height as usize);
        let step = (px / MAX_SAMPLES).max(1);

        // Interleaved (luma, melanopic) byte pairs per sampled pixel —
        // one pass, one allocation.
        let mut out = Vec::with_capacity(px / step * 2 + 2);
        let mut i = 0usize;
        while i < px {
            let off = HEADER_SIZE + (i / width as usize) * stride + (i % width as usize) * bpp;
            if off + bpp > HEADER_SIZE + size {
                break;
            }
            // SAFETY: `off + bpp` was bounds-checked against the frame
            // payload above; `base` is the live shm mapping.
            let (y, mel) = unsafe {
                if rgb {
                    let (r, g, b) = if bgr {
                        (*base.add(off + 2), *base.add(off + 1), *base.add(off))
                    } else {
                        (*base.add(off), *base.add(off + 1), *base.add(off + 2))
                    };
                    // Rec.601 luma; melanopsin ~480 nm peaks blue —
                    // green carries the long-wavelength tail.
                    let y = (0.299 * f64::from(r) + 0.587 * f64::from(g) + 0.114 * f64::from(b))
                        / 255.0;
                    let mel = (0.30 * f64::from(g) + 0.70 * f64::from(b)) / 255.0;
                    (y, mel)
                } else {
                    let y = f64::from(*base.add(off)) / 255.0;
                    (y, y) // achromatic: irradiance is its own proxy
                }
            };
            out.push((y * 255.0) as u8);
            out.push((mel * 255.0) as u8);
            i += step;
        }
        out
    }

    /// Commit a seqlock-verified pooled frame into the channel EMAs.
    fn commit(&mut self, pooled: Vec<u8>) {
        let n = (pooled.len() / 2).max(1) as f64;
        let luma_mean = pooled.iter().step_by(2).map(|&v| f64::from(v)).sum::<f64>() / (n * 255.0);
        let mel_mean = pooled
            .iter()
            .skip(1)
            .step_by(2)
            .map(|&v| f64::from(v))
            .sum::<f64>()
            / (n * 255.0);

        // Motion: pooled per-cell absolute luma change, scaled — a
        // light switch moves nearly every cell (≈1.0), ordinary scene
        // motion a small fraction.
        let luma: Vec<u8> = pooled.iter().step_by(2).copied().collect();
        let motion_inst = if self.prev_luma.len() == luma.len() {
            let diff = luma
                .iter()
                .zip(self.prev_luma.iter())
                .map(|(&a, &b)| (i32::from(a) - i32::from(b)).unsigned_abs() as f64)
                .sum::<f64>()
                / (n * 255.0);
            finite_clamp((diff * 4.0) as f32, 0.0, 1.0)
        } else {
            0.0
        };
        self.prev_luma = luma;

        // Channel traces: the first frame seeds outright (an afferent
        // reports its current rate — it does not ramp toward it), then
        // EMAs smooth across frames. Motion is a leaky max — rises on
        // the transient itself, decays between frames — and orienting
        // is the unsmoothed burst the colliculus actually fires.
        if self.frames_seen == 0 {
            self.illuminance = luma_mean as f32;
            self.melanopic = mel_mean as f32;
        } else {
            self.illuminance += ((luma_mean as f32) - self.illuminance) * 0.3;
            self.melanopic += ((mel_mean as f32) - self.melanopic) * 0.3;
        }
        self.motion = motion_inst.max(self.motion * 0.6);
        self.orienting = finite_clamp(motion_inst * 3.0, 0.0, 1.0);
    }

    /// Whether the eye is delivering frames right now.
    pub fn fresh(&self, now_ms: u64) -> bool {
        self.last_fresh_ms != 0 && now_ms.saturating_sub(self.last_fresh_ms) < FRESH_WINDOW_MS
    }

    /// How open the lids are — the phase's transmission factor,
    /// deepened by the mind's voluntary closure. Reads `gate_phase`,
    /// so it tracks the stage with at most one advance of lag; a lid
    /// that lags the flip-flop by a tick is physiology, not
    /// staleness.
    pub fn lid(&self) -> f32 {
        lid_transmission(self.gate_phase).min(self.voluntary_lid)
    }

    /// The mind's voluntary lid openness, clamped into [0, 1] —
    /// `SET_LID`. 1.0 releases the eye to the phase's own position;
    /// lower values hold the lid down on whatever the phase allows.
    /// It composes with `min`, so a request to open against a
    /// sleeping phase changes nothing — the stage owns the floor.
    /// Voluntary state is held, not momentary: lids stay where the
    /// mind left them until it moves them again, the way a held
    /// blink or a squint persists.
    pub fn set_lid(&mut self, openness: f32) {
        self.voluntary_lid = finite_clamp(openness, 0.0, 1.0);
    }

    /// The retinohypothalamic input for the sleep oscillator, in its
    /// lux units — `None` while the eye is dark or absent, which hands
    /// entrainment back to the built-in photoperiod. Lid-filtered:
    /// a bright room drives the clock weakly through closed lids,
    /// never at full strength.
    pub fn retinal_lux(&self, now_ms: u64) -> Option<f64> {
        if !self.fresh(now_ms) {
            return None;
        }
        Some(f64::from(self.melanopic * self.lid()).powf(MELANOPIC_LUX_GAMMA) * MELANOPIC_LUX_SCALE)
    }

    /// The collicular alerting pulse in LC-input units — applied only
    /// while the network is below its own wake threshold, so a salient
    /// visual transient is a wake-up call rather than a constant bias.
    /// `f_lc` is the wake population's firing rate; the caller reads it
    /// per sub-step so the pulse follows the state it is pushing on.
    pub fn alert_lc(&self, f_lc: f64) -> f64 {
        if f_lc >= crate::state::sleep_network::SLEEP_THRESHOLD {
            return 0.0;
        }
        // Lid-filtered like the RHT channel — a flash through closed
        // lids still reaches the colliculus, attenuated.
        f64::from(self.orienting) * 2.0 * f64::from(self.lid())
    }

    /// The optic-nerve afference for one integration step.
    pub fn afference(
        &self,
        now_ms: u64,
        f_lc: f64,
    ) -> crate::state::sleep_network::VisualAfference {
        crate::state::sleep_network::VisualAfference {
            light: self.retinal_lux(now_ms),
            alert_lc: self.alert_lc(f_lc),
        }
    }

    /// Recompute the thalamic relay gate for the body's current
    /// phase. Called each advance after the network's stage is known.
    pub fn update_gate(&mut self, phase: MentalPhase) {
        self.gate_phase = phase;
        self.relay_gain = finite_clamp(
            relay_base(phase) * (CORTICAL_FLOOR + CORTICAL_SPAN * self.cortical_gain),
            0.0,
            1.0,
        );
    }

    /// The mind's corticothalamic feedback, clamped into [0, 1].
    /// Modulatory only — see [`CORTICAL_FLOOR`]. The relay re-derives
    /// against the last known stage so the feedback lands now, not
    /// at the next advance.
    pub fn set_cortical_gain(&mut self, gain: f32) {
        self.cortical_gain = finite_clamp(gain, 0.0, 1.0);
        let phase = self.gate_phase;
        self.update_gate(phase);
    }

    /// Integrate the pupil one step on the advance clock.
    ///
    /// The set point is the sphincter/dilator balance: a log-compressive
    /// light reflex off the photopic channel (the pretectal arc) plus a
    /// sympathetic dilation from effective norepinephrine — LC tone is
    /// the same arousal signal that widens a biological pupil. The
    /// approach is first-order with an asymmetric tau: constriction is
    /// the fast parasympathetic arc, dilation the slow sympathetic mix.
    /// `ne` is the caller's current effective norepinephrine [0, 1].
    ///
    /// The elapsed interval is measured between calls, not assumed —
    /// a caller at any cadence gets the same trajectory, matching the
    /// body's no-`dt`-parameter rule. With no eye the illuminance
    /// channel sits at zero and the pupil rests dark-adapted;
    /// `eye_present` says whether the eye itself is there.
    pub fn update_pupil(&mut self, now_ms: u64, ne: f32) {
        if self.pupil_last_ms == 0 {
            self.pupil_last_ms = now_ms;
            return;
        }
        let dt = (now_ms.saturating_sub(self.pupil_last_ms) as f32 / 1000.0).clamp(0.0, 5.0);
        self.pupil_last_ms = now_ms;
        if dt <= 0.0 {
            return;
        }
        // Light reflex: diameter tracks log illuminance — darkness
        // pulls toward PUPIL_MAX, a saturated channel toward ~3.5 mm.
        // The pretectal input is lid-filtered like every afferent
        // channel: shut lids show the reflex a near-dark retina.
        let lid = self.lid();
        let light =
            PUPIL_MAX_MM - PUPIL_LIGHT_SPAN_MM * (1.0 + 9.0 * self.illuminance * lid).log10();
        // …plus central miosis while the lids are shut. Without it a
        // sleeping eye would read the lid's own darkness as "night"
        // and dilate — the opposite of a sleeping pupil.
        let sleep = PUPIL_SLEEP_CONSTRICT_MM * (1.0 - lid);
        let target = finite_clamp(
            light + PUPIL_NE_GAIN_MM * finite_clamp(ne, 0.0, 1.0) - sleep,
            PUPIL_MIN_MM,
            PUPIL_MAX_MM,
        );
        let tau = if target < self.pupil_mm {
            PUPIL_TAU_CONSTRICT_S
        } else {
            PUPIL_TAU_DILATE_S
        };
        self.pupil_mm += (target - self.pupil_mm) * (1.0 - (-dt / tau).exp());
    }

    /// The mind's focus request — a distance in metres, converted to
    /// the ciliary drive in dioptres (`1/m`). Non-positive or
    /// non-finite distances relax the eye to infinity (0 D); nearer
    /// than the ~17 cm near point clamps at `ACCOM_MAX_DPT`.
    /// Monocular: there is no vergence component — this is the ciliary
    /// muscle's own servo, driven by what the mind is attending.
    pub fn set_focus_m(&mut self, meters: f32) {
        let dpt = if meters.is_finite() && meters > 0.0 {
            1.0 / meters
        } else {
            0.0
        };
        self.accom_target_dpt = finite_clamp(dpt, ACCOM_MIN_DPT, ACCOM_MAX_DPT);
    }

    /// Integrate the lens power one step on the advance clock.
    ///
    /// First-order approach toward the requested power with the
    /// ciliary muscle's asymmetry: releasing back to distance is the
    /// faster direction, pulling in for near focus the slower one.
    /// The elapsed interval is measured between calls — the same
    /// trajectory at any cadence, like the pupil beside it.
    pub fn update_accommodation(&mut self, now_ms: u64) {
        if self.accom_last_ms == 0 {
            self.accom_last_ms = now_ms;
            return;
        }
        let dt = (now_ms.saturating_sub(self.accom_last_ms) as f32 / 1000.0).clamp(0.0, 5.0);
        self.accom_last_ms = now_ms;
        if dt <= 0.0 {
            return;
        }
        // With the lids shut the ciliary muscle relaxes to its tonic
        // rest — no fixation demand exists to sustain, and a request
        // from before sleep does not survive it. The next SET_FOCUS
        // on waking is a fresh demand.
        if 1.0 - self.lid() >= LID_CLOSED {
            self.accom_target_dpt = ACCOM_REST_DPT;
        }
        let tau = if self.accom_target_dpt > self.accommodation_dpt {
            ACCOM_TAU_NEAR_S
        } else {
            ACCOM_TAU_FAR_S
        };
        self.accommodation_dpt +=
            (self.accom_target_dpt - self.accommodation_dpt) * (1.0 - (-dt / tau).exp());
    }

    /// Current channel state for `GET_OPTIC_STATE`.
    pub fn state(&self, now_ms: u64) -> OpticState {
        OpticState {
            eye_present: self.frames_seen > 0,
            frames_fresh: self.fresh(now_ms),
            illuminance: self.illuminance,
            melanopic: self.melanopic,
            motion: self.motion,
            orienting: self.orienting,
            relay_gain: self.relay_gain,
            cortical_gain: self.cortical_gain,
            retinal_lux: self.retinal_lux(now_ms).unwrap_or(0.0) as f32,
            frame_seq: self.last_seq,
            frame_age_ms: if self.last_fresh_ms == 0 {
                -1
            } else {
                now_ms.saturating_sub(self.last_fresh_ms) as i64
            },
            pupil_mm: self.pupil_mm,
            accommodation_dpt: self.accommodation_dpt,
            lid: self.lid(),
        }
    }
}

impl Default for OpticNerve {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a private retinal segment carrying `payload` bytes after a
    /// valid header, then hand back the segment (kept alive by the
    /// caller so the mapping persists) and its name.
    fn fake_retina(
        tag: &str,
        width: u32,
        height: u32,
        fourcc: &[u8; 4],
        frame: &[u8],
    ) -> (shared_memory::Shmem, String) {
        let name = format!("/genesis_retina_test_{}_{}", std::process::id(), tag);
        let stride = width * 3;
        let size = (stride * height) as usize;
        let shmem = shared_memory::ShmemConf::new()
            .os_id(&name)
            .size(HEADER_SIZE + size.max(frame.len()))
            .create()
            .expect("create shm");
        let base = shmem.as_ptr();
        // SAFETY: base is a fresh mapping of HEADER_SIZE + size bytes;
        // all writes here are inside it.
        unsafe {
            std::ptr::write_volatile(base.add(8) as *mut u32, width);
            std::ptr::write_volatile(base.add(12) as *mut u32, height);
            std::ptr::copy_nonoverlapping(fourcc.as_ptr(), base.add(16), 4);
            std::ptr::write_volatile(base.add(20) as *mut u32, size.max(frame.len()) as u32);
            std::ptr::write_volatile(base.add(24) as *mut u32, stride);
            std::ptr::copy_nonoverlapping(frame.as_ptr(), base.add(HEADER_SIZE), frame.len());
            // Publish: even counter = a complete frame.
            (*(base as *const AtomicU64)).store(2, Ordering::Release);
        }
        (shmem, name.trim_start_matches('/').to_string())
    }

    #[test]
    fn absent_segment_is_no_fault_and_no_eye() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_absent_nope");
        optic.sample(1000);
        let s = optic.state(1000);
        assert!(!s.eye_present && !s.frames_fresh && s.frame_age_ms == -1);
        assert_eq!(optic.retinal_lux(1000), None);
    }

    #[test]
    fn rgb_frame_feeds_the_channels() {
        // 64x48 RGB3, mid-blue — melanopsin-weighted should read high.
        let frame = [30u8, 40u8, 200u8].repeat(64 * 48);
        let (seg, name) = fake_retina("rgb", 64, 48, b"RGB3", &frame);
        let mut optic = OpticNerve::with_name(&name);
        optic.sample(10_000);
        let s = optic.state(10_000);
        assert!(s.eye_present && s.frames_fresh);
        // Rec.601 luma of (30,40,200) ≈ (9+23.5+22.8)/255 ≈ 0.216.
        assert!((s.illuminance - 0.216).abs() < 0.05, "{}", s.illuminance);
        // Melanopic: (0.3*40 + 0.7*200)/255 ≈ 0.60 — the blue-weighted
        // channel reads far above photopic luma on a blue frame.
        assert!((s.melanopic - 0.60).abs() < 0.05, "{}", s.melanopic);
        assert_eq!(s.motion, 0.0, "first frame has no predecessor");
        let lux = optic
            .retinal_lux(10_000)
            .expect("fresh frames feed the SCN");
        assert!(lux > 500.0 && lux < 2500.0, "lux {lux}");
        // Freshness expires — the photoperiod takes back the clock.
        assert_eq!(optic.retinal_lux(10_000 + FRESH_WINDOW_MS + 1), None);
        drop(seg);
    }

    #[test]
    fn a_changed_frame_is_motion_and_orienting() {
        let (seg, name) = fake_retina("motion", 64, 48, b"RGB3", &[0u8].repeat(64 * 48 * 3));
        let mut optic = OpticNerve::with_name(&name);
        optic.sample(1_000);
        // Rewrite the payload bright and bump the seqlock.
        let base = seg.as_ptr();
        // SAFETY: same mapping as above; writes stay inside it.
        unsafe {
            (*(base as *const AtomicU64)).store(3, Ordering::Relaxed);
            std::ptr::write_bytes(base.add(HEADER_SIZE), 200u8, 64 * 48 * 3);
            std::sync::atomic::fence(Ordering::SeqCst);
            (*(base as *const AtomicU64)).store(4, Ordering::Release);
        }
        optic.sample(2_000);
        let s = optic.state(2_000);
        assert!(s.motion > 0.9, "light-switch is ~full motion: {}", s.motion);
        assert!(s.orienting > 0.9, "a big transient is an orienting burst");
    }

    #[test]
    fn gate_follows_the_sleep_stage_and_cortical_feedback() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_gate_nope");
        optic.set_cortical_gain(0.5);
        optic.update_gate(MentalPhase::Active);
        assert!(optic.state(0).relay_gain > 0.7);
        optic.update_gate(MentalPhase::NREM);
        assert!(optic.state(0).relay_gain < 0.05, "TRN blocks NREM");
        optic.update_gate(MentalPhase::REM);
        let rem = optic.state(0).relay_gain;
        assert!(rem < 0.15 && rem > 0.0, "REM is gated but not zero: {rem}");
        // Corticothalamic feedback modulates within bounds — full
        // engagement cannot open the NREM gate.
        optic.set_cortical_gain(1.0);
        optic.update_gate(MentalPhase::NREM);
        assert!(optic.state(0).relay_gain < 0.05);
        // …but it does move the waking relay.
        optic.update_gate(MentalPhase::Active);
        optic.set_cortical_gain(0.0);
        optic.update_gate(MentalPhase::Drowsy);
        let low = optic.state(0).relay_gain;
        optic.set_cortical_gain(1.0);
        optic.update_gate(MentalPhase::Drowsy);
        assert!(optic.state(0).relay_gain > low);
    }

    #[test]
    fn alert_only_fires_below_the_wake_threshold() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_alert_nope");
        optic.orienting = 0.5;
        assert_eq!(
            optic.alert_lc(crate::state::sleep_network::SLEEP_THRESHOLD + 1.0),
            0.0
        );
        assert_eq!(optic.alert_lc(1.0), 1.0, "asleep: pulse in LC units");
    }

    #[test]
    fn pupil_constricts_to_light_and_dilates_to_dark() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_pupil_nope");
        // First call only seeds the clock — the resting diameter holds.
        optic.update_pupil(1_000, 0.3);
        assert_eq!(optic.state(1_000).pupil_mm, PUPIL_REST_MM);
        // Five seconds in darkness (illuminance channel at zero):
        // the reflex approaches the dark-adapted end of the range.
        for t in 1..=10 {
            optic.update_pupil(1_000 + t * 500, 0.0);
        }
        let dark = optic.state(6_000).pupil_mm;
        assert!(dark > 6.5, "dark-adapted pupil opens: {dark}");
        // Five seconds at a saturated photopic channel: fast
        // constriction toward ~3.5 mm.
        optic.illuminance = 1.0;
        for t in 1..=10 {
            optic.update_pupil(6_000 + t * 500, 0.0);
        }
        let bright = optic.state(11_000).pupil_mm;
        assert!(bright < 4.0, "light-adapted pupil closes: {bright}");
    }

    #[test]
    fn pupil_dilates_with_norepinephrine() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_pupil_ne_nope");
        optic.illuminance = 0.5;
        optic.update_pupil(1_000, 0.0);
        for t in 1..=20 {
            optic.update_pupil(1_000 + t * 250, 0.0);
        }
        let baseline = optic.state(6_000).pupil_mm;
        // The same light with LC tone saturated: the dilator pulls
        // against the sphincter's set point.
        for t in 1..=20 {
            optic.update_pupil(6_000 + t * 250, 1.0);
        }
        let aroused = optic.state(11_000).pupil_mm;
        assert!(
            aroused > baseline + 0.5,
            "NE widens the same scene's pupil: {baseline} -> {aroused}"
        );
    }

    #[test]
    fn pupil_constriction_outruns_dilation() {
        // Both nerves start at rest; one steps toward a bright target,
        // the other toward darkness. Same elapsed step: the
        // parasympathetic arc covers more of its travel.
        let mut constricting = OpticNerve::with_name("/genesis_retina_test_pc_nope");
        let mut dilating = OpticNerve::with_name("/genesis_retina_test_pd_nope");
        constricting.illuminance = 1.0;
        dilating.illuminance = 0.0;
        constricting.update_pupil(1_000, 0.0);
        dilating.update_pupil(1_000, 0.0);
        constricting.update_pupil(1_250, 0.0);
        dilating.update_pupil(1_250, 0.0);
        let target_c = PUPIL_MAX_MM - PUPIL_LIGHT_SPAN_MM * (1.0f32 + 9.0).log10();
        let target_d = PUPIL_MAX_MM;
        let frac_c =
            (PUPIL_REST_MM - constricting.state(1_250).pupil_mm) / (PUPIL_REST_MM - target_c);
        let frac_d = (dilating.state(1_250).pupil_mm - PUPIL_REST_MM) / (target_d - PUPIL_REST_MM);
        assert!(frac_c > 0.0 && frac_d > 0.0);
        assert!(
            frac_c > frac_d,
            "constriction covers more of its travel: {frac_c} vs {frac_d}"
        );
    }

    #[test]
    fn wire_is_fifty_eight_bytes_and_carries_the_ocular_state() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_wire_nope");
        optic.pupil_mm = 5.25;
        optic.accommodation_dpt = 2.5;
        optic.update_gate(MentalPhase::Drowsy);
        let wire = optic.state(0).to_wire();
        assert_eq!(wire.len(), 58);
        let pupil = f32::from_le_bytes(wire[46..50].try_into().unwrap());
        assert!((pupil - 5.25).abs() < 1e-6, "pupil rides the wire: {pupil}");
        let dpt = f32::from_le_bytes(wire[50..54].try_into().unwrap());
        assert!((dpt - 2.5).abs() < 1e-6, "lens power rides the wire: {dpt}");
        let lid = f32::from_le_bytes(wire[54..58].try_into().unwrap());
        assert!((lid - 0.5).abs() < 1e-6, "the lid rides the wire: {lid}");
    }

    #[test]
    fn focus_distance_maps_to_dioptres() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_accom_nope");
        optic.set_focus_m(1.0);
        assert_eq!(optic.accom_target_dpt, 1.0);
        optic.set_focus_m(0.5);
        assert_eq!(optic.accom_target_dpt, 2.0);
        // Nearer than the ~17 cm near point clamps at the amplitude.
        optic.set_focus_m(0.05);
        assert_eq!(optic.accom_target_dpt, ACCOM_MAX_DPT);
        // Non-positive and non-finite distances relax to infinity.
        optic.set_focus_m(0.0);
        assert_eq!(optic.accom_target_dpt, 0.0);
        optic.set_focus_m(f32::NAN);
        assert_eq!(optic.accom_target_dpt, 0.0);
        optic.set_focus_m(f32::INFINITY);
        assert_eq!(optic.accom_target_dpt, 0.0);
    }

    #[test]
    fn accommodation_slews_toward_the_requested_power() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_slew_nope");
        // First call seeds the clock at the tonic rest position.
        optic.update_accommodation(1_000);
        assert_eq!(optic.state(1_000).accommodation_dpt, ACCOM_REST_DPT);
        // Request a 25 cm near point (4 D) and hold for three
        // seconds — the ciliary muscle covers most of the travel.
        optic.set_focus_m(0.25);
        for t in 1..=12 {
            optic.update_accommodation(1_000 + t * 250);
        }
        let dpt = optic.state(4_000).accommodation_dpt;
        assert!(dpt > 3.0, "near focus pulled in: {dpt}");
        // Release to infinity — the faster direction, so five seconds
        // brings the lens comfortably back near rest.
        optic.set_focus_m(f32::INFINITY);
        for t in 1..=20 {
            optic.update_accommodation(4_000 + t * 250);
        }
        let dpt = optic.state(9_000).accommodation_dpt;
        assert!(dpt < 1.5, "released back toward distance: {dpt}");
    }

    #[test]
    fn accommodation_releases_faster_than_it_pulls_in() {
        let mut near = OpticNerve::with_name("/genesis_retina_test_an_nope");
        let mut far = OpticNerve::with_name("/genesis_retina_test_af_nope");
        // Equal distances from target: near starts at rest (1 D) and
        // pulls toward 3 D; far starts at 5 D and releases toward 3 D.
        near.accom_target_dpt = 3.0;
        far.accommodation_dpt = 5.0;
        far.accom_target_dpt = 3.0;
        near.update_accommodation(1_000);
        far.update_accommodation(1_000);
        near.update_accommodation(1_250);
        far.update_accommodation(1_250);
        let frac_near = (near.state(1_250).accommodation_dpt - ACCOM_REST_DPT) / 2.0;
        let frac_far = (5.0 - far.state(1_250).accommodation_dpt) / 2.0;
        assert!(
            frac_far > frac_near,
            "release outruns pull-in: {frac_far} vs {frac_near}"
        );
    }

    #[test]
    fn closed_lids_dim_the_oscillator_but_do_not_black_it_out() {
        // A bright blue frame — the melanopic channel reads high.
        let frame = [30u8, 40u8, 200u8].repeat(64 * 48);
        let (seg, name) = fake_retina("lids", 64, 48, b"RGB3", &frame);
        let mut optic = OpticNerve::with_name(&name);
        optic.sample(10_000);
        optic.update_gate(MentalPhase::Active);
        let awake = optic.retinal_lux(10_000).expect("fresh frame");
        // Same bright room through NREM lids: the SCN drive collapses
        // to a few percent — entrained weakly, not abolished.
        optic.update_gate(MentalPhase::NREM);
        let asleep = optic.retinal_lux(10_000).expect("still a reading");
        assert!(asleep > 0.0, "lids leak light, they do not black it out");
        assert!(
            asleep < awake * 0.1,
            "closed lids attenuate the RHT drive: {asleep} vs {awake}"
        );
        // Half-mast sits between.
        optic.update_gate(MentalPhase::Drowsy);
        let drowsy = optic.retinal_lux(10_000).expect("still a reading");
        assert!(drowsy > asleep && drowsy < awake);
        drop(seg);
    }

    #[test]
    fn a_flash_through_closed_lids_still_reaches_the_colliculus() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_lidalert_nope");
        optic.orienting = 0.5;
        optic.update_gate(MentalPhase::Active);
        let awake = optic.alert_lc(1.0);
        optic.update_gate(MentalPhase::NREM);
        let asleep = optic.alert_lc(1.0);
        assert!(asleep > 0.0, "a transient leaks through shut lids");
        assert!(asleep < awake * 0.1, "but it is a whisper, not a shout");
    }

    #[test]
    fn sleep_pulls_the_pupil_to_miosis_not_dilation() {
        // Dark room, no NE — an awake pupil dark-adapts toward 8 mm;
        // the sleeping pupil constricts despite the same darkness.
        let mut optic = OpticNerve::with_name("/genesis_retina_test_miosis_nope");
        optic.illuminance = 0.0;
        optic.update_gate(MentalPhase::Active);
        optic.update_pupil(1_000, 0.0);
        for t in 1..=20 {
            optic.update_pupil(1_000 + t * 250, 0.0);
        }
        let awake_dark = optic.state(6_000).pupil_mm;
        assert!(awake_dark > 6.5, "awake in the dark dilates: {awake_dark}");
        // The flip-flop lands in NREM — the same room, lids shut.
        optic.update_gate(MentalPhase::NREM);
        for t in 1..=20 {
            optic.update_pupil(6_000 + t * 250, 0.0);
        }
        let asleep = optic.state(11_000).pupil_mm;
        assert!(
            (2.5..4.5).contains(&asleep),
            "the sleeping pupil is miotic, not dark-adapted: {asleep}"
        );
    }

    #[test]
    fn accommodation_relaxes_to_rest_in_sleep() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_accomsleep_nope");
        optic.update_gate(MentalPhase::Active);
        optic.update_accommodation(1_000);
        // Awake: a 25 cm near point pulls the lens to ~4 D.
        optic.set_focus_m(0.25);
        for t in 1..=12 {
            optic.update_accommodation(1_000 + t * 250);
        }
        assert!(optic.state(4_000).accommodation_dpt > 3.0);
        // Sleep releases the request itself — the ciliary rests.
        optic.update_gate(MentalPhase::NREM);
        for t in 1..=20 {
            optic.update_accommodation(4_000 + t * 250);
        }
        assert_eq!(optic.accom_target_dpt, ACCOM_REST_DPT);
        let dpt = optic.state(9_000).accommodation_dpt;
        assert!(
            (dpt - ACCOM_REST_DPT).abs() < 0.3,
            "the lens relaxes to tonic rest: {dpt}"
        );
    }

    #[test]
    fn the_will_shuts_but_never_opens_the_lid() {
        let mut optic = OpticNerve::with_name("/genesis_retina_test_willlid_nope");
        optic.update_gate(MentalPhase::Active);
        // Awake, unrequested: the phase's lid is open.
        assert!((optic.lid() - 1.0).abs() < 1e-6);
        // A voluntary close dims an open phase to the requested
        // level — held, not momentary.
        optic.set_lid(0.05);
        assert!((optic.lid() - 0.05).abs() < 1e-6);
        // Half-mast against a drowsy phase: min picks the deeper
        // closure — will can deepen, never lighten.
        optic.update_gate(MentalPhase::Drowsy);
        assert!((optic.lid() - 0.05).abs() < 1e-6);
        // Released, the lid rides the phase again — half-mast drowsy.
        optic.set_lid(1.0);
        assert!((optic.lid() - 0.5).abs() < 1e-6);
        // And a request to open against a sleeping phase changes
        // nothing — the stage owns the floor.
        optic.update_gate(MentalPhase::NREM);
        optic.set_lid(1.0);
        assert!((optic.lid() - 0.03).abs() < 1e-6);
        // …though a shut eye shut even harder is still its request.
        optic.set_lid(0.0);
        assert_eq!(optic.lid(), 0.0);
    }

    #[test]
    fn a_voluntary_close_dims_the_oscillator_too() {
        // The voluntary lid rides the same afference as the phase
        // lid — a bright room through held-shut lids feeds the SCN
        // weakly, like sleep.
        let frame = [30u8, 40u8, 200u8].repeat(64 * 48);
        let (seg, name) = fake_retina("willlux", 64, 48, b"RGB3", &frame);
        let mut optic = OpticNerve::with_name(&name);
        optic.sample(10_000);
        optic.update_gate(MentalPhase::Active);
        let open = optic.retinal_lux(10_000).expect("fresh frame");
        optic.set_lid(0.05);
        let shut = optic.retinal_lux(10_000).expect("still a reading");
        assert!(shut > 0.0, "held lids leak light like sleeping lids");
        assert!(
            shut < open * 0.1,
            "a voluntary close attenuates the RHT drive: {shut} vs {open}"
        );
        drop(seg);
    }
}
