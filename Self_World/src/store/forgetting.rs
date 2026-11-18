//! Gentle forgetting — the power-law decay of memory accessibility.
//!
//! Forgetting is the mirror of encoding: the store's `store()` binds a
//! trace, and this module decides how that trace's accessibility decays
//! over time. The model follows the empirical biology rather than an
//! arbitrary eviction policy:
//!
//! 1. **Power-law trace decay** (Anderson & Schooler 1991; Wixted &
//!    Ebbesen 1991): each presentation of a memory leaves a trace whose
//!    strength decays as t^(−d), d ≈ 0.5. Repeated presentation leaves
//!    multiple traces; total trace mass is the sum of the surviving
//!    strengths, M_i = Σ_k (age_k)^(−d).
//!
//! 2. **Base-level activation** (ACT-R): B_i = ln M_i + w·salience.
//!    Retrieval succeeds when activation exceeds a threshold, with
//!    activation noise s·χ making the boundary stochastic — matching
//!    the probabilistic retrieval observed in biological memory.
//!
//! 3. **Dopaminergic forgetting** (Davis & Zhong 2017): dopamine
//!    "forgetting cells" accelerate trace decay while a memory is not
//!    being retrieved. High dopamine lowers the effective retrieval
//!    threshold τ_eff, raising apparent accessibility for recently
//!    encoded items and deepening the forgetting of untouched ones —
//!    modeled as a transient shift τ_eff = τ₀ − g·(DA − DA_baseline).
//!
//! 4. **Nothing is deleted while it is still reachable.** The prune
//!    predicate `should_forget` fires only when a memory's activation
//!    is below the point where even a maximal-similarity cue could
//!    reach it — B_i < τ₀ − w_cue — i.e. the engram is functionally
//!    dismantled and only its storage remains. Rehearsal (every call
//!    to `retrieve`) appends a fresh trace, so a memory that is still
//!    being used cannot drift to the forget boundary.
//!
//! Rehearsal is the declared opposite of forgetting: `touch()` is the
//! `+` to decay's `−`.

/// Power-law decay exponent. The classic value (Anderson & Schooler
/// 1991; Wixted 2004) is d ≈ 0.5.
pub const TRACE_DECAY: f64 = 0.5;

/// Milliseconds per day — decay is measured in days so TRACE_DECAY's
/// empirical value stays meaningful.
pub const DAY_MS: f64 = 86_400_000.0;

/// Weight of episode salience in base-level activation. A
/// maximally-salient first-hearing memory gets +1.0 on top of its
/// trace term, shifting its forget horizon outward by e^2 ≈ 7.4×.
pub const SALIENCE_GAIN: f64 = 1.0;

/// How strongly retrieval-cue similarity contributes to accessibility.
/// A maximal cue (identical hash) contributes +w_cue.
pub const CUE_WEIGHT: f64 = 2.0;

/// Base retrieval threshold — accessibility ≥ τ is retrievable.
pub const RETRIEVAL_TAU: f64 = 0.0;

/// Activation-noise scale — the logistic s parameter. The ±1σ band
/// around the threshold is ≈ 1.54·s ≈ ±0.39 logits.
pub const NOISE_SCALE: f64 = 0.25;

/// Dopamine baseline (resting). The forgetting gate uses deviation
/// from this, not the absolute level.
pub const DA_BASELINE: f64 = 0.5;

/// How strongly above-baseline dopamine eases retrieval (and, below
/// baseline, deepens it). Kept small so chemistry modulates the
/// boundary rather than dominating it.
pub const DA_FORGETTING_GAIN: f64 = 0.4;

/// An episode must be at least this old to be a prune candidate —
/// protection against clock anomalies and freshly encoded items.
pub const PRUNE_AGE_FLOOR_DAYS: f64 = 30.0;

/// The activation below which a memory is unreachable even by a
/// perfect retrieval cue: τ₀ − w_cue. Below this the engram is
/// functionally gone and pruning only reclaims its residue.
pub const FORGET_BOUND: f64 = RETRIEVAL_TAU - CUE_WEIGHT;

/// One retrieval/rehearsal event for a single episode: how many times
/// it has been activated and when it was last touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trace {
    /// Number of encodings + retrievals (≥ 1 while the episode exists).
    pub access_count: u32,
    /// Wall-clock millisecond timestamp of the most recent access.
    pub last_access_ms: u64,
}

impl Trace {
    /// A fresh trace for an episode encoded at `timestamp_ms`.
    pub fn encoded(timestamp_ms: u64) -> Self {
        Self {
            access_count: 1,
            last_access_ms: timestamp_ms,
        }
    }
}

/// Total surviving trace mass for an episode — the sum of each
/// presentation's power-law-decayed strength.
///
/// For n presentations over the episode's lifetime, exact mass is
/// Σ_k a_k^(−d), but only the count and the two endpoint ages are
/// persisted. Two estimators bound the unknown interior:
///
/// - **Blend** (Anderson & Schooler 1991's merged-trace
///   approximation): presentations treated as uniformly spaced, so
///   M ≈ 2n / (√a₀ + √a_L). Best when reactivations are spread
///   evenly through the episode's life.
/// - **Endpoint bound**: the newest trace contributes its exact
///   a_L^(−d); every earlier trace is at least as old as a₀ and
///   therefore contributes at least a₀^(−d), giving
///   M ≥ a_L^(−d) + (n−1)·a₀^(−d). Exact for n ≤ 2, and the honest
///   floor for a memory rehearsed once after a long silence — where
///   the blend would spread the fresh trace's strength back across
///   the whole span and under-count it.
///
/// The mass is the larger of the two — the estimator that least
/// underestimates for the history the episode actually lived.
///
/// `age_first_days`/`age_last_days` are ages of the oldest and newest
/// trace in days; ages are floored so a just-encoded trace never
/// produces an infinite mass.
pub fn trace_mass(access_count: u32, age_first_days: f64, age_last_days: f64) -> f64 {
    if access_count == 0 {
        return 0.0;
    }
    let a_first = age_first_days.max(MIN_AGE_DAYS);
    let a_last = age_last_days.max(MIN_AGE_DAYS);
    if access_count == 1 {
        return a_last.powf(-TRACE_DECAY);
    }
    let n = f64::from(access_count);
    let blend = 2.0 * n / (a_first.sqrt() + a_last.sqrt());
    let bound = a_last.powf(-TRACE_DECAY) + (n - 1.0) * a_first.powf(-TRACE_DECAY);
    blend.max(bound)
}

/// Minimum age — the decay integral diverges at age 0, and real
/// presentations are separated by at least the encoding quantum.
const MIN_AGE_DAYS: f64 = 1.0 / 1440.0; // one minute

/// Base-level activation: ln M_i + w·salience.
///
/// `encoded_ms` is the episode's original encoding timestamp (the
/// oldest trace's age comes from it); `trace` carries access count
/// and recency. `salience` is the 0..1 salience stored at encoding.
pub fn base_level(trace: Trace, encoded_ms: u64, salience: f32, now_ms: u64) -> f64 {
    let age_first = (now_ms.saturating_sub(encoded_ms) as f64) / DAY_MS;
    let age_last = (now_ms.saturating_sub(trace.last_access_ms) as f64) / DAY_MS;
    trace_mass(trace.access_count, age_first, age_last).ln()
        + SALIENCE_GAIN * f64::from(salience.max(0.0))
}

/// Deterministic activation noise χ ∈ (−1, 1), hashed from the
/// episode and the query so the retrieval boundary is stochastic but
/// reproducible for the same probe — matching how a given cue at a
/// given moment either surfaces a memory or doesn't.
pub fn activation_noise(episode_id: u64, query_hash: u64) -> f64 {
    let mut z = episode_id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(query_hash);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Map the 53-bit mantissa into [0,1), then to (−1,1).
    let u = ((z >> 11) as f64) / 9007199254740992.0; // 2^53
    2.0 * u - 1.0
}

/// The retrieval threshold after the dopaminergic forgetting-cell
/// modulation. Above-baseline dopamine lowers τ (easier retrieval of
/// anything still reachable); below-baseline raises it.
pub fn effective_threshold(dopamine: f64) -> f64 {
    RETRIEVAL_TAU - DA_FORGETTING_GAIN * (dopamine - DA_BASELINE)
}

/// Full accessibility score for a candidate: B_i + w·sim + s·χ − τ_eff.
/// Positive means the cue reaches the memory.
pub fn accessibility_score(
    base: f64,
    cue_similarity: f64,
    episode_id: u64,
    query_hash: u64,
    dopamine: f64,
) -> f64 {
    base + CUE_WEIGHT * cue_similarity.clamp(0.0, 1.0)
        + NOISE_SCALE * activation_noise(episode_id, query_hash)
        - effective_threshold(dopamine)
}

/// Whether a candidate survives the forgetting gate for this probe.
pub fn is_accessible(
    base: f64,
    cue_similarity: f64,
    episode_id: u64,
    query_hash: u64,
    dopamine: f64,
) -> bool {
    accessibility_score(base, cue_similarity, episode_id, query_hash, dopamine) >= 0.0
}

/// The deep-forget predicate: has this memory decayed past the point
/// where any cue could reach it?
///
/// Fires only when B_i < τ₀ − w_cue, so even a maximal-similarity
/// probe with no noise assistance cannot retrieve it, AND the episode
/// is older than the age floor. Deliberately conservative — the bar
/// sits a full CUE_WEIGHT below the retrieval threshold, so memories
/// that are merely hard to reach stay stored; only the functionally
/// dismantled are released.
pub fn should_forget(base: f64, age_first_days: f64) -> bool {
    base < FORGET_BOUND && age_first_days >= PRUNE_AGE_FLOOR_DAYS
}

/// Predicted days until a memory with this activation reaches the
/// forget boundary — positive while it is still retrievable, and the
/// number a mind would use to decide "rehearse it now or let it go".
///
/// B_i = ln(2n/(√a₀ + √a_L)) + w·s → solves for the a₀ at which
/// B_i hits FORGET_BOUND, holding n, a_L ≈ a₀ (the uniform blend):
///   a_forget ≈ (2n / e^{FORGET_BOUND − w·s})² / 4
/// Returns days remaining until that age; ∞ if it can never fade
/// below the bound with the current trace count.
pub fn days_until_forgotten(access_count: u32, salience: f32, current_age_days: f64) -> f64 {
    // With a₀ ≈ a_L = a, trace_mass = 2n/(2√a) = n·a^(−1/2); B =
    // ln n − (d)·ln a + w·s  →  a_forget = (n / e^{B̄})^(1/d) where
    // B̄ = FORGET_BOUND − w·s.
    let barrier = FORGET_BOUND - SALIENCE_GAIN * f64::from(salience.max(0.0));
    let n = f64::from(access_count.max(1));
    let a_forget = (n / barrier.exp()).powf(1.0 / TRACE_DECAY);
    a_forget - current_age_days
}

// ─────────────────────────────────────────────────────────────────
//  Trace persistence — the .traces sidecar
// ─────────────────────────────────────────────────────────────────
//
// The `.meta` IndexEntry format is frozen at 64 bytes — the Python
// index scanner unpacks it directly — so rehearsal state lives in a
// fourth file beside the store: `{base}.traces`.
//
// ```text
// ┌─────────────────────────────────────────────────────┐
// │  Traces file (.traces) — random-access              │
// │  16-byte header: magic "LTMT" + version + count     │
// │  24-byte records: id u64 | n u32 | pad u32 | t u64  │
// └─────────────────────────────────────────────────────┘
// ```
//
// Writes are write-through but not fsync'd per touch — a lost
// rehearsal record only weakens a trace slightly (biologically the
// same thing happens: not every reactivation consolidates). The
// store's `sync()` fsyncs the file along with everything else.
// A missing or corrupt traces file opens empty and heals: every
// episode without a record reads as exactly one presentation at its
// encoding timestamp, which is its true floor anyway.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic bytes for the traces file: ASCII `"LTMT"`.
pub const TRACES_MAGIC: [u8; 4] = *b"LTMT";
/// Traces file schema version.
pub const TRACES_VERSION: u32 = 1;
/// Traces header size in bytes.
pub const TRACES_HEADER_SIZE: u64 = 16;
/// One trace record: episode_id u64 | access_count u32 | _pad u32 |
/// last_access_ms u64.
pub const TRACE_RECORD_SIZE: u64 = 24;

/// Rehearsal/access state for every episode — persisted beside the
/// LTM store so retrieval strengthens memory without touching the
/// frozen `.meta` row format.
pub struct TraceStore {
    file: std::fs::File,
    /// episode_id → record slot (byte offset = HEADER + slot·RECORD)
    slots: HashMap<u64, u32>,
    /// episode_id → trace. Entries not present read as the default
    /// encoded-once trace — see [`TraceStore::get`].
    traces: HashMap<u64, Trace>,
    next_slot: u32,
    path: PathBuf,
}

impl TraceStore {
    /// Open the traces file for `base`, creating it if absent. A
    /// corrupt or partial tail record is ignored rather than fatal —
    /// the worst case is a trace that looks one rehearsal weaker.
    pub fn open_or_create(base_path: &Path) -> std::io::Result<Self> {
        let path = base_path.with_extension("traces");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let len = file.metadata()?.len();
        let mut slots = HashMap::new();
        let mut traces = HashMap::new();
        let mut next_slot = 0u32;

        if len >= TRACES_HEADER_SIZE {
            let mut header = [0u8; TRACES_HEADER_SIZE as usize];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut header)?;
            if header[0..4] == TRACES_MAGIC
                && u32::from_le_bytes(header[4..8].try_into().unwrap_or_default()) == TRACES_VERSION
            {
                let declared = u32::from_le_bytes(header[8..12].try_into().unwrap_or_default());
                let present = ((len - TRACES_HEADER_SIZE) / TRACE_RECORD_SIZE) as u32;
                let readable = declared.min(present);
                let mut buf = vec![0u8; (readable * TRACE_RECORD_SIZE as u32) as usize];
                file.read_exact(&mut buf)?;
                for slot in 0..readable {
                    let off = (slot * TRACE_RECORD_SIZE as u32) as usize;
                    let episode_id =
                        u64::from_le_bytes(buf[off..off + 8].try_into().unwrap_or_default());
                    let access_count =
                        u32::from_le_bytes(buf[off + 8..off + 12].try_into().unwrap_or_default());
                    let last_access_ms =
                        u64::from_le_bytes(buf[off + 16..off + 24].try_into().unwrap_or_default());
                    if episode_id == 0 || access_count == 0 {
                        continue;
                    }
                    slots.insert(episode_id, slot);
                    traces.insert(
                        episode_id,
                        Trace {
                            access_count,
                            last_access_ms,
                        },
                    );
                }
                next_slot = present;
            }
        }

        if len < TRACES_HEADER_SIZE {
            // Fresh file — write the header now so the file is never
            // zero-length garbage on disk.
            Self::write_header(&mut file, 0)?;
        }

        Ok(Self {
            file,
            slots,
            traces,
            next_slot,
            path,
        })
    }

    fn write_header(file: &mut std::fs::File, count: u32) -> std::io::Result<()> {
        let mut header = [0u8; TRACES_HEADER_SIZE as usize];
        header[0..4].copy_from_slice(&TRACES_MAGIC);
        header[4..8].copy_from_slice(&TRACES_VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&count.to_le_bytes());
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        Ok(())
    }

    fn write_record(
        file: &mut std::fs::File,
        slot: u32,
        episode_id: u64,
        trace: Trace,
    ) -> std::io::Result<()> {
        let mut rec = [0u8; TRACE_RECORD_SIZE as usize];
        rec[0..8].copy_from_slice(&episode_id.to_le_bytes());
        rec[8..12].copy_from_slice(&trace.access_count.to_le_bytes());
        rec[16..24].copy_from_slice(&trace.last_access_ms.to_le_bytes());
        file.seek(SeekFrom::Start(
            TRACES_HEADER_SIZE + u64::from(slot) * TRACE_RECORD_SIZE,
        ))?;
        file.write_all(&rec)?;
        Ok(())
    }

    /// The trace for an episode; absent records default to a single
    /// presentation at `encoded_ms` — every stored episode has at
    /// least its encoding trace.
    pub fn get(&self, episode_id: u64, encoded_ms: u64) -> Trace {
        self.traces
            .get(&episode_id)
            .copied()
            .unwrap_or_else(|| Trace::encoded(encoded_ms))
    }

    /// Record an encoding: the first trace, stamped at the episode's
    /// own timestamp.
    pub fn record_encoding(&mut self, episode_id: u64, timestamp_ms: u64) -> std::io::Result<()> {
        self.put(episode_id, Trace::encoded(timestamp_ms))
    }

    /// Rehearsal — the opposite of forgetting. Each retrieval adds a
    /// presentation to the trace: n += 1, t_last = now. The added
    /// mass is strongest exactly when it lands, so a just-rehearsed
    /// memory jumps well clear of the retrieval boundary.
    pub fn touch(&mut self, episode_id: u64, now_ms: u64) -> std::io::Result<()> {
        let current = self
            .traces
            .get(&episode_id)
            .copied()
            .unwrap_or_else(|| Trace::encoded(now_ms));
        self.put(
            episode_id,
            Trace {
                access_count: current.access_count.saturating_add(1),
                last_access_ms: now_ms,
            },
        )
    }

    fn put(&mut self, episode_id: u64, trace: Trace) -> std::io::Result<()> {
        let slot = match self.slots.get(&episode_id) {
            Some(&slot) => slot,
            None => {
                let slot = self.next_slot;
                self.next_slot = self.next_slot.saturating_add(1);
                self.slots.insert(episode_id, slot);
                Self::write_header(&mut self.file, self.next_slot)?;
                slot
            }
        };
        Self::write_record(&mut self.file, slot, episode_id, trace)?;
        self.traces.insert(episode_id, trace);
        Ok(())
    }

    /// Forget an episode's trace — the record on disk is left as dead
    /// space until `retain` rewrites the file, mirroring how a deleted
    /// episode's `.meta` row lingers until compaction.
    pub fn remove(&mut self, episode_id: u64) {
        self.traces.remove(&episode_id);
        self.slots.remove(&episode_id);
    }

    /// Drop every trace whose episode no longer exists — called
    /// during compaction so the sidecar shrinks with the store it
    /// shadows.
    pub fn retain(&mut self, live: &std::collections::HashSet<u64>) -> std::io::Result<()> {
        self.traces.retain(|id, _| live.contains(id));
        // Rewrite the file compactly: the slot space is dense again.
        let mut entries: Vec<(u64, Trace)> = self.traces.iter().map(|(&id, &t)| (id, t)).collect();
        entries.sort_unstable_by_key(|(id, _)| *id);
        self.file.set_len(0)?;
        Self::write_header(&mut self.file, entries.len() as u32)?;
        self.slots.clear();
        for (slot, (id, trace)) in entries.into_iter().enumerate() {
            let slot = slot as u32;
            Self::write_record(&mut self.file, slot, id, trace)?;
            self.slots.insert(id, slot);
        }
        self.next_slot = self.slots.len() as u32;
        self.file.sync_all()
    }

    /// Flush the traces file. Called from `LtmStore::sync()`.
    pub fn sync(&mut self) -> std::io::Result<()> {
        self.file.sync_all()
    }

    /// Filesystem path of the backing file (for compaction staging).
    pub fn path(&self) -> &Path {
        &self.path
    }
}
