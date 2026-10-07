//! Legacy schema v3 layout — the on-disk binary format of state files
//! written by Genesis binaries before schema v4.
//!
//! This module exists solely for **migration**: `MmapState::open`
//! decodes a v3 file through this `#[repr(C)]` type, then converts it
//! into the current [`GenesisCoreState`]. Schema v4 inserted the
//! persistent [`IonState`] before the checksum, so a v3 file cannot be
//! reinterpreted in place.
//!
//! DO NOT modify these types: they are a byte-exact description of a
//! historical format. The compile-time assertion below pins the size.

use super::core_state::GenesisCoreState;
use super::header::CoreStateHeader;
use super::inference::InferenceSignals;
use super::ions::IonState;
use super::manifest::RuntimeManifest;
use super::memory::MemoryPointers;
use super::neurochemical::NeurochemicalVector;
use super::zones::ActiveZones;

/// The v3 `GenesisCoreState` (3296 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GenesisCoreStateV3 {
    /// Identity, versioning, and seqlock.
    pub header: CoreStateHeader,
    /// 18-chemical coupled neurochemical dynamics.
    pub neurochemicals: NeurochemicalVector,
    /// Active task zone, emergent phase, and subcognitive flags.
    pub zones: ActiveZones,
    /// Memory store pointers and emotional gating weights.
    pub memory: MemoryPointers,
    /// Runtime module manifest table.
    pub manifest: RuntimeManifest,
    /// CRC32 checksum of all preceding bytes (header.seq_lock excluded).
    pub checksum: u32,
    /// Active inference signals — the generative self-model's
    /// projection. Lives in the former reserved region (64 bytes).
    /// Not covered by the CRC32 checksum (derived state, recomputed
    /// every tick). See [`InferenceSignals`].
    pub inference_signals: InferenceSignals,
}

impl GenesisCoreStateV3 {
    /// Convert a decoded v3 snapshot into the current schema.
    ///
    /// Every field except `ions` carries across unchanged: v4 only
    /// inserted the ion state ahead of the checksum, so no earlier
    /// field moved. `inference_signals` is derived state that the
    /// v3 layout already carried, so it is preserved rather than reset.
    pub fn into_current(self) -> GenesisCoreState {
        GenesisCoreState {
            header: self.header,
            neurochemicals: self.neurochemicals,
            zones: self.zones,
            memory: self.memory,
            manifest: self.manifest,
            ions: IonState::new(),
            checksum: self.checksum,
            inference_signals: self.inference_signals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_layout_is_pinned() {
        assert_eq!(core::mem::size_of::<GenesisCoreStateV3>(), 3296);
    }
}
