//! Legacy schema v4 layout — the on-disk binary format of state files
//! written by Genesis binaries before schema v5.
//!
//! This module exists solely for **migration**: `MmapState::open`
//! decodes a v4 file through this `#[repr(C)]` type, then converts it
//! into the current [`GenesisCoreState`]. Schema v5 inserted the
//! sleep-network state before the checksum, so a v4 file cannot be
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
use super::sleep_network;
use super::zones::ActiveZones;

/// The v4 `GenesisCoreState` (3416 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GenesisCoreStateV4 {
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
    /// Ion electrochemistry.
    pub ions: IonState,
    /// CRC32 checksum of all preceding bytes (header.seq_lock excluded).
    pub checksum: u32,
    /// Active inference signals (not covered by the checksum).
    pub inference_signals: InferenceSignals,
}

impl GenesisCoreStateV4 {
    /// Widen a v4 state to the current layout. The sleep network did
    /// not exist in v4, so it starts at the source's initial conditions.
    /// Migration recomputes the checksum afterwards.
    pub fn into_current(self) -> GenesisCoreState {
        GenesisCoreState {
            header: self.header,
            neurochemicals: self.neurochemicals,
            zones: self.zones,
            memory: self.memory,
            manifest: self.manifest,
            ions: self.ions,
            sleep_network: sleep_network::INITIAL,
            checksum: self.checksum,
            inference_signals: self.inference_signals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_layout_is_pinned() {
        assert_eq!(core::mem::size_of::<GenesisCoreStateV4>(), 3416);
    }
}
