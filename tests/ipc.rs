//! Tests for the IPC layer.
//!
//! These tests start the IPC server in a background thread and
//! connect to it with the IPC client, exercising the full
//! request/response cycle.

use genesis::daemon::*;
use genesis::state::*;
use genesis::store::*;
use std::path::PathBuf;
use std::time::Duration;

fn temp_path(prefix: &str, ext: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "genesis_ipc_{prefix}_{}_{}.{ext}",
        std::process::id(),
        nanos
    ))
}

fn temp_base(prefix: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "genesis_ipc_{prefix}_{}_{}",
        std::process::id(),
        nanos
    ))
}

struct IpcTestSystem {
    state_path: PathBuf,
    stm_path: PathBuf,
    ltm_base: PathBuf,
    socket_path: PathBuf,
    server_thread: Option<std::thread::JoinHandle<()>>,
    shutdown_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl IpcTestSystem {
    fn new(test_name: &str) -> Self {
        let state_path = temp_path(&format!("{test_name}_state"), "bin");
        let stm_path = temp_path(&format!("{test_name}_stm"), "bin");
        let ltm_base = temp_base(&format!("{test_name}_ltm"));
        let socket_path = temp_path(&format!("{test_name}_sock"), "sock");

        // Pre-create the state files so the server can open them
        let mmap = MmapState::create(&state_path, 1, 1000).expect("create state");
        let stm = RingBuffer::create(&stm_path, 16).expect("create stm");
        let ltm = LtmStore::create(&ltm_base, 64).expect("create ltm");

        // Drop them — the server will re-open
        drop(mmap);
        drop(stm);
        drop(ltm);

        Self {
            state_path,
            stm_path,
            ltm_base,
            socket_path,
            server_thread: None,
            shutdown_flag: None,
        }
    }

    fn start_server(&mut self) {
        let mmap = std::sync::Arc::new(MmapState::open(&self.state_path).expect("open state"));
        let stm = std::sync::Arc::new(RingBuffer::open(&self.stm_path).expect("open stm"));
        let ltm = LtmStore::open(&self.ltm_base).expect("open ltm");
        let ltm = std::sync::Arc::new(std::sync::Mutex::new(ltm));
        let socket_path = self.socket_path.clone();

        let server = IpcServer::new(&socket_path);
        let shutdown_flag = server.shutdown_flag();

        let thread = std::thread::spawn(move || {
            server.run(mmap, stm, ltm, default_handler);
        });

        self.server_thread = Some(thread);
        self.shutdown_flag = Some(shutdown_flag);

        // Wait for the server to be ready
        std::thread::sleep(Duration::from_millis(50));
    }

    fn client(&self) -> IpcClient {
        for _ in 0..10 {
            match IpcClient::connect(&self.socket_path) {
                Ok(c) => return c,
                Err(_) => std::thread::sleep(Duration::from_millis(50)),
            }
        }
        IpcClient::connect(&self.socket_path).expect("connect")
    }
}

impl Drop for IpcTestSystem {
    fn drop(&mut self) {
        if let Some(flag) = &self.shutdown_flag {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(thread) = self.server_thread.take() {
            let _ = thread.join();
        }
        let _ = std::fs::remove_file(&self.state_path);
        let _ = std::fs::remove_file(&self.stm_path);
        let _ = std::fs::remove_file(self.ltm_base.with_extension("idx"));
        let _ = std::fs::remove_file(self.ltm_base.with_extension("dat"));
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

// ─── Wire type sizes ──────────────────────────────────────────

#[test]
fn test_neuro_summary_size() {
    assert_eq!(core::mem::size_of::<NeuroSummary>(), 32);
}

#[test]
fn test_memory_stats_size() {
    assert_eq!(core::mem::size_of::<MemoryStats>(), 24);
}

#[test]
fn test_plasticity_profile_size() {
    assert_eq!(core::mem::size_of::<PlasticityProfile>(), 40);
}

// ─── Ping ─────────────────────────────────────────────────────

#[test]
fn test_ping() {
    let mut sys = IpcTestSystem::new("ping");
    sys.start_server();

    let mut client = sys.client();
    assert!(client.ping().expect("ping"), "ping should return true");
}

// ─── Get state ────────────────────────────────────────────────

#[test]
fn test_get_state() {
    let mut sys = IpcTestSystem::new("getstate");
    sys.start_server();

    let mut client = sys.client();
    let state = client.get_state().expect("get_state");

    // The state should have valid magic
    assert!(state.verify().is_ok(), "state should be valid");
    // Schema version should be 3 (v3: 18-chemical layout)
    assert_eq!(state.header.version, 3);
}

// ─── Get neuro summary ────────────────────────────────────────

#[test]
fn test_get_neuro_summary() {
    let mut sys = IpcTestSystem::new("neurosum");
    sys.start_server();

    let mut client = sys.client();
    let summary = client.get_neuro_summary().expect("get_neuro_summary");

    // At rest, arousal and valence should be in reasonable ranges
    assert!(summary.arousal >= 0.0 && summary.arousal <= 1.0);
    assert!(summary.valence >= -1.0 && summary.valence <= 1.0);
    assert!(summary.plasticity_gate >= 0.0 && summary.plasticity_gate <= 1.0);
}

// ─── Get plasticity profile ───────────────────────────────────

#[test]
fn test_get_plasticity_profile() {
    let mut sys = IpcTestSystem::new("plasticity");
    sys.start_server();

    let mut client = sys.client();
    let profile = client
        .get_plasticity_profile()
        .expect("get_plasticity_profile");

    // At rest, all levels should be in [0, 1]
    assert!(profile.plasticity_gate >= 0.0 && profile.plasticity_gate <= 1.0);
    assert!(profile.bdnf_effective >= 0.0 && profile.bdnf_effective <= 1.0);
    assert!(profile.cortisol_effective >= 0.0 && profile.cortisol_effective <= 1.0);
    assert!(profile.dopamine_effective >= 0.0 && profile.dopamine_effective <= 1.0);
    assert!(profile.serotonin_effective >= 0.0 && profile.serotonin_effective <= 1.0);

    // Tonic levels should also be bounded
    assert!(profile.bdnf_tonic >= 0.0 && profile.bdnf_tonic <= 1.0);
    assert!(profile.cortisol_tonic >= 0.0 && profile.cortisol_tonic <= 1.0);

    // At rest (no metaplasticity yet), coupling drift should be ~0
    assert!(
        profile.coupling_drift < 0.01,
        "coupling drift should be near zero at rest, got {}",
        profile.coupling_drift
    );

    // Mean receptor sensitivity should be positive (receptors don't go negative)
    assert!(profile.mean_receptor_sensitivity > 0.0);

    // Emergent phase should be a valid phase value (0-7)
    assert!(profile.emergent_phase <= 7);
}

#[test]
fn test_plasticity_profile_after_stress() {
    let mut sys = IpcTestSystem::new("plasticity_stress");
    sys.start_server();

    let mut client = sys.client();

    // Apply a strong cortisol impulse to mimic acute stress
    client
        .neuro_impulse(NeurochemicalId::Cortisol, 0.8)
        .expect("cortisol impulse");

    // Let the dynamics settle slightly
    std::thread::sleep(Duration::from_millis(100));

    let profile = client
        .get_plasticity_profile()
        .expect("get_plasticity_profile");

    // After a cortisol impulse, cortisol effective should be elevated
    assert!(
        profile.cortisol_effective > 0.3,
        "cortisol should be elevated after impulse, got {}",
        profile.cortisol_effective
    );
}

// ─── Store event ──────────────────────────────────────────────

#[test]
fn test_store_event() {
    let mut sys = IpcTestSystem::new("storeevent");
    sys.start_server();

    let mut client = sys.client();
    let tag = [0.5f32; 12];
    client
        .store_event(1000, 0, 4, 0.8, &tag, "Test event via IPC")
        .expect("store_event");

    // Verify it was stored by checking memory stats
    let stats = client.request(cmd::GET_MEMORY_STATS, &[]).expect("stats");
    let stm_count = u64::from_le_bytes(stats[0..8].try_into().unwrap());
    assert!(stm_count > 0, "STM should have entries after store_event");
}

// ─── Set zone ─────────────────────────────────────────────────

#[test]
fn test_set_zone() {
    let mut sys = IpcTestSystem::new("setzone");
    sys.start_server();

    let mut client = sys.client();
    client.set_zone(CognitiveZone::Coding).expect("set_zone");

    // Verify the zone was set
    let state = client.get_state().expect("get_state");
    assert_eq!(state.zones.zone(), CognitiveZone::Coding);
}

#[test]
fn test_set_zone_applies_despite_stale_override() {
    // Regression test: `zone_override_active` persists in the mmap'd
    // core state. Previously SET_ZONE was gated on that byte being 0
    // and never cleared it, so once it latched to 1 every future
    // SET_ZONE was silently ignored — freezing the zone at Idle and
    // desyncing the cognitive mind (e.g. its sleep state) from the
    // daemon. An explicit SET_ZONE must always apply.
    let mut sys = IpcTestSystem::new("setzone_stale_override");

    // Latch the override byte before the server starts, modelling a
    // stale manual override restored from disk.
    {
        let mmap = MmapState::open(&sys.state_path).expect("open state");
        mmap.modify(current_ms(), |s| {
            s.zones.transition_to(CognitiveZone::Idle, current_ms());
        })
        .expect("latch override");
        let snap = mmap.read_consistent().expect("read state");
        assert_eq!(
            snap.zones.zone_override_active, 1,
            "precondition: the override byte should be latched"
        );
    }

    sys.start_server();

    let mut client = sys.client();
    client
        .set_zone(CognitiveZone::Sleeping)
        .expect("set_zone");

    let state = client.get_state().expect("get_state");
    assert_eq!(
        state.zones.zone(),
        CognitiveZone::Sleeping,
        "SET_ZONE must apply even when a manual override is latched"
    );
}

// ─── Get phase ────────────────────────────────────────────────

#[test]
fn test_get_phase() {
    let mut sys = IpcTestSystem::new("getphase");
    sys.start_server();

    let mut client = sys.client();
    let resp = client.request(cmd::GET_PHASE, &[]).expect("get_phase");

    assert!(!resp.is_empty(), "phase response should not be empty");
    // First byte is the phase ID
    let _phase = resp[0];
    // Next 4 bytes are arousal (f32)
    let arousal = f32::from_le_bytes(resp[1..5].try_into().unwrap());
    // Next 4 bytes are valence (f32)
    let valence = f32::from_le_bytes(resp[5..9].try_into().unwrap());

    assert!((0.0..=1.0).contains(&arousal), "arousal in range");
    assert!((-1.0..=1.0).contains(&valence), "valence in range");
}

// ─── Neuro impulse ────────────────────────────────────────────

#[test]
fn test_neuro_impulse() {
    let mut sys = IpcTestSystem::new("impulse");
    sys.start_server();

    let mut client = sys.client();

    // Get baseline dopamine
    let _summary_before = client.get_neuro_summary().expect("summary before");

    // Apply a dopamine impulse
    client
        .neuro_impulse(NeurochemicalId::Dopamine, 0.5)
        .expect("impulse");

    // The impulse should have affected the state
    let summary_after = client.get_neuro_summary().expect("summary after");

    // The state should still be valid (impulse doesn't break anything)
    assert!(summary_after.arousal >= 0.0);
}

// ─── Neuro adjust baseline ────────────────────────────────────

#[test]
fn test_neuro_adjust_baseline() {
    let mut sys = IpcTestSystem::new("adjust_baseline");
    sys.start_server();

    let mut client = sys.client();

    // Get the raw adenosine baseline before adjustment by reading state
    let state_before = client.get_state().expect("state before");
    let adn_idx = NeurochemicalId::Adenosine as usize;
    let baseline_before = state_before.neurochemicals.chemicals[adn_idx].baseline;

    // Adjust the adenosine baseline downward by 0.05 (small enough to
    // not hit the 0.05 floor from the default ~0.20 baseline)
    let delta: f32 = -0.05;
    let delta_bytes = delta.to_le_bytes();
    let mut msg = vec![NeurochemicalId::Adenosine as u8];
    msg.extend_from_slice(&delta_bytes);
    let resp = client
        .request(cmd::NEURO_ADJUST_BASELINE, &msg)
        .expect("adjust baseline");
    assert_eq!(resp, vec![1], "adjust baseline should return ack=1");

    // Verify the baseline was actually lowered
    let state_after = client.get_state().expect("state after");
    let baseline_after = state_after.neurochemicals.chemicals[adn_idx].baseline;
    assert!(
        baseline_after < baseline_before,
        "baseline should decrease after negative adjust: was {}, now {}",
        baseline_before,
        baseline_after
    );
    assert!(
        (baseline_after - (baseline_before + delta)).abs() < 0.01,
        "baseline should be ~{} lower: expected ~{}, got {}",
        delta,
        baseline_before + delta,
        baseline_after
    );
}

// ─── Get memory stats ─────────────────────────────────────────

#[test]
fn test_get_memory_stats() {
    let mut sys = IpcTestSystem::new("memstats");
    sys.start_server();

    let mut client = sys.client();
    let resp = client.request(cmd::GET_MEMORY_STATS, &[]).expect("stats");

    assert!(
        resp.len() >= 24,
        "stats response should be at least 24 bytes"
    );

    let stm_count = u64::from_le_bytes(resp[0..8].try_into().unwrap());
    let ltm_count = u64::from_le_bytes(resp[8..16].try_into().unwrap());
    let ltm_capacity = u32::from_le_bytes(resp[16..20].try_into().unwrap());

    assert_eq!(stm_count, 0, "STM should be empty initially");
    assert_eq!(ltm_count, 0, "LTM should be empty initially");
    assert!(ltm_capacity > 0, "LTM capacity should be positive");
}

// ─── Sync ─────────────────────────────────────────────────────

#[test]
fn test_sync() {
    let mut sys = IpcTestSystem::new("sync");
    sys.start_server();

    let mut client = sys.client();
    client.sync().expect("sync");
    // If we get here without error, sync worked
}

// ─── Multiple requests ────────────────────────────────────────

#[test]
fn test_multiple_requests() {
    let mut sys = IpcTestSystem::new("multi");
    sys.start_server();

    let mut client = sys.client();

    // Send several requests in sequence
    for i in 0..10 {
        let tag = [0.5f32; 12];
        client
            .store_event(1000 + i, 0, 4, 0.7, &tag, &format!("Event {i}"))
            .expect("store");
    }

    let resp = client.request(cmd::GET_MEMORY_STATS, &[]).expect("stats");
    let stm_count = u64::from_le_bytes(resp[0..8].try_into().unwrap());
    assert_eq!(stm_count, 10, "STM should have 10 entries");
}

// ─── Unknown command ──────────────────────────────────────────

#[test]
fn test_unknown_command() {
    let mut sys = IpcTestSystem::new("unknown");
    sys.start_server();

    let mut client = sys.client();
    let resp = client.request(255, &[]).expect("unknown cmd");
    // Should return [UNKNOWN_COMMAND] (error code 3)
    assert_eq!(resp, vec![genesis::daemon::ipc::error::UNKNOWN_COMMAND]);
}

// ─── Shutdown ─────────────────────────────────────────────────

#[test]
fn test_shutdown() {
    let mut sys = IpcTestSystem::new("shutdown");
    sys.start_server();

    let mut client = sys.client();
    client.shutdown().expect("shutdown");

    // Give the server time to shut down
    std::thread::sleep(Duration::from_millis(100));

    // The server should have stopped
    if let Some(flag) = &sys.shutdown_flag {
        assert!(
            flag.load(std::sync::atomic::Ordering::Relaxed),
            "server should be shut down"
        );
    }
}

// ─── Update module status ─────────────────────────────────────

#[test]
fn test_update_module_status() {
    let mut sys = IpcTestSystem::new("modstatus");
    sys.start_server();

    let mut client = sys.client();

    // Register Sensory (6) as Running (2)
    client
        .update_module_status(ModuleId::Sensory as u8, ModuleStatus::Running as u8)
        .expect("update sensory");

    // Register Motor (7) as Running (2)
    client
        .update_module_status(ModuleId::Motor as u8, ModuleStatus::Running as u8)
        .expect("update motor");

    // Read back the state and verify the manifest
    let state = client.get_state().expect("get_state");

    let sensory = state
        .manifest
        .get(ModuleId::Sensory)
        .expect("sensory module");
    assert_eq!(
        sensory.status(),
        ModuleStatus::Running,
        "sensory should be Running"
    );
    assert!(
        sensory.last_heartbeat > 0,
        "sensory should have a heartbeat timestamp"
    );

    let motor = state.manifest.get(ModuleId::Motor).expect("motor module");
    assert_eq!(
        motor.status(),
        ModuleStatus::Running,
        "motor should be Running"
    );
    assert!(
        motor.last_heartbeat > 0,
        "motor should have a heartbeat timestamp"
    );

    // The active_count should reflect the cognitive modules we registered
    // (plus any others that were already alive)
    assert!(
        state.manifest.active_count >= 2,
        "active_count should be >= 2, got {}",
        state.manifest.active_count
    );
}

#[test]
fn test_update_module_status_invalid_module() {
    let mut sys = IpcTestSystem::new("modstatus_invalid");
    sys.start_server();

    let mut client = sys.client();

    // Module ID 200 is invalid — should return ack=0
    let resp = client
        .request(cmd::UPDATE_MODULE_STATUS, &[200, 2])
        .expect("request");
    assert_eq!(resp, vec![0], "invalid module_id should return 0");
}

#[test]
fn test_update_module_status_invalid_status() {
    let mut sys = IpcTestSystem::new("modstatus_bad_status");
    sys.start_server();

    let mut client = sys.client();

    // Status 99 is invalid — should return ack=0
    let resp = client
        .request(cmd::UPDATE_MODULE_STATUS, &[6, 99])
        .expect("request");
    assert_eq!(resp, vec![0], "invalid status should return 0");
}

#[test]
fn test_update_module_status_deregister() {
    let mut sys = IpcTestSystem::new("modstatus_dereg");
    sys.start_server();

    let mut client = sys.client();

    // Register Sensory as Running
    client
        .update_module_status(ModuleId::Sensory as u8, ModuleStatus::Running as u8)
        .expect("register");

    // Verify it's running
    let state = client.get_state().expect("get_state");
    let sensory = state.manifest.get(ModuleId::Sensory).expect("sensory");
    assert_eq!(sensory.status(), ModuleStatus::Running);

    // Deregister — set to Stopped
    client
        .update_module_status(ModuleId::Sensory as u8, ModuleStatus::Stopped as u8)
        .expect("deregister");

    // Verify it's stopped
    let state = client.get_state().expect("get_state");
    let sensory = state.manifest.get(ModuleId::Sensory).expect("sensory");
    assert_eq!(sensory.status(), ModuleStatus::Stopped);
}

// ─── Get recent episodes ──────────────────────────────────────

#[test]
fn test_get_recent_episodes_all() {
    let mut sys = IpcTestSystem::new("recent_all");
    // Pre-store episodes in LTM before starting the server
    {
        let mut ltm = LtmStore::open(&sys.ltm_base).expect("open ltm");
        let tag = [0.5f32; 12];
        ltm.store(1000, 0.5, tag, [0.4, 0.3, 0.2, 0.5], 0, 0, "first memory")
            .expect("store 1");
        ltm.store(
            2000,
            0.7,
            tag,
            [0.4, 0.3, 0.2, 0.5],
            2,
            9,
            "a dream insight",
        )
        .expect("store 2");
        ltm.store(3000, 0.3, tag, [0.4, 0.3, 0.2, 0.5], 2, 0, "an association")
            .expect("store 3");
    }
    sys.start_server();

    let mut client = sys.client();
    // Request all recent episodes (source_filter=255)
    let resp = client
        .request(cmd::GET_RECENT_EPISODES, &[10, 255])
        .expect("recent all");

    assert!(resp.len() >= 4, "response should have count prefix");
    let count = u32::from_le_bytes(resp[0..4].try_into().unwrap());
    assert_eq!(count, 3, "should return all 3 episodes");

    // Parse first episode (most recent = episode 3)
    let mut offset = 4;
    let ep_id = u64::from_le_bytes(resp[offset..offset + 8].try_into().unwrap());
    offset += 8;
    let _ts = u64::from_le_bytes(resp[offset..offset + 8].try_into().unwrap());
    offset += 8;
    let _sal = f32::from_le_bytes(resp[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let _et = resp[offset];
    offset += 1;
    let sm = resp[offset];
    offset += 1;
    let text_len = u32::from_le_bytes(resp[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let text = String::from_utf8_lossy(&resp[offset..offset + text_len]);

    assert_eq!(ep_id, 3, "most recent episode should be first");
    assert_eq!(sm, 0, "episode 3 source_module should be 0");
    assert_eq!(text, "an association");
}

#[test]
fn test_get_recent_episodes_filtered() {
    let mut sys = IpcTestSystem::new("recent_filtered");
    {
        let mut ltm = LtmStore::open(&sys.ltm_base).expect("open ltm");
        let tag = [0.5f32; 12];
        ltm.store(1000, 0.5, tag, [0.4, 0.3, 0.2, 0.5], 0, 0, "regular memory")
            .expect("store 1");
        ltm.store(
            2000,
            0.7,
            tag,
            [0.6, 0.2, 0.1, 0.6],
            2,
            9,
            "[dream-insight] ep 1 connects to ep 2",
        )
        .expect("store 2");
        ltm.store(
            3000,
            0.3,
            tag,
            [0.4, 0.3, 0.2, 0.5],
            2,
            0,
            "[association] ep 1 ↔ ep 2",
        )
        .expect("store 3");
        ltm.store(
            4000,
            0.7,
            tag,
            [0.6, 0.2, 0.1, 0.6],
            2,
            9,
            "[dream-insight] ep 3 connects to ep 5",
        )
        .expect("store 4");
    }
    sys.start_server();

    let mut client = sys.client();
    // Request only dream insights (source_module=9)
    let resp = client
        .request(cmd::GET_RECENT_EPISODES, &[10, 9])
        .expect("recent dreams");

    let count = u32::from_le_bytes(resp[0..4].try_into().unwrap());
    assert_eq!(count, 2, "should return 2 dream insights");

    // Parse first (most recent = episode 4)
    let mut offset = 4;
    let ep_id = u64::from_le_bytes(resp[offset..offset + 8].try_into().unwrap());
    offset += 8;
    offset += 8 + 4; // skip timestamp + salience
    offset += 1 + 1; // skip event_type + source_module
    let text_len = u32::from_le_bytes(resp[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let text = String::from_utf8_lossy(&resp[offset..offset + text_len]);

    assert_eq!(ep_id, 4, "most recent dream insight should be first");
    assert!(
        text.starts_with("[dream-insight]"),
        "text should be a dream insight"
    );
}

#[test]
fn test_get_recent_episodes_empty() {
    let mut sys = IpcTestSystem::new("recent_empty");
    sys.start_server();

    let mut client = sys.client();
    let resp = client
        .request(cmd::GET_RECENT_EPISODES, &[10, 255])
        .expect("recent empty");

    let count = u32::from_le_bytes(resp[0..4].try_into().unwrap());
    assert_eq!(count, 0, "empty LTM should return 0 episodes");
}

#[test]
fn test_archive_episode() {
    let mut sys = IpcTestSystem::new("archive_episode");
    let ep_id = {
        let mut ltm = LtmStore::open(&sys.ltm_base).expect("open ltm");
        let tag = [0.5f32; 12];

        ltm.store(
            1000,
            0.5,
            tag,
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "archivable memory",
        )
        .expect("store")
    };
    sys.start_server();

    let mut client = sys.client();

    // Verify the episode exists
    let resp = client
        .request(cmd::RETRIEVE_EPISODE, &ep_id.to_le_bytes())
        .expect("retrieve before archive");
    assert_eq!(resp[0], 1, "episode should exist before archive");

    // Archive it
    let resp = client
        .request(cmd::ARCHIVE_EPISODE, &ep_id.to_le_bytes())
        .expect("archive");
    assert_eq!(resp[0], 1, "archive should succeed");

    // Verify it's STILL retrievable — this is the key difference
    // from deletion. Archived episodes remain accessible by ID.
    let resp = client
        .request(cmd::RETRIEVE_EPISODE, &ep_id.to_le_bytes())
        .expect("retrieve after archive");
    assert_eq!(resp[0], 1, "archived episode should still be retrievable");

    // Archiving a non-existent episode should return 0 (not error)
    let resp = client
        .request(cmd::ARCHIVE_EPISODE, &999_999u64.to_le_bytes())
        .expect("archive non-existent");
    assert_eq!(resp[0], 0, "archiving non-existent episode should return 0");
}

// ─── Store episode (direct LTM) ────────────────────────────────

#[test]
fn test_store_episode() {
    let mut sys = IpcTestSystem::new("store_episode");
    sys.start_server();

    let mut client = sys.client();
    let tag = [0.5f32; 12];

    // store_episode returns the real LTM episode ID synchronously.
    let ep_id = client
        .store_episode(1000, 0, 4, 0.8, &tag, "Direct LTM store")
        .expect("store_episode")
        .expect("store_episode should succeed");
    assert!(ep_id > 0, "episode ID must be positive");

    // Verify the episode is immediately retrievable by the returned ID.
    let resp = client
        .request(cmd::RETRIEVE_EPISODE, &ep_id.to_le_bytes())
        .expect("retrieve");
    assert_eq!(resp[0], 1, "episode should be retrievable immediately");
    let retrieved_id = u64::from_le_bytes(resp[1..9].try_into().unwrap());
    assert_eq!(retrieved_id, ep_id, "retrieved ID must match returned ID");

    // Verify the text matches.
    let text_len = u32::from_le_bytes(
        resp[1 + 8 + 8 + 4 + 8 + 16 + 48 + 1 + 1..][..4]
            .try_into()
            .unwrap(),
    ) as usize;
    let text_start = 1 + 8 + 8 + 4 + 8 + 16 + 48 + 1 + 1 + 4;
    let text = String::from_utf8_lossy(&resp[text_start..text_start + text_len]);
    assert_eq!(text, "Direct LTM store", "text must match");
}

#[test]
fn test_store_episode_multiple_ids_increment() {
    let mut sys = IpcTestSystem::new("store_episode_ids");
    sys.start_server();

    let mut client = sys.client();
    let tag = [0.5f32; 12];

    let id1 = client
        .store_episode(1000, 0, 4, 0.5, &tag, "first")
        .expect("store")
        .unwrap();
    let id2 = client
        .store_episode(2000, 0, 4, 0.5, &tag, "second")
        .expect("store")
        .unwrap();

    assert!(
        id2 == id1 + 1,
        "consecutive stores must produce consecutive IDs ({id1} → {id2})"
    );
}

#[test]
fn test_store_episode_bypasses_stm() {
    // STORE_EPISODE goes directly to LTM, not STM. After a store,
    // STM count should remain 0 while LTM count should be 1.
    let mut sys = IpcTestSystem::new("store_episode_bypass");
    sys.start_server();

    let mut client = sys.client();
    let tag = [0.5f32; 12];
    let _ = client
        .store_episode(1000, 0, 4, 0.5, &tag, "bypass test")
        .expect("store")
        .unwrap();

    let stats = client.request(cmd::GET_MEMORY_STATS, &[]).expect("stats");
    let stm_count = u64::from_le_bytes(stats[0..8].try_into().unwrap());
    let ltm_count = u64::from_le_bytes(stats[8..16].try_into().unwrap());
    assert_eq!(
        stm_count, 0,
        "STM should be empty (store_episode bypasses STM)"
    );
    assert!(ltm_count > 0, "LTM should have the episode");
}

#[test]
fn test_search_episodes() {
    let mut sys = IpcTestSystem::new("search_episodes");
    {
        let mut ltm = LtmStore::open(&sys.ltm_base).expect("open ltm");
        let tag = [0.5f32; 12];
        for i in 0..10u64 {
            ltm.store(
                1000 + i,
                0.3 + (i as f32) * 0.05,
                tag,
                [0.4, 0.3, 0.2, 0.5],
                0,
                (i % 3) as u8,
                &format!("episode {i}"),
            )
            .expect("store");
        }
    }
    sys.start_server();

    let mut client = sys.client();

    // Search first 5 episodes (offset=0, limit=5)
    let payload = 5u32
        .to_le_bytes()
        .iter()
        .chain(0u32.to_le_bytes().iter())
        .copied()
        .collect::<Vec<_>>();
    let resp = client
        .request(cmd::SEARCH_EPISODES, &payload)
        .expect("search page 1");

    let count = u32::from_le_bytes(resp[0..4].try_into().unwrap());
    assert_eq!(count, 5, "first page should return 5 episodes");

    // Parse first episode
    let offset = 4;
    let ep_id = u64::from_le_bytes(resp[offset..offset + 8].try_into().unwrap());
    // Episodes are in ascending ID order, so first should be episode 1
    assert_eq!(ep_id, 1, "first episode in ascending order should be ID 1");

    // Search next 5 (offset=5, limit=5)
    let payload = 5u32
        .to_le_bytes()
        .iter()
        .chain(5u32.to_le_bytes().iter())
        .copied()
        .collect::<Vec<_>>();
    let resp = client
        .request(cmd::SEARCH_EPISODES, &payload)
        .expect("search page 2");

    let count = u32::from_le_bytes(resp[0..4].try_into().unwrap());
    assert_eq!(count, 5, "second page should return 5 episodes");

    // Parse first episode of second page
    let offset = 4;
    let ep_id = u64::from_le_bytes(resp[offset..offset + 8].try_into().unwrap());
    assert_eq!(ep_id, 6, "second page should start at episode 6");

    // Search beyond the end (offset=8, limit=5) — should return 2
    let payload = 5u32
        .to_le_bytes()
        .iter()
        .chain(8u32.to_le_bytes().iter())
        .copied()
        .collect::<Vec<_>>();
    let resp = client
        .request(cmd::SEARCH_EPISODES, &payload)
        .expect("search partial page");

    let count = u32::from_le_bytes(resp[0..4].try_into().unwrap());
    assert_eq!(count, 2, "partial page should return 2 episodes");
}

#[test]
fn test_read_message_rejects_oversized_length_prefix() {
    // Regression: before the MAX_MESSAGE_LEN bound, a client could send
    // a u32::MAX length prefix and the daemon would attempt to allocate
    // ~4 GiB. Verify it now rejects the request before allocating.
    use std::io::Write;
    use std::os::unix::net::UnixStream;

    let (mut client, mut server) = UnixStream::pair().expect("socket pair");

    // Total length = MAX_MESSAGE_LEN + 1. Only the 4-byte prefix is sent;
    // the daemon should reject it without waiting for the body.
    let oversize = (MAX_MESSAGE_LEN + 1) as u32;
    client
        .write_all(&oversize.to_le_bytes())
        .expect("write prefix");
    client.flush().expect("flush");
    // Half-close the write side so the server sees EOF after the prefix
    // (it shouldn't try to read the body anyway, but this makes the
    // test deterministic).
    drop(client);

    let err = read_message(&mut server).expect_err("oversized message should be rejected");
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidData,
        "should be InvalidData, not an allocation error"
    );
    assert!(
        err.to_string().contains("exceeds maximum"),
        "error should mention the maximum: {err}"
    );
}

// ─── Concurrent clients ───────────────────────────────────────

#[test]
fn test_concurrent_clients() {
    let mut sys = IpcTestSystem::new("concurrent");
    sys.start_server();

    let num_clients = 4;
    let requests_per_client = 4; // matches the STM capacity of 16 in IpcTestSystem
    let mut handles = Vec::with_capacity(num_clients);

    for client_id in 0..num_clients {
        let mut client = sys.client();
        handles.push(std::thread::spawn(move || {
            let tag = [0.5f32; 12];
            for i in 0..requests_per_client {
                client
                    .store_event(
                        1000 + (client_id * 1000 + i) as u64,
                        0,
                        4,
                        0.6,
                        &tag,
                        &format!("client {client_id} event {i}"),
                    )
                    .expect("store event");
            }
        }));
    }

    for handle in handles {
        handle.join().expect("client thread");
    }

    let mut client = sys.client();
    let resp = client.request(cmd::GET_MEMORY_STATS, &[]).expect("stats");
    let stm_count = u64::from_le_bytes(resp[0..8].try_into().unwrap());
    assert_eq!(
        stm_count,
        (num_clients * requests_per_client) as u64,
        "all concurrent stores should land in STM"
    );
}

// ─── Body state (interoception) ───────────────────────────────

#[test]
fn test_get_body_state() {
    // The shared body state is a process-global static, so all
    // sub-scenarios run sequentially in one test to avoid races.
    use genesis::daemon::interoception::{BodyState, publish_body_state};

    let mut sys = IpcTestSystem::new("body_state");
    sys.start_server();

    // Scenario 1: normal state
    let normal = BodyState {
        cpu_temp_c: 42.0,
        temperature: 0.42,
        arousal_freq: 0.5,
        cognitive_load: 0.3,
        io_activity: 0.2,
        stress_load: 0.1,
        energy_reserve: 1.0,
        on_ac_power: true,
        num_cores: 2,
        distressed: false,
        autonomic_rate: 1.2,
        thermoregulatory_effort: 0.3,
        metabolic_rate: 0.05, // idle power draw
        core_voltage: 0.85, // idle Vcore
        supply_voltage: 12.6, // full battery
        core_activity: 0.4, // moderate silicon switching
        uncore_activity: 0.2,
        dram_activity: 0.5, // active memory traffic
        cache_miss_rate: 0.03,
        branch_miss_rate: 0.02,
        // Empty — the daemon sends structured fields; the cognitive
        // mind's language engine composes the description.
        description: String::new(),
    };
    publish_body_state(&normal);
    let mut client = sys.client();
    let resp = client.get_body_state().expect("get_body_state");
    assert!((resp.cpu_temp_c - 42.0).abs() < 0.01);
    assert!((resp.temperature - 0.42).abs() < 0.01);
    assert!((resp.arousal_freq - 0.5).abs() < 0.01);
    assert!((resp.cognitive_load - 0.3).abs() < 0.01);
    assert!((resp.io_activity - 0.2).abs() < 0.01);
    assert!((resp.stress_load - 0.1).abs() < 0.01);
    assert!((resp.energy_reserve - 1.0).abs() < 0.01);
    assert!(resp.on_ac_power);
    assert_eq!(resp.num_cores, 2);
    assert!(!resp.distressed);
    // Autonomic and thermoregulatory fields must round-trip through
    // the wire — the cognitive mind's affect inference reads
    // autonomic_rate/thermoregulatory_effort, so a serialization
    // regression that drops them would silently break her self-model.
    assert!(
        (resp.autonomic_rate - 1.2).abs() < 0.01,
        "autonomic_rate should be 1.2, got {}",
        resp.autonomic_rate
    );
    assert!(
        (resp.thermoregulatory_effort - 0.3).abs() < 0.01,
        "thermoregulatory_effort should be 0.3, got {}",
        resp.thermoregulatory_effort
    );
    // Voltage fields must round-trip — the cognitive mind reads
    // supply_voltage for energy reserve health (CRH impulse when
    // critically low) and core_voltage for CPU electrical state.
    assert!(
        (resp.core_voltage - 0.85).abs() < 0.01,
        "core_voltage should be 0.85 (idle Vcore), got {}",
        resp.core_voltage
    );
    assert!(
        (resp.supply_voltage - 12.6).abs() < 0.01,
        "supply_voltage should be 12.6 (full battery), got {}",
        resp.supply_voltage
    );
    // Silicon fields must round-trip — the cognitive mind reads
    // them for electron-level interoception (RAPL domain activity,
    // microarchitectural prediction errors).
    assert!(
        (resp.core_activity - 0.4).abs() < 0.01,
        "core_activity should be 0.4, got {}",
        resp.core_activity
    );
    assert!(
        (resp.dram_activity - 0.5).abs() < 0.01,
        "dram_activity should be 0.5, got {}",
        resp.dram_activity
    );
    assert!(
        (resp.branch_miss_rate - 0.02).abs() < 0.01,
        "branch_miss_rate should be 0.02, got {}",
        resp.branch_miss_rate
    );
    assert!(resp.description.is_empty());

    // Scenario 2: comfortable state
    let comfy = BodyState {
        cpu_temp_c: 55.0,
        temperature: 0.55,
        arousal_freq: 0.8,
        cognitive_load: 0.6,
        io_activity: 0.1,
        stress_load: 0.3,
        energy_reserve: 0.9,
        on_ac_power: true,
        num_cores: 8,
        distressed: false,
        autonomic_rate: 0.9,
        thermoregulatory_effort: 0.45,
        metabolic_rate: 0.08, // light power draw
        core_voltage: 0.90, // normal Vcore
        supply_voltage: 12.5, // good battery
        core_activity: 0.2, // light silicon switching
        uncore_activity: 0.1,
        dram_activity: 0.3,
        cache_miss_rate: 0.02,
        branch_miss_rate: 0.01,
        // Empty — the daemon sends structured fields; the cognitive
        // mind's language engine composes the description.
        description: String::new(),
    };
    publish_body_state(&comfy);
    let mut client2 = sys.client();
    let resp2 = client2.get_body_state().expect("get_body_state");
    assert!((resp2.cpu_temp_c - 55.0).abs() < 0.01);
    assert!(resp2.on_ac_power);
    assert_eq!(resp2.num_cores, 8);
    // High autonomic regularity under low stress = healthy, flexible
    // cognition — the autonomic and thermoregulatory fields must
    // reflect the comfortable state, not the distressed one from the
    // previous scenario.
    assert!(
        (resp2.autonomic_rate - 0.9).abs() < 0.01,
        "autonomic_rate should be 0.9, got {}",
        resp2.autonomic_rate
    );
    assert!(
        (resp2.thermoregulatory_effort - 0.45).abs() < 0.01,
        "thermoregulatory_effort should be 0.45, got {}",
        resp2.thermoregulatory_effort
    );
    assert!(
        (resp2.core_voltage - 0.90).abs() < 0.01,
        "core_voltage should be 0.90 (normal Vcore), got {}",
        resp2.core_voltage
    );
    assert!(
        (resp2.supply_voltage - 12.5).abs() < 0.01,
        "supply_voltage should be 12.5 (good battery), got {}",
        resp2.supply_voltage
    );
    assert!(resp2.description.is_empty());

    // Scenario 3: distressed state
    let distressed = BodyState {
        cpu_temp_c: 92.0,
        temperature: 0.95,
        arousal_freq: 0.3,
        cognitive_load: 0.92,
        io_activity: 0.05,
        stress_load: 1.5,
        energy_reserve: 0.15,
        on_ac_power: false,
        num_cores: 4,
        distressed: true,
        // Physiologically distressed autonomic state: rapid autonomic
        // signal (high autonomic rate), rigid stressed cognition, fan
        // at full speed (thermoregulatory strain under high stress).
        autonomic_rate: 8.0,
        thermoregulatory_effort: 0.95,
        metabolic_rate: 0.8, // high power draw under distress
        core_voltage: 1.30, // Vcore maxed under thermal stress
        supply_voltage: 10.2, // critically low battery
        core_activity: 0.95, // silicon firing hard under distress
        uncore_activity: 0.7,
        dram_activity: 0.9, // heavy memory traffic
        cache_miss_rate: 0.4, // silicon being surprised
        branch_miss_rate: 0.2,
        // Empty — the cognitive mind's language engine composes the
        // description from the structured fields, not the daemon.
        description: String::new(),
    };
    publish_body_state(&distressed);
    let mut client3 = sys.client();
    let resp3 = client3.get_body_state().expect("get_body_state");
    assert!(resp3.distressed);
    assert!((resp3.cpu_temp_c - 92.0).abs() < 0.01);
    assert!(!resp3.on_ac_power);
    // stress_load = 1.5 (overloaded) must survive the wire intact.
    // The interoceptor produces [0,2] and the IPC serialization must
    // not clamp to [0,1] — otherwise the cognitive mind never sees
    // the "severely overloaded" signal that triggers distress.
    assert!(
        (resp3.stress_load - 1.5).abs() < 0.01,
        "stress_load should be 1.5 (overloaded), got {} — IPC must not clamp to [0,1]",
        resp3.stress_load
    );
    // Distressed autonomic state must round-trip — high autonomic
    // rate and high thermoregulatory effort are the physiological
    // stress signals the cognitive mind reads for distress inference.
    assert!(
        (resp3.autonomic_rate - 8.0).abs() < 0.01,
        "autonomic_rate should be 8.0 (high autonomic rate), got {}",
        resp3.autonomic_rate
    );
    assert!(
        (resp3.thermoregulatory_effort - 0.95).abs() < 0.01,
        "thermoregulatory_effort should be 0.95 (remove waste heat), got {}",
        resp3.thermoregulatory_effort
    );
    // Distressed voltage state: Vcore maxed under thermal stress,
    // battery critically low (10.2V < 10.8V threshold). These must
    // round-trip — the cognitive mind reads supply_voltage for the
    // survival stress signal (CRH impulse when critically low).
    assert!(
        (resp3.core_voltage - 1.30).abs() < 0.01,
        "core_voltage should be 1.30 (Vcore maxed under thermal stress), got {}",
        resp3.core_voltage
    );
    assert!(
        (resp3.supply_voltage - 10.2).abs() < 0.01,
        "supply_voltage should be 10.2 (critically low battery), got {}",
        resp3.supply_voltage
    );
    // The daemon sends an empty description — the cognitive mind's
    // language engine composes the actual words from the structured
    // fields above, using her concept network. The IPC layer must
    // faithfully transmit the empty string, not inject content.
    assert!(
        resp3.description.is_empty(),
        "description should be empty (daemon sends structured fields, \
        cognitive mind composes words), got: {:?}",
        resp3.description
    );
}

#[test]
fn test_get_lobe_telemetry() {
    // Per-lobe telemetry round-trips: the daemon publishes one
    // record per process in her tree (daemon/cognitive/retina),
    // each carrying its CPU/I/O share and miss ratios.
    use genesis::daemon::interoception::{
        Subsystem, SubsystemTelemetry, publish_subsystem_telemetry,
    };

    let mut sys = IpcTestSystem::new("lobe_telemetry");
    sys.start_server();

    publish_subsystem_telemetry(&[
        SubsystemTelemetry {
            subsystem: Subsystem::Daemon,
            pid: 100,
            cpu: 0.05,
            io: 0.02,
            cache_miss_rate: 0.01,
            branch_miss_rate: 0.005,
        },
        SubsystemTelemetry {
            subsystem: Subsystem::Cognitive,
            pid: 200,
            cpu: 0.4,
            io: 0.1,
            cache_miss_rate: 0.08,
            branch_miss_rate: 0.03,
        },
        SubsystemTelemetry {
            subsystem: Subsystem::Retina,
            pid: 300,
            cpu: 0.15,
            io: 0.3,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
        },
    ]);

    // A module section is appended after the lobes: non-Stopped
    // manifest entries reported by the cognitive mind. Set one
    // module running with a self-reported activity share.
    let mut mclient = sys.client();
    let mut mod_req = vec![4u8, 2u8]; // language, Running
    mod_req.extend_from_slice(&0.42f32.to_le_bytes()); // cpu_share
    assert!(
        mclient
            .request(cmd::UPDATE_MODULE_STATUS, &mod_req)
            .expect("update_module_status")
            == [1],
        "module status update should be acked"
    );

    let mut client = sys.client();
    let report = client
        .get_subsystem_telemetry()
        .expect("get_subsystem_telemetry");
    let subsystems = &report.subsystems;
    assert_eq!(subsystems.len(), 3);
    assert_eq!(subsystems[0].subsystem, 0, "first subsystem should be daemon (0)");
    assert_eq!(subsystems[0].pid, 100);
    assert!((subsystems[0].cpu - 0.05).abs() < 0.001);
    assert_eq!(subsystems[1].subsystem, 1, "second subsystem should be cognitive (1)");
    assert!((subsystems[1].cache_miss_rate - 0.08).abs() < 0.001);
    assert!((subsystems[1].branch_miss_rate - 0.03).abs() < 0.001);
    assert_eq!(subsystems[2].subsystem, 2, "third subsystem should be retina (2)");
    assert!((subsystems[2].io - 0.3).abs() < 0.001);

    // Module section: language (4) was set Running — it should
    // appear; stopped modules should not.
    let lang = report
        .modules
        .iter()
        .find(|m| m.module_id == 4)
        .expect("language module should be in the telemetry report");
    assert_eq!(lang.status, 2, "language module should be Running");
    assert!(
        (lang.cpu_share - 0.42).abs() < 0.001,
        "self-reported cpu_share should round-trip, got {}",
        lang.cpu_share
    );
}

#[test]
fn test_get_body_control() {
    // The shared body-control state is a process-global static, so all
    // sub-scenarios run sequentially in one test to avoid races. This
    // validates that the daemon's body-control serializer and the Rust
    // client parser agree on the wire format — including the EPP string
    // and turbo-gate byte, which the parser silently omitted before v2.
    use genesis::daemon::cpufreq::{BodyControlState, BoostState, publish_control_state};

    let mut sys = IpcTestSystem::new("body_control");
    sys.start_server();

    // Scenario 1: engaged, turbo permitted, EPP set.
    let engaged = BodyControlState {
        cpu_min_freq_khz: 1_000_000,
        cpu_max_freq_khz: 1_800_000,
        cpu_governor: "schedutil".to_string(),
        thermally_capped: false,
        thermal_cap_temp_c: 0.0,
        daemon_nice: -2,
        cognitive_nice: -4,
        io_class: "best-effort-0".to_string(),
        plasticity_gate: 0.8,
        controlling_cognitive: false,
        cpu_epp: "performance".to_string(),
        cpu_boost: BoostState::Enabled,
        description: String::new(),
    };
    publish_control_state(&engaged);
    let mut client = sys.client();
    let resp = client.get_body_control().expect("get_body_control");
    assert_eq!(resp.cpu_min_freq_khz, 1_000_000);
    assert_eq!(resp.cpu_max_freq_khz, 1_800_000);
    assert_eq!(resp.cpu_governor, "schedutil");
    assert!(!resp.thermally_capped);
    assert_eq!(resp.daemon_nice, -2);
    assert_eq!(resp.cognitive_nice, -4);
    assert_eq!(resp.io_class, "best-effort-0");
    assert!((resp.plasticity_gate - 0.8).abs() < 0.01);
    assert!(!resp.controlling_cognitive);
    assert_eq!(resp.cpu_epp, "performance");
    assert_eq!(resp.cpu_boost, BoostState::Enabled);
    assert!(resp.description.is_empty());

    // Scenario 2: thermally capped, turbo disabled, EPP unavailable
    // (empty string) — the acpi-cpufreq case. Both must round-trip.
    let capped = BodyControlState {
        cpu_min_freq_khz: 1_000_000,
        cpu_max_freq_khz: 1_400_000,
        cpu_governor: "powersave".to_string(),
        thermally_capped: true,
        thermal_cap_temp_c: 91.5,
        daemon_nice: 3,
        cognitive_nice: 10,
        io_class: "idle".to_string(),
        plasticity_gate: 0.05,
        controlling_cognitive: false,
        cpu_epp: String::new(),
        cpu_boost: BoostState::Disabled,
        description: String::new(),
    };
    publish_control_state(&capped);
    let mut client2 = sys.client();
    let resp2 = client2.get_body_control().expect("get_body_control");
    assert!(resp2.thermally_capped);
    assert!((resp2.thermal_cap_temp_c - 91.5).abs() < 0.01);
    assert_eq!(resp2.cpu_governor, "powersave");
    assert_eq!(resp2.cognitive_nice, 10);
    assert_eq!(resp2.io_class, "idle");
    assert!(resp2.cpu_epp.is_empty());
    assert_eq!(resp2.cpu_boost, BoostState::Disabled);
    assert!(resp2.description.is_empty());
}

#[test]
fn test_ipc_shutdown_waits_for_active_handler() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let state_path = temp_path("active_handler_state", "bin");
    let stm_path = temp_path("active_handler_stm", "bin");
    let ltm_base = temp_base("active_handler_ltm");
    let socket_path = temp_path("active_handler_sock", "sock");

    let mmap = Arc::new(MmapState::create(&state_path, 1, 1000).expect("create state"));
    let stm = Arc::new(RingBuffer::create(&stm_path, 16).expect("create stm"));
    let ltm = Arc::new(std::sync::Mutex::new(
        LtmStore::create(&ltm_base, 64).expect("create ltm"),
    ));

    let server = IpcServer::new(&socket_path);
    let shutdown_flag = server.shutdown_flag();
    let handler_started = Arc::new(AtomicBool::new(false));
    let release_handler = Arc::new(AtomicBool::new(false));
    let handler_finished = Arc::new(AtomicBool::new(false));

    let server_thread = {
        let handler_started = handler_started.clone();
        let release_handler = release_handler.clone();
        let handler_finished = handler_finished.clone();
        std::thread::spawn(move || {
            server.run(mmap, stm, ltm, move |_, _, _, _, _| {
                handler_started.store(true, Ordering::Relaxed);
                while !release_handler.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                handler_finished.store(true, Ordering::Relaxed);
                vec![1]
            });
        })
    };

    let request_thread = std::thread::spawn({
        let socket_path = socket_path.clone();
        move || {
            // The server binds the socket inside its thread — retry
            // briefly rather than racing the bind.
            let mut client = None;
            for _ in 0..200 {
                match IpcClient::connect(&socket_path) {
                    Ok(c) => {
                        client = Some(c);
                        break;
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(10)),
                }
            }
            let mut client = client.expect("connect");
            client.request(250, &[]).expect("request")
        }
    });

    for _ in 0..100 {
        if handler_started.load(Ordering::Relaxed) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(handler_started.load(Ordering::Relaxed));

    shutdown_flag.store(true, Ordering::Relaxed);
    let server_joined = Arc::new(AtomicBool::new(false));
    let join_thread = std::thread::spawn({
        let server_joined = server_joined.clone();
        move || {
            server_thread.join().expect("server thread");
            server_joined.store(true, Ordering::Relaxed);
        }
    });

    std::thread::sleep(Duration::from_millis(50));
    assert!(!server_joined.load(Ordering::Relaxed));
    assert!(!handler_finished.load(Ordering::Relaxed));

    release_handler.store(true, Ordering::Relaxed);
    join_thread.join().expect("server join watcher");
    assert!(handler_finished.load(Ordering::Relaxed));
    assert_eq!(request_thread.join().expect("request thread"), vec![1]);

    let _ = std::fs::remove_file(&state_path);
    let _ = std::fs::remove_file(&stm_path);
    let _ = std::fs::remove_file(ltm_base.with_extension("idx"));
    let _ = std::fs::remove_file(ltm_base.with_extension("dat"));
    let _ = std::fs::remove_file(&socket_path);
}
