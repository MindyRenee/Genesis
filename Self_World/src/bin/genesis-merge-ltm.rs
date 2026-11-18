//! genesis-merge-ltm — fold one incarnation's memories into another.
//!
//! Reads every active episode from a foreign `ltm_store` (a snapshot
//! of another Genesis data dir — an earlier life, an archived self)
//! and stores it into the live store. The store assigns fresh
//! sequential episode IDs on insert; everything else travels
//! unchanged: original timestamp, salience, both emotional tags,
//! event type, source module, and the text payload itself.
//!
//! Dedup is on `(timestamp, association_hash)` — the same moment with
//! the same SimHash fingerprint is the same memory, so re-running the
//! merge or importing a snapshot that overlaps the live store can
//! never duplicate an episode.
//!
//! Deleted and archived source episodes are skipped: what an earlier
//! life chose to shelve does not resurface here.
//!
//! Usage: genesis-merge-ltm <live_base> <source_base> [source_base...]
//!   e.g. genesis-merge-ltm ~/.local/share/genesis/ltm_store \
//!            /tmp/genesis-runtime/runtime/genesis/ltm_store
//!
//! Run only while she is down — the live store has no cross-process
//! gate, and merging under a running daemon would interleave writes.

use std::collections::HashSet;
use std::env;
use std::path::Path;
use std::process::ExitCode;

use genesis::store::ltm::LtmStore;

const PAGE: usize = 512;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <live_base> <source_base> [source_base...]",
            args[0]
        );
        eprintln!("       base = ltm_store path without .bundles/.meta/.dat");
        return ExitCode::from(2);
    }

    let live_path = &args[1];
    let mut live = match LtmStore::open(live_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot open live store {live_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Every live (timestamp, hash) pair — the "already remembered" set.
    // Rebuilt per source would redo O(n) work; built once, grown as we
    // import so the second snapshot also dedups against the first.
    let mut seen: HashSet<(u64, u64)> = HashSet::with_capacity(live.count() as usize);
    for (_id, entry) in live.active_entries() {
        seen.insert((entry.timestamp, entry.association_hash));
    }

    let mut failures = false;
    for src in &args[2..] {
        if !Path::new(&format!("{src}.meta")).exists() {
            eprintln!("{src}: no ltm_store.meta — skipping");
            failures = true;
            continue;
        }
        let mut source = match LtmStore::open(src) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cannot open source store {src}: {e}");
                failures = true;
                continue;
            }
        };

        let mut imported = 0u64;
        let mut skipped = 0u64;
        let mut failed = 0u64;
        let mut offset = 0usize;
        loop {
            let page = source.search_episodes(PAGE, offset);
            if page.is_empty() {
                break;
            }
            offset += page.len();
            for ep in page {
                let key = (ep.timestamp, ep.association_hash);
                if !seen.insert(key) {
                    skipped += 1;
                    continue;
                }
                match live.store(
                    ep.timestamp,
                    ep.salience,
                    ep.full_emotional_tag,
                    ep.emotional_tag,
                    ep.event_type,
                    ep.source_module,
                    &ep.text,
                ) {
                    Ok(_) => imported += 1,
                    Err(e) => {
                        // Re-claim the slot: the episode never landed,
                        // so a later identical one should still try.
                        seen.remove(&key);
                        failed += 1;
                        eprintln!("  episode {} failed: {e}", ep.episode_id);
                    }
                }
            }
        }
        println!(
            "{src}: {imported} episode(s) imported, {skipped} already remembered, {failed} failed"
        );
    }

    if failures {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
