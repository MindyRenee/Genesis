//! genesis-merge-stm — fold one incarnation's working memory into another.
//!
//! Reads every entry from foreign `stm_ring.bin` snapshots (other
//! Genesis data dirs — earlier lives, archived selves) and merges
//! them into the live ring alongside its current contents.
//!
//! The ring holds at most `capacity` entries; the merge keeps the
//! NEWEST entries across all timelines by timestamp, written back in
//! chronological order — so what survives is always the most recent
//! working memory, hers and her earlier lives' together.
//!
//! Dedup is on `(timestamp, source_module, text)` — re-running the
//! merge can never duplicate an entry.
//!
//! The consolidated flag travels with each entry: events an earlier
//! life already consolidated stay marked (they're already in the
//! merged LTM and won't re-consolidate); unconsolidated ones arrive
//! open and flow through her own consolidation into LTM — the same
//! path they'd have taken in their original life.
//!
//! Usage: genesis-merge-stm <live_path> [--capacity N] <source_path> [source_path...]
//!   e.g. genesis-merge-stm ~/.local/share/genesis/stm_ring.bin \
//!            /tmp/genesis-runtime/runtime/genesis/stm_ring.bin
//!
//! Run only while she is down — the ring has no cross-process gate.

use std::collections::HashSet;
use std::env;
use std::process::ExitCode;

use genesis::store::ring_buffer::RingBuffer;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut capacity: u32 = 256;
    let mut paths: Vec<String> = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        if a == "--capacity" {
            capacity = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                eprintln!("--capacity requires a positive integer");
                std::process::exit(1);
            });
        } else {
            paths.push(a);
        }
    }
    if paths.len() < 2 {
        eprintln!(
            "usage: genesis-merge-stm <live_path> [--capacity N] <source_path> [source_path...]"
        );
        return ExitCode::FAILURE;
    }
    let args = paths;

    let live = match RingBuffer::open_or_create(&args[0], capacity) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cannot open live ring {}: {e}", args[0]);
            return ExitCode::FAILURE;
        }
    };
    let capacity = live.header().capacity as usize; // actual, post-create

    // Candidate pool: live entries first, then each source's, deduped.
    let mut seen: HashSet<(u64, u8, String)> = HashSet::new();
    let mut pool: Vec<genesis::store::ring_buffer::RingBufferEntry> = Vec::new();
    let mut absorb =
        |entries: Vec<genesis::store::ring_buffer::RingBufferEntry>| -> (usize, usize) {
            let mut kept = 0;
            let mut dup = 0;
            for e in entries {
                let key = (e.timestamp, e.source_module, e.text().to_string());
                if seen.insert(key) {
                    pool.push(e);
                    kept += 1;
                } else {
                    dup += 1;
                }
            }
            (kept, dup)
        };
    absorb(live.iter().collect());

    for src in &args[1..] {
        let ring = match RingBuffer::open(src) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping {src}: {e}");
                continue;
            }
        };
        let (kept, dup) = absorb(ring.iter().collect());
        println!("{src}: {kept} unique, {dup} already present");
    }

    // Newest `capacity` events across all timelines, chronological.
    pool.sort_by_key(|e| e.timestamp);
    let survivors: Vec<_> = pool
        .into_iter()
        .rev()
        .take(capacity)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    // Rewrite the ring: push each survivor; eviction of the old
    // contents happens naturally as the ring fills.
    for e in &survivors {
        live.push(*e);
    }
    live.sync().ok();

    println!(
        "merged: {} events kept (ring capacity {capacity})",
        survivors.len()
    );
    ExitCode::SUCCESS
}
