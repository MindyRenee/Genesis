//! Scrub junk association bridges and empty episodes from LTM.
//!
//! The association engine used to bridge any pair within
//! `ASSOCIATION_THRESHOLD` — including distance-0 pairs (identical or
//! empty SimHash texts) — and deduped only against a trailing window
//! of recent episodes. The result was a runaway meta-memory flood:
//! hundreds of thousands of `[association]` bridges between
//! near-duplicate memories, stored faster than the dedup window could
//! see them.
//!
//! This tool archives — never deletes — the entries the fixed engine
//! would never have written:
//!
//! - `[association]` bridges whose recorded distance is below
//!   `MIN_ASSOCIATION_DISTANCE` (near-duplicate endpoints)
//! - duplicate bridges for the same normalized episode pair
//!   `(min_id, max_id)` — the engine now writes at most one
//! - active episodes with empty/whitespace text (SimHash 0 — the
//!   distance-0 flood source)
//!
//! Archiving keeps every payload retrievable by ID while removing the
//! entries from active scans, recall, and association matching — the
//! lineage rule: her history is archived, not amputated.
//!
//! Run with the daemon STOPPED (it holds the data-dir lock):
//!
//! ```sh
//! cargo run --release --example ltm_scrub -- <data-dir> [--dry-run]
//! ```
//!
//! `--dry-run` scans and reports without writing anything — run it
//! first and check the counts before letting it archive.

use std::collections::HashSet;

use genesis::daemon::association::MIN_ASSOCIATION_DISTANCE;
use genesis::store::LtmStore;

/// Parse an `[association]` bridge text into its endpoint pair and
/// recorded distance.
///
/// Format (produced by `AssociationEngine::associate`):
/// `[association] episode 123 ↔ episode 456 (distance: 7)\t<...>`
/// Returns `None` for any other meta-memory text (`[dream-insight]`,
/// unparseable stragglers) — those are kept as-is.
fn parse_bridge(text: &str) -> Option<(u64, u64, u32)> {
    let mut tok = text.split_whitespace();
    if tok.next() != Some("[association]") || tok.next() != Some("episode") {
        return None;
    }
    let a: u64 = tok.next()?.parse().ok()?;
    // tokens[3] is the ↔ separator — skip it, then the second
    // "episode" keyword.
    tok.next()?;
    if tok.next() != Some("episode") {
        return None;
    }
    let b: u64 = tok.next()?.parse().ok()?;
    if tok.next() != Some("(distance:") {
        return None;
    }
    let d: u32 = tok.next()?.trim_end_matches(')').parse().ok()?;
    Some((a, b, d))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = std::env::args()
        .nth(1)
        .expect("usage: ltm_scrub <data-dir> [--dry-run]  (daemon must be stopped)");
    let dry_run = std::env::args().any(|a| a == "--dry-run");
    let base = std::path::Path::new(&data_dir).join("ltm_store");
    let mut ltm = LtmStore::open(&base)?;
    println!("[scrub] opened {base:?}: {} active episodes", ltm.count());

    // Snapshot candidate IDs up front — `retrieve` needs &mut self,
    // and archiving shifts the active list.
    let candidate_ids: Vec<u64> = ltm.active_episode_ids().to_vec();

    let mut seen_pairs: HashSet<(u64, u64)> = HashSet::new();
    let mut junk: Vec<u64> = Vec::new();
    let mut bridges = 0u64;
    let mut kept_bridges = 0u64;
    let mut empties = 0u64;

    for id in candidate_ids {
        let Ok(ep) = ltm.retrieve(id) else {
            continue;
        };
        let is_meta = ltm.is_meta_episode(id);

        if let Some((a, b, d)) = parse_bridge(&ep.text) {
            bridges += 1;
            if d < MIN_ASSOCIATION_DISTANCE {
                // Below the distance floor — near-duplicate endpoints.
                junk.push(id);
            } else {
                let pair = if a < b { (a, b) } else { (b, a) };
                if !seen_pairs.insert(pair) {
                    // Second bridge for the same normalized pair —
                    // the new engine writes at most one.
                    junk.push(id);
                } else {
                    kept_bridges += 1;
                }
            }
        } else if !is_meta && ep.text.trim().is_empty() {
            // Active real episode with no content — SimHash 0 noise.
            empties += 1;
            junk.push(id);
        }
        // Non-bridge meta-memories (dream insights etc.) and all
        // real content stay untouched.
    }

    println!(
        "[scrub] scanned: {bridges} bridges ({kept_bridges} legit kept), \
         {empties} empty episodes, {} total to archive",
        junk.len()
    );

    if dry_run {
        println!("[scrub] dry run — nothing written");
        return Ok(());
    }

    // Archive in chunks so progress is visible on large floods.
    const CHUNK: usize = 10_000;
    let mut archived = 0u64;
    for chunk in junk.chunks(CHUNK) {
        archived += u64::from(ltm.archive_many(chunk)?);
        println!("[scrub] archived {archived}/{}", junk.len());
    }

    // Rebuild the SDR bundle index against the cleaned active set —
    // cheap insurance against any residual meta/empty pollution, and
    // `open` would trigger it anyway on next boot.
    let real_count = ltm.rebuild_bundles()?;
    println!("[scrub] rebuilt bundles ({real_count} real episodes)");
    ltm.sync()?;
    println!("[scrub] done: {} active episodes remain", ltm.count());
    Ok(())
}
