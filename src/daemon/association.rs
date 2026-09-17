//! Association engine — finds connections between memories and creates
//! associative links.
//!
//! # How human association works
//!
//! In the human brain, association is the process of linking related
//! memories. When you hear "salt lake," you think of Utah, water,
//! drought, data centers — because those memories are associatively
//! linked. This isn't similarity (Utah isn't "similar" to water) —
//! it's **co-activation**: memories that fire together wire together.
//!
//! During sleep, the brain runs **free association**: it reactivates
//! random memories and finds unexpected connections. This is the
//! neurobiological basis of creativity — "sleeping on it" works
//! because the brain builds new associative links offline.
//!
//! # How Genesis's association works
//!
//! Genesis uses SimHash fingerprints for associative matching. When a
//! new memory is consolidated, the association engine:
//!
//! 1. **Scans** the LTM index for memories with similar SimHashes
//! 2. **Scores** each match by Hamming distance (closer = stronger)
//! 3. **Records** the strongest associations as "links" — stored as
//!    episodes in LTM themselves (a meta-memory: "I noticed that
//!    memory X is related to memory Y")
//!
//! During dreaming (sleep phase), the engine runs **free association**:
//! 1. Picks a random memory from LTM
//! 2. Finds its associations
//! 3. Picks one of those associations at random
//! 4. Finds THAT memory's associations
//! 5. If it finds a connection between two memories that weren't
//!    previously linked, records it as a "dream insight"
//!
//! This is genuinely novel — no AI system does offline associative
//! discovery. It's the computational equivalent of dreaming.

use std::collections::HashSet;

use crate::store::{LtmStore, hamming_distance, simhash};

/// Maximum Hamming distance for two memories to be considered "associated."
/// Lower = stricter matching. 20 out of 64 bits is ~31% dissimilarity.
pub const ASSOCIATION_THRESHOLD: u32 = 20;

/// How many candidates to fetch from `find_associations` before filtering
/// out meta-memories. We over-fetch to compensate for filtered-out
/// `[association]` and `[dream-insight]` episodes so the engine still
/// has enough real source memories to build bridges between.
const ASSOCIATION_CANDIDATE_LIMIT: usize = 20;

/// Returns true if the episode text is a meta-memory (an association
/// bridge or dream insight), not a real source episode.
///
/// The association engine must not try to build bridges *from*
/// meta-memories — that causes tail-chasing where each pass associates
/// the previous pass's bridges with each other instead of finding new
/// connections between real experiences.
fn is_meta_memory(text: &str) -> bool {
    text.starts_with("[association]") || text.starts_with("[dream-insight]")
}

/// The result of an association pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct AssociationResult {
    /// Number of new associations discovered.
    pub new_associations: u32,
    /// Number of memories scanned.
    pub scanned: u32,
    /// Number of associations that already existed (duplicate).
    pub duplicates: u32,
}

/// The result of a dreaming pass.
#[derive(Clone, Debug, Default)]
pub struct DreamResult {
    /// Number of dream insights (novel connections discovered).
    pub insights: u32,
    /// The chain of memories traversed during dreaming.
    pub chain: Vec<u64>,
    /// Number of hops in the longest chain.
    pub max_depth: u32,
}

/// The association engine.
pub struct AssociationEngine;

impl AssociationEngine {
    /// Find associations for a specific episode.
    ///
    /// Returns a list of (episode_id, hamming_distance) pairs for
    /// memories whose SimHash is within `ASSOCIATION_THRESHOLD` bits
    /// of the query episode's hash.
    pub fn find_associations(ltm: &LtmStore, episode_hash: u64, limit: usize) -> Vec<(u64, u32)> {
        let mut results: Vec<(u64, u32)> = Vec::new();

        // Iterate only active entries via in-memory index — O(active)
        // instead of O(capacity) scanning all 65K slots.
        for (_id, entry) in ltm.active_entries() {
            let dist = hamming_distance(episode_hash, entry.association_hash);
            if dist <= ASSOCIATION_THRESHOLD {
                results.push((entry.episode_id, dist));
            }
        }

        results.sort_by_key(|(_, dist)| *dist);
        results.truncate(limit);
        results
    }

    /// Check if a hash close to `target_hash` exists among the most
    /// recent episodes. Only scans the last `scan_count` entries instead
    /// of the full LTM, making it O(scan_count) instead of O(active).
    ///
    /// This is used for duplicate detection during association: a
    /// duplicate meta-memory would have been stored recently, so we
    /// only need to check the tail of the index.
    fn hash_exists_recent(
        ltm: &LtmStore,
        target_hash: u64,
        threshold: u32,
        scan_count: usize,
    ) -> bool {
        let ids = ltm.active_episode_ids();
        let start = ids.len().saturating_sub(scan_count);
        for &id in &ids[start..] {
            if let Some(hash) = ltm.episode_hash(id)
                && hamming_distance(target_hash, hash) < threshold
            {
                return true;
            }
        }
        false
    }

    /// Run an association pass on recently-stored episodes.
    ///
    /// For each episode, finds its associations and records novel
    /// connections as meta-memories in LTM.
    ///
    /// # Parameters
    /// - `ltm`: the long-term memory store
    /// - `recent_ids`: episode IDs to find associations for
    /// - `now_ms`: current timestamp
    pub fn associate(ltm: &mut LtmStore, recent_ids: &[u64], now_ms: u64) -> AssociationResult {
        let mut result = AssociationResult::default();

        for &episode_id in recent_ids {
            result.scanned += 1;

            // Retrieve the episode to get its hash and text
            let episode = match ltm.retrieve(episode_id) {
                Ok(ep) => ep,
                Err(_) => continue,
            };

            // Don't build bridges from meta-memories — only from real
            // source episodes. Without this, the engine chases its own
            // tail: each pass associates the previous pass's
            // "[association] ..." bridges with each other instead of
            // finding new connections between real experiences.
            if is_meta_memory(&episode.text) {
                continue;
            }

            // Find associations. We over-fetch (ASSOCIATION_CANDIDATE_LIMIT)
            // to compensate for meta-memories that will be filtered out
            // below, so we still have enough real candidates to build
            // bridges between.
            let associations =
                Self::find_associations(ltm, episode.association_hash, ASSOCIATION_CANDIDATE_LIMIT);

            for (assoc_id, dist) in associations {
                if assoc_id == episode_id {
                    continue; // Don't associate with self
                }

                // Don't build bridges to meta-memories — only to real
                // source episodes. This prevents meta-memory-to-meta-memory
                // bridges from diluting the association graph.
                let assoc_episode = match ltm.retrieve(assoc_id) {
                    Ok(ep) => ep,
                    Err(_) => continue,
                };
                if is_meta_memory(&assoc_episode.text) {
                    continue;
                }

                // Check if this association already exists by searching
                // for a meta-memory about this pair.
                //
                // We embed the text of both endpoint episodes directly
                // in the association text (tab-delimited) so the
                // insight is self-contained. Without this, the Python
                // side would need to retrieve the episodes by ID later
                // — but sleep compression may have deleted them by
                // then, leaving stale references.
                let ep_a_text = truncate_text(&episode.text, 200);
                let ep_b_text = truncate_text(&assoc_episode.text, 200);
                let meta_text = format!(
                    "[association] episode {} ↔ episode {} (distance: {})\t{}\t{}",
                    episode_id, assoc_id, dist, ep_a_text, ep_b_text
                );
                let meta_hash = simhash(&meta_text);

                // Check if we already have this meta-memory.
                // Instead of scanning all 259K+ entries via
                // find_associations (O(n) per check, up to 100 checks
                // per pass), only scan the most recent episodes — a
                // duplicate would be among them since associations are
                // stored sequentially in LTM.
                if Self::hash_exists_recent(ltm, meta_hash, 5, 200) {
                    result.duplicates += 1;
                    continue;
                }

                // Store the association as a meta-memory
                let neutral_tag = [0.5f32; 12];
                let compact_tag = [0.5, 0.0, 0.2, 0.5];
                if ltm
                    .store_meta(
                        now_ms,
                        0.3 + (1.0 - dist as f32 / ASSOCIATION_THRESHOLD as f32) * 0.4,
                        neutral_tag,
                        compact_tag,
                        2, // Internal event
                        0, // Subcognitive module
                        &meta_text,
                    )
                    .is_ok()
                {
                    result.new_associations += 1;
                }
            }
        }

        result
    }

    /// Run a dreaming pass — free association through the memory graph.
    ///
    /// This is what happens when Genesis sleeps. The engine:
    /// 1. Picks a random memory
    /// 2. Finds its associations
    /// 3. Hops to a random associated memory
    /// 4. Repeats, building a chain
    /// 5. If it discovers a connection between two memories that
    ///    weren't directly linked before, records it as a "dream insight"
    ///
    /// # Parameters
    /// - `ltm`: the long-term memory store
    /// - `now_ms`: current timestamp
    /// - `max_hops`: maximum chain length
    pub fn dream(ltm: &mut LtmStore, now_ms: u64, max_hops: u32) -> DreamResult {
        let mut result = DreamResult::default();
        let mut chain: Vec<u64> = Vec::new();
        let mut visited: HashSet<u64> = HashSet::new();

        // Seed a xorshift PRNG from the current time. This gives each
        // dream pass a different trajectory — essential for genuine
        // free association. The previous approach used
        // `(now_ms + chain.len()) % len`, which was nearly deterministic
        // within a single pass (now_ms is constant, only chain.len()
        // varies), producing the same walk every time.
        let mut rng = XorShift64::from_seed(now_ms);

        // Pick a random starting memory
        let start_id = match Self::pick_random_episode(ltm, &mut rng) {
            Some(id) => id,
            None => return result, // LTM is empty
        };

        chain.push(start_id);
        visited.insert(start_id);
        let mut current_hash = match get_episode_hash(ltm, start_id) {
            Some(h) => h,
            None => return result,
        };

        for _hop in 0..max_hops {
            // Find associations for the current memory
            let associations = Self::find_associations(ltm, current_hash, 5);

            if associations.is_empty() {
                break; // Dead end
            }

            // Pick a random association using the PRNG
            let pick_idx = rng.next_usize() % associations.len();

            let (next_id, _dist) = associations[pick_idx];

            // Skip if already in chain (avoid loops) — O(1) HashSet check
            if visited.contains(&next_id) {
                break;
            }

            chain.push(next_id);
            visited.insert(next_id);

            // Check for novel insight: is the start memory associated
            // with the current memory through an indirect path?
            if chain.len() >= 3 {
                let start = chain[0];
                let Some(&end) = chain.last() else {
                    continue;
                };

                // Record a dream insight if the start and end are
                // more distant than the association threshold but
                // connected through the chain
                let start_hash = match get_episode_hash(ltm, start) {
                    Some(h) => h,
                    None => continue,
                };
                let end_hash = match get_episode_hash(ltm, end) {
                    Some(h) => h,
                    None => continue,
                };

                let direct_dist = hamming_distance(start_hash, end_hash);

                if direct_dist > ASSOCIATION_THRESHOLD {
                    // This is a novel insight — two memories that
                    // aren't directly similar but are connected through
                    // a chain of associations.
                    //
                    // We embed the text of both endpoint episodes
                    // directly in the insight text (tab-delimited)
                    // so the insight is self-contained. Without this,
                    // the Python side would need to retrieve the
                    // episodes by ID later — but sleep compression
                    // may have deleted them by then, leaving stale
                    // references that produce "episode not found" errors.
                    let start_text = ltm
                        .retrieve(start)
                        .map(|ep| truncate_text(&ep.text, 200))
                        .unwrap_or_default();
                    let end_text = ltm
                        .retrieve(end)
                        .map(|ep| truncate_text(&ep.text, 200))
                        .unwrap_or_default();
                    let insight_text = format!(
                        "[dream-insight] episode {} connects to episode {} through {} hops (direct distance: {})\t{}\t{}",
                        start,
                        end,
                        chain.len() - 1,
                        direct_dist,
                        start_text,
                        end_text
                    );

                    let neutral_tag = [0.5f32; 12];
                    let compact_tag = [0.6, 0.2, 0.1, 0.6]; // mildly positive, curious
                    if ltm
                        .store_meta(
                            now_ms,
                            0.7, // Dream insights are high-salience
                            neutral_tag,
                            compact_tag,
                            2, // Internal
                            9, // Dreaming module
                            &insight_text,
                        )
                        .is_ok()
                    {
                        result.insights += 1;
                    }
                }
            }

            // Move to the next memory
            current_hash = match get_episode_hash(ltm, next_id) {
                Some(h) => h,
                None => break,
            };
        }

        result.max_depth = chain.len() as u32;
        result.chain = chain;
        result
    }

    /// Pick a random episode from the LTM store.
    fn pick_random_episode(ltm: &LtmStore, rng: &mut XorShift64) -> Option<u64> {
        // O(1) via in-memory active_ids list — no scan needed
        let ids = ltm.active_episode_ids();
        if ids.is_empty() {
            return None;
        }
        Some(ids[rng.next_usize() % ids.len()])
    }
}

/// Get the SimHash of an episode from the LTM index.
/// O(1) via in-memory episode ID → slot index.
fn get_episode_hash(ltm: &LtmStore, episode_id: u64) -> Option<u64> {
    ltm.episode_hash(episode_id)
}

/// Truncate text to a maximum length, adding an ellipsis if truncated.
/// Used when embedding episode text in dream insights and association
/// traces to keep meta-memory entries compact.
///
/// Tabs are replaced with spaces so the tab delimiter between the
/// structural header and embedded texts remains unambiguous — without
/// this, an embedded text that is itself an association trace (which
/// contains its own tab delimiters) would break the Python parser.
fn truncate_text(text: &str, max_len: usize) -> String {
    let sanitized: String = text.replace('\t', " ");
    if sanitized.len() <= max_len {
        return sanitized;
    }
    // Truncate at a character boundary to avoid splitting UTF-8.
    let truncated = sanitized
        .char_indices()
        .take(max_len)
        .last()
        .map(|(i, _)| i)
        .unwrap_or(max_len);
    let mut result = sanitized[..truncated].to_string();
    result.push('…');
    result
}

// ─────────────────────────────────────────────────────────────────
//  XorShift64 — a minimal, deterministic PRNG for dreaming
// ─────────────────────────────────────────────────────────────────

/// A xorshift64 PRNG.
///
/// Used by the dreaming engine to produce genuine per-hop randomness.
/// This is not cryptographically secure — it doesn't need to be. It
/// just needs to produce a different trajectory each dream pass, which
/// the previous `(now_ms + chain.len()) % len` approach failed to do
/// because `now_ms` is constant within a single pass.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create from a seed. A zero seed would produce only zeros, so
    /// we fall back to a fixed nonzero constant in that case.
    fn from_seed(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEADBEEFCAFEBABE } else { seed },
        }
    }

    /// Next raw u64.
    fn next_u64(&mut self) -> u64 {
        // Marsaglia xorshift64: shift, xor, shift, xor, shift, xor
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Next usize (platform-width).
    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }
}
