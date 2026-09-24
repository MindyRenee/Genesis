//! Sparse Distributed Representations (SDR) and LogHD bundle store.
//!
//! This module implements Genesis's associative memory indexing layer,
//! based on two ideas from neuroscience and hyperdimensional computing:
//!
//! # SDR Encoding
//!
//! Each episode is encoded as a 10,000-bit sparse binary vector with
//! ~2% sparsity (~200 active bits). The encoding combines:
//!
//! - **Content** (~170 bits): text tokens are hashed to random
//!   positions in a 9,800-bit content space. Similar texts produce
//!   overlapping bit patterns — the Jaccard similarity of two SDRs
//!   approximates the semantic similarity of the texts.
//!
//! - **Emotional context** (12 bits): each of the 12 neurochemical
//!   levels is thresholded (level > 0.5 → bit set) and mapped to a
//!   fixed position in a reserved range. Episodes with similar
//!   emotional states share emotional bits.
//!
//! - **Temporal context** (6 bits): the timestamp is hashed into 6
//!   positions, giving coarse temporal locality.
//!
//! # LogHD Bundle Store
//!
//! Instead of storing one SDR per episode (which grows linearly), we
//! use LogHD class-axis compression (Yun et al., 2025):
//!
//! - Each episode gets a unique k-ary code of length n
//!   (k=4, n=20 → capacity = 4^20 ≈ 1 trillion episodes).
//! - We store n "bundle" hypervectors, each D-dimensional. A bundle
//!   is a weighted superposition of all episode SDRs, weighted by
//!   their code symbols.
//! - At query time, we compute the query's activation vector
//!   (similarity to each bundle) and match it against the expected
//!   activation of each episode's code.
//!
//! This gives us a **fixed-size index** (n × D × 4 bytes = 800 KB)
//! regardless of the number of episodes, with O(n) similarity
//! computation per query plus O(N) decoding over episode codes.
//!
//! # Why not just use SimHash?
//!
//! The previous design used a 64-bit SimHash per episode. With 65K+
//! episodes, 64 bits cannot discriminate well — many unrelated
//! episodes have Hamming distance < 10, producing false matches.
//! A 10,000-bit SDR gives ~150x more discrimination, and the LogHD
//! bundle store keeps the index size fixed at 800 KB instead of
//! growing with the episode count.

use std::os::unix::io::AsRawFd;

use libc::{
    MAP_FAILED, MAP_SHARED, MS_SYNC, PROT_READ, PROT_WRITE, c_void, close, fdatasync, ftruncate,
    mmap, msync, munmap,
};

// ─────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────

/// SDR dimensionality — total number of bits.
pub const SDR_DIM: usize = 10_000;

/// Target number of active bits (~2% sparsity).
pub const SDR_TARGET_ACTIVE: usize = 200;

/// Content region: bits 0 .. CONTENT_BITS.
pub const CONTENT_BITS: usize = 9_800;

/// Temporal region: bits 9_800 .. 9_806 (6 bits).
pub const TEMPORAL_BASE: usize = 9_800;
/// Number of bits reserved for temporal context.
pub const TEMPORAL_BITS: usize = 6;

/// Emotional region: bits 9_806 .. 9_818 (12 bits, one per chemical).
pub const EMOTIONAL_BASE: usize = 9_806;
/// Number of bits reserved for emotional context (one per neurochemical).
pub const EMOTIONAL_BITS: usize = 12;

/// Number of random positions each token sets in the content space.
/// 5 positions per token × ~40 tokens ≈ 200 active bits (with
/// collisions, ~170 unique).
const BITS_PER_TOKEN: usize = 5;

// ─── LogHD parameters ───────────────────────────────────────────

/// Alphabet size for LogHD codes. k=4 means each code digit is in
/// {0, 1, 2, 3}.
pub const LOGHD_K: u32 = 4;

/// Number of bundle hypervectors. n=20 with k=4 gives capacity
/// 4^20 ≈ 1.1 trillion episodes.
pub const LOGHD_N: usize = 20;

/// Magic bytes for the bundle file: ASCII "LHDB".
pub const BUNDLE_MAGIC: [u8; 4] = *b"LHDB";
/// Schema version for the LogHD bundle file.
pub const BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Size of the bundle file header (bytes).
pub const BUNDLE_HEADER_SIZE: usize = 64;

/// Total bundle file size: header + n × D × sizeof(f32).
///
/// This is the **logical** size — the actual data occupies exactly
/// this many bytes. The on-disk file is page-aligned (see
/// [`bundle_mmap_size`]) so the entire `mmap` mapping is backed by
/// the file.
pub const BUNDLE_FILE_SIZE: usize = BUNDLE_HEADER_SIZE + LOGHD_N * SDR_DIM * 4;

/// Compute the page-aligned mmap size for the bundle store.
///
/// `ftruncate` and `mmap` use this size so the entire mapping is
/// backed by the file, eliminating the risk of `SIGBUS` when
/// accessing the tail of the last page. Data access is still bounded
/// by [`BUNDLE_FILE_SIZE`] (the logical size).
pub fn bundle_mmap_size() -> usize {
    crate::store::page_align_up(BUNDLE_FILE_SIZE)
}

// ─────────────────────────────────────────────────────────────────
//  SDR — sparse binary vector
// ─────────────────────────────────────────────────────────────────

/// A sparse distributed representation: a sorted list of active bit
/// positions in a 10,000-bit vector.
///
/// Stored as `Vec<u16>` (positions 0..=9999 fit in u16). Sorted for
/// fast intersection/union and similarity computation.
#[derive(Clone, Debug)]
pub struct Sdr {
    /// Sorted active bit positions.
    pub active: Vec<u16>,
}

impl Sdr {
    /// Create an empty SDR (no active bits).
    pub fn empty() -> Self {
        Self { active: Vec::new() }
    }

    /// Number of active bits.
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// True if no bits are active.
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Jaccard similarity (intersection over union) with another SDR.
    /// Both must be sorted. Returns a value in [0, 1].
    pub fn jaccard(&self, other: &Sdr) -> f32 {
        if self.active.is_empty() && other.active.is_empty() {
            return 1.0;
        }
        let union = self.active.len() + other.active.len()
            - intersection_count(&self.active, &other.active);
        if union == 0 {
            return 0.0;
        }
        intersection_count(&self.active, &other.active) as f32 / union as f32
    }

    /// Overlap count (number of shared active bits).
    pub fn overlap(&self, other: &Sdr) -> usize {
        intersection_count(&self.active, &other.active)
    }
}

/// Count the number of elements in the intersection of two sorted slices.
fn intersection_count(a: &[u16], b: &[u16]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut count = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

// ─────────────────────────────────────────────────────────────────
//  SDR encoding
// ─────────────────────────────────────────────────────────────────

/// Encode text into an SDR content vector.
///
/// Each token (unigram and bigram) is hashed to `BITS_PER_TOKEN`
/// random positions in the content space. Positions are accumulated
/// in a count array, then the top positions by count are selected to
/// maintain the target sparsity.
pub fn encode_content(text: &str) -> Sdr {
    let mut counts = vec![0u16; CONTENT_BITS];

    // Tokenise: split on non-alphanumeric
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();

    // Unigrams
    for word in &words {
        let h = fnv1a_lower(word.as_bytes());
        for i in 0..BITS_PER_TOKEN {
            let pos = wrangle(h, i) % CONTENT_BITS as u64;
            counts[pos as usize] = counts[pos as usize].saturating_add(1);
        }
    }

    // Bigrams
    for i in 0..words.len().saturating_sub(1) {
        let h = fnv1a_lower_bigram(words[i].as_bytes(), words[i + 1].as_bytes());
        for j in 0..BITS_PER_TOKEN {
            let pos = wrangle(h, j) % CONTENT_BITS as u64;
            counts[pos as usize] = counts[pos as usize].saturating_add(1);
        }
    }

    // Select top positions by count, up to ~170 content bits
    let target_content_bits = SDR_TARGET_ACTIVE - TEMPORAL_BITS - EMOTIONAL_BITS;
    let mut positions: Vec<(usize, u16)> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, &c)| (i, c))
        .collect();
    // Sort by count descending, then by position for determinism
    positions.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    positions.truncate(target_content_bits);

    let mut active: Vec<u16> = positions.into_iter().map(|(p, _)| p as u16).collect();
    active.sort_unstable();
    Sdr { active }
}

/// Encode emotional context into bits.
///
/// Each of the 12 neurochemical levels is thresholded at 0.5.
/// If level > 0.5, the corresponding bit in the emotional region
/// is activated.
pub fn encode_emotional(tag: &[f32; 12]) -> Vec<u16> {
    let mut bits = Vec::with_capacity(12);
    for (i, &level) in tag.iter().enumerate() {
        if level > 0.5 {
            bits.push((EMOTIONAL_BASE + i) as u16);
        }
    }
    bits
}

/// Encode temporal context into bits.
///
/// The timestamp (ms) is hashed into `TEMPORAL_BITS` positions in
/// the temporal region. This gives coarse temporal locality —
/// episodes from similar times share some temporal bits.
pub fn encode_temporal(timestamp_ms: u64) -> Vec<u16> {
    let mut bits = Vec::with_capacity(TEMPORAL_BITS);
    let h = fnv1a(&timestamp_ms.to_le_bytes());
    for i in 0..TEMPORAL_BITS {
        let pos = (TEMPORAL_BASE + (wrangle(h, i) as usize % TEMPORAL_BITS)) as u16;
        bits.push(pos);
    }
    bits.sort_unstable();
    bits.dedup();
    bits
}

/// Encode a complete episode as an SDR.
///
/// Combines content, emotional, and temporal bits into a single
/// sorted sparse vector.
pub fn encode_episode(text: &str, emotional_tag: &[f32; 12], timestamp_ms: u64) -> Sdr {
    let content = encode_content(text);
    let emotional = encode_emotional(emotional_tag);
    let temporal = encode_temporal(timestamp_ms);

    // Merge sorted vectors
    let mut active = Vec::with_capacity(content.active.len() + emotional.len() + temporal.len());
    active.extend(content.active);
    active.extend(emotional);
    active.extend(temporal);
    active.sort_unstable();
    active.dedup();
    Sdr { active }
}

// ─────────────────────────────────────────────────────────────────
//  LogHD code assignment
// ─────────────────────────────────────────────────────────────────

/// Convert an episode ID to a packed base-k code.
///
/// The code is `LOGHD_N` digits in base `LOGHD_K`, packed 2 digits
/// per byte into `LOGHD_N / 2 = 10` bytes.
pub fn episode_id_to_code(id: u64) -> [u8; LOGHD_N / 2] {
    let mut code = [0u8; LOGHD_N / 2];
    let mut remaining = id;
    for i in (0..LOGHD_N).rev() {
        let digit = (remaining % LOGHD_K as u64) as u8;
        let byte_idx = i / 2;
        if i % 2 == 0 {
            code[byte_idx] |= digit << 4;
        } else {
            code[byte_idx] |= digit;
        }
        remaining /= LOGHD_K as u64;
    }
    code
}

/// Extract digit `i` from a packed code.
pub fn code_digit(code: &[u8; LOGHD_N / 2], i: usize) -> u8 {
    let byte_idx = i / 2;
    if i % 2 == 0 {
        (code[byte_idx] >> 4) & 0x0F
    } else {
        code[byte_idx] & 0x0F
    }
}

/// Compute the expected activation for a code.
///
/// `g(s) = s / (k-1)` maps code digits {0,1,2,3} → {0.0, 0.33, 0.67, 1.0}.
/// The expected activation of episode `e` against bundle `j` is
/// `g(code[e][j])`.
pub fn code_to_expected_activation(code: &[u8; LOGHD_N / 2]) -> [f32; LOGHD_N] {
    let mut activation = [0.0f32; LOGHD_N];
    for (i, slot) in activation.iter_mut().enumerate() {
        *slot = code_digit(code, i) as f32 / (LOGHD_K - 1) as f32;
    }
    activation
}

// ─────────────────────────────────────────────────────────────────
//  LogHD bundle store
// ─────────────────────────────────────────────────────────────────

/// Bundle file header (64 bytes).
///
/// ```text
/// offset  field              type    notes
/// ------  -----              ----    -----
///   0     magic              [u8;4] b"LHDB"
///   4     version            u32
///   8     dim                u32    SDR dimensionality (10,000)
///  12     n                  u32    number of bundles (20)
///  16     k                  u32    alphabet size (4)
///  20     episode_count      u32    episodes stored
///  24     next_episode_id    u64    monotonic ID counter
///  32     _reserved          [u8;32] future expansion
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BundleHeader {
    /// Magic bytes identifying the bundle file (`BUNDLE_MAGIC`).
    pub magic: [u8; 4],
    /// Schema version of the bundle file.
    pub version: u32,
    /// SDR dimensionality (10,000).
    pub dim: u32,
    /// Number of bundles (20).
    pub n: u32,
    /// LogHD alphabet size (4).
    pub k: u32,
    /// Number of episodes stored.
    pub episode_count: u32,
    /// Next monotonic episode ID to assign.
    pub next_episode_id: u64,
    /// Reserved bytes for future expansion.
    pub _reserved: [u8; 32],
}

impl BundleHeader {
    /// Create a fresh bundle header with default SDR parameters.
    pub fn new() -> Self {
        Self {
            magic: BUNDLE_MAGIC,
            version: BUNDLE_SCHEMA_VERSION,
            dim: SDR_DIM as u32,
            n: LOGHD_N as u32,
            k: LOGHD_K,
            episode_count: 0,
            next_episode_id: 1,
            _reserved: [0; 32],
        }
    }

    /// Returns `true` if the magic bytes match `BUNDLE_MAGIC`.
    pub fn verify_magic(&self) -> bool {
        self.magic == BUNDLE_MAGIC
    }
}

impl Default for BundleHeader {
    fn default() -> Self {
        Self::new()
    }
}

const _: () = {
    assert!(core::mem::size_of::<BundleHeader>() == BUNDLE_HEADER_SIZE);
};

/// The LogHD bundle store — a memory-mapped file containing n
/// dense f32 hypervectors.
///
/// Each bundle is a D-dimensional f32 vector (40 KB). With n=20
/// bundles, the total bundle data is 800 KB. This is fixed
/// regardless of the number of episodes.
///
/// Bundles are updated incrementally: when a new episode is stored,
/// each bundle `j` is updated by `M_j += g(code[j]) * SDR`, where
/// `g(s) = s / (k-1)` and SDR is the episode's sparse binary vector.
pub struct BundleStore {
    /// mmap'd region
    ptr: *mut u8,
    /// file descriptor
    fd: i32,
    /// total mmap size
    len: usize,
}

unsafe impl Send for BundleStore {}
unsafe impl Sync for BundleStore {}

impl BundleStore {
    /// Create a new bundle store file.
    pub fn create(path: impl AsRef<std::path::Path>) -> Result<Self, std::io::Error> {
        Self::create_file(path.as_ref(), false)
    }

    pub(crate) fn create_new(path: &std::path::Path) -> Result<Self, std::io::Error> {
        Self::create_file(path, true)
    }

    fn create_file(path: &std::path::Path, exclusive: bool) -> Result<Self, std::io::Error> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(!exclusive)
            .create_new(exclusive)
            .truncate(!exclusive)
            .open(path)?;
        let fd = file.as_raw_fd();

        let mmap_len = bundle_mmap_size();

        // Allocate the file with the page-aligned size so the entire
        // mmap mapping is backed by the file.
        // SAFETY: `fd` is a valid open file descriptor obtained from `as_raw_fd`
        // on the file handle above, and `mmap_len` is a positive page-aligned
        // constant.
        unsafe {
            if ftruncate(fd, mmap_len as i64) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        // Persist the file size metadata so a crash doesn't leave a
        // zero-length or truncated file. `msync` only flushes data
        // pages, not the inode's i_size.
        // SAFETY: `fd` is a valid open file descriptor.
        unsafe {
            if fdatasync(fd) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }

        let ptr = Self::do_mmap(fd, mmap_len)?;
        // Keep the file alive until the BundleStore is constructed.
        // `std::mem::forget(file)` would leak the fd if any of the
        // operations below return early with `Err`. Instead, we
        // transfer ownership to the BundleStore at the end. If an
        // error occurs before that, `file`'s `Drop` closes the fd,
        // and we explicitly `munmap` the mapping.

        // Write header
        let header = BundleHeader::new();
        // SAFETY: `ptr` points to a freshly mmap'd region of `mmap_len`
        // bytes (≥ `BUNDLE_HEADER_SIZE`), so there is enough space for a
        // `BundleHeader`. The region is `PROT_READ | PROT_WRITE` and not
        // concurrently accessed.
        unsafe {
            std::ptr::write_volatile(ptr as *mut BundleHeader, header);
        }

        // Zero the bundle data
        // SAFETY: `ptr.add(BUNDLE_HEADER_SIZE)` stays within the mmap'd region
        // (the region is `mmap_len` bytes and the header occupies the
        // first `BUNDLE_HEADER_SIZE`), and the byte count is the logical
        // data size (`BUNDLE_FILE_SIZE - BUNDLE_HEADER_SIZE` ≤ `mmap_len - BUNDLE_HEADER_SIZE`).
        // The region is writable and not concurrently accessed.
        unsafe {
            std::ptr::write_bytes(
                ptr.add(BUNDLE_HEADER_SIZE),
                0,
                BUNDLE_FILE_SIZE - BUNDLE_HEADER_SIZE,
            );
        }

        // Sync
        // SAFETY: `ptr` is a valid mmap'd address of length `mmap_len`
        // obtained from `do_mmap` above, and `MS_SYNC` is a valid flag.
        let rc = unsafe { msync(ptr as *mut c_void, mmap_len, MS_SYNC) };
        if rc != 0 {
            // Cleanup: munmap the mapping before returning the error.
            // `file` is still alive and will be dropped (closing the fd).
            unsafe { libc::munmap(ptr as *mut c_void, mmap_len) };
            return Err(std::io::Error::last_os_error());
        }

        // Transfer ownership: the fd is now managed by `file` (which
        // we forget here — the BundleStore's Drop will close it via
        // the raw fd). The mmap is managed by the BundleStore's Drop
        // via `munmap`.
        std::mem::forget(file);

        Ok(Self {
            ptr,
            fd,
            len: mmap_len,
        })
    }

    /// Open an existing bundle store file.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, std::io::Error> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "bundle file not found",
            ));
        }

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        let fd = file.as_raw_fd();
        let meta = file.metadata()?;
        let len = meta.len() as usize;

        if len < BUNDLE_HEADER_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bundle file too small",
            ));
        }
        if len < BUNDLE_FILE_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bundle file truncated: expected {BUNDLE_FILE_SIZE} bytes, found {len}"),
            ));
        }

        let ptr = Self::do_mmap(fd, len)?;
        // Keep `file` alive until the BundleStore is constructed.
        // If any verification below fails, `file`'s `Drop` closes
        // the fd, and we explicitly `munmap` the mapping. This is
        // safer than `std::mem::forget(file)` early, which would
        // leak the fd if verification fails.

        // Verify header
        // SAFETY: `ptr` is a valid mmap'd region of at least `BUNDLE_HEADER_SIZE`
        // bytes (checked above), properly aligned for `BundleHeader` (it is the
        // start of the page-aligned mmap), and readable (`PROT_READ`).
        let header = unsafe { &*(ptr as *const BundleHeader) };
        if !header.verify_magic() {
            // SAFETY: `ptr` is a valid mmap'd region of length `len`.
            // `file` is still alive and will close the fd on drop.
            unsafe { munmap(ptr as *mut c_void, len) };
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bundle file bad magic",
            ));
        }
        if header.version != BUNDLE_SCHEMA_VERSION {
            // SAFETY: `ptr` is a valid mmap'd region of length `len`.
            unsafe { munmap(ptr as *mut c_void, len) };
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bundle version mismatch: expected {}, found {}",
                    BUNDLE_SCHEMA_VERSION, header.version
                ),
            ));
        }

        // Validate dim/n/k against compile-time constants. A corrupted
        // or truncated header with a smaller n could cause bundle()/
        // bundle_mut() to compute out-of-bounds offsets and SIGBUS or
        // corrupt neighboring bundle data.
        if header.dim as usize != SDR_DIM || header.n as usize != LOGHD_N || header.k != LOGHD_K {
            // SAFETY: `ptr` is a valid mmap'd region of length `len`.
            unsafe { munmap(ptr as *mut c_void, len) };
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bundle header mismatch: expected dim={}, n={}, k={}, found dim={}, n={}, k={}",
                    SDR_DIM, LOGHD_N, LOGHD_K, header.dim, header.n, header.k
                ),
            ));
        }

        // Ensure the file is page-aligned so the entire mmap mapping is
        // backed by the file. Files created by older versions of this
        // code may have a non-page-aligned size. We extend the file to
        // the page-aligned mmap size and re-mmap if necessary.
        let needed_mmap_len = bundle_mmap_size();
        let (final_ptr, final_mmap_len) = if len < needed_mmap_len {
            // SAFETY: `ptr` is a valid mmap'd region of length `len`.
            // `file` is still alive and owns the fd.
            unsafe {
                munmap(ptr as *mut c_void, len);
            }
            // SAFETY: `fd` is a valid open file descriptor with write
            // permissions. ftruncate extends the file to
            // `needed_mmap_len`, zero-filling the new bytes.
            unsafe {
                if ftruncate(fd, needed_mmap_len as i64) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            // SAFETY: `fd` is a valid open file descriptor.
            unsafe {
                if fdatasync(fd) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            // Re-mmap with the page-aligned size.
            let new_ptr = Self::do_mmap(fd, needed_mmap_len)?;
            (new_ptr, needed_mmap_len)
        } else {
            // File is already large enough (page-aligned or larger).
            // Use the existing mapping.
            (ptr, len)
        };

        // Transfer ownership: the fd is now managed by the BundleStore's
        // Drop (via the raw fd). The mmap is managed by Drop via munmap.
        // `file` must be forgotten so its Drop doesn't close the fd
        // out from under the BundleStore.
        std::mem::forget(file);

        Ok(Self {
            ptr: final_ptr,
            fd,
            len: final_mmap_len,
        })
    }

    /// Open or create.
    ///
    /// Tries `create_new` first (which fails with `EEXIST` if the file
    /// already exists), then falls back to `open`. This avoids the
    /// TOCTOU race of checking `exists()` then calling `create()` —
    /// between the check and the create, another process could create
    /// the file, and `create` (with truncate) would destroy it.
    pub fn open_or_create(path: impl AsRef<std::path::Path>) -> Result<Self, std::io::Error> {
        match Self::create_new(path.as_ref()) {
            Ok(store) => Ok(store),
            Err(ref e) if e.raw_os_error() == Some(libc::EEXIST) => Self::open(path),
            Err(e) => Err(e),
        }
    }

    // ─── Accessors ──────────────────────────────────────────────

    fn header(&self) -> &BundleHeader {
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `BUNDLE_HEADER_SIZE` bytes, page-aligned and readable. The
        // `BundleStore` invariant guarantees the mmap stays valid for the
        // lifetime of `self`.
        unsafe { &*(self.ptr as *const BundleHeader) }
    }

    fn header_mut(&mut self) -> &mut BundleHeader {
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `BUNDLE_HEADER_SIZE` bytes, page-aligned and writable. The
        // `&mut self` borrow ensures no aliasing references exist.
        unsafe { &mut *(self.ptr as *mut BundleHeader) }
    }

    /// Get a reference to bundle `j` (D f32 values).
    fn bundle(&self, j: usize) -> &[f32] {
        let offset = BUNDLE_HEADER_SIZE + j * SDR_DIM * 4;
        // SAFETY: `offset + SDR_DIM * 4` is within the mmap'd region because
        // `j < LOGHD_N` and the region is `BUNDLE_FILE_SIZE` bytes. The
        // pointer is aligned (header size and stride are multiples of 4) and
        // the `&self` borrow prevents mutable aliasing.
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset) as *const f32, SDR_DIM) }
    }

    /// Get a mutable reference to bundle `j`.
    fn bundle_mut(&mut self, j: usize) -> &mut [f32] {
        let offset = BUNDLE_HEADER_SIZE + j * SDR_DIM * 4;
        // SAFETY: `offset + SDR_DIM * 4` is within the mmap'd region because
        // `j < LOGHD_N` and the region is `BUNDLE_FILE_SIZE` bytes. The
        // pointer is aligned (header size and stride are multiples of 4) and
        // the `&mut self` borrow ensures exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset) as *mut f32, SDR_DIM) }
    }

    // ─── Operations ─────────────────────────────────────────────

    /// Number of episodes stored.
    pub fn episode_count(&self) -> u32 {
        self.header().episode_count
    }

    /// Next episode ID.
    pub fn next_episode_id(&self) -> u64 {
        self.header().next_episode_id
    }

    pub(crate) fn ensure_next_episode_id(&mut self, next_id: u64) {
        let header = self.header_mut();
        header.next_episode_id = header.next_episode_id.max(next_id);
    }

    /// Update bundles with a new episode's SDR.
    ///
    /// For each bundle `j`: `M_j += g(code[j]) * SDR`
    /// where `g(s) = s / (k-1)`.
    pub fn add_episode(&mut self, code: &[u8; LOGHD_N / 2], sdr: &Sdr) {
        for j in 0..LOGHD_N {
            let digit = code_digit(code, j) as f32;
            let weight = digit / (LOGHD_K - 1) as f32;
            if weight == 0.0 {
                continue;
            }
            let bundle = self.bundle_mut(j);
            for &pos in &sdr.active {
                bundle[pos as usize] += weight;
            }
        }

        // Update header
        let h = self.header_mut();
        h.episode_count = h.episode_count.saturating_add(1);
        h.next_episode_id = h.next_episode_id.saturating_add(1);
    }

    /// Compute the activation vector of a query SDR against all bundles.
    ///
    /// `A[j] = δ(M_j, query)` where δ is the dot product of the
    /// query SDR (binary) with the bundle (f32), normalised by the
    /// number of active bits in the query.
    pub fn query_activation(&self, query: &Sdr) -> [f32; LOGHD_N] {
        let mut activation = [0.0f32; LOGHD_N];
        if query.active.is_empty() {
            return activation;
        }
        let norm = query.active.len() as f32;
        for (j, slot) in activation.iter_mut().enumerate() {
            let bundle = self.bundle(j);
            let mut sum = 0.0f32;
            for &pos in &query.active {
                sum += bundle[pos as usize];
            }
            *slot = sum / norm;
        }
        activation
    }

    /// Sync to disk.
    pub fn sync(&self) -> Result<(), std::io::Error> {
        // SAFETY: `self.ptr` is a valid mmap'd region of length `self.len`
        // (maintained by the `BundleStore` invariant), and `MS_SYNC` is a
        // valid flag.
        let rc = unsafe { msync(self.ptr as *mut c_void, self.len, MS_SYNC) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    // ─── Internal ───────────────────────────────────────────────

    fn do_mmap(fd: i32, len: usize) -> Result<*mut u8, std::io::Error> {
        // SAFETY: `fd` is a valid open file descriptor, `len` is a positive
        // file size, `PROT_READ | PROT_WRITE` and `MAP_SHARED` are valid
        // flags, and an offset of 0 is within the file.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(ptr as *mut u8)
    }
}

impl Drop for BundleStore {
    fn drop(&mut self) {
        // Best-effort msync before munmap. Without this, any bundle
        // changes made since the last explicit sync() are lost when
        // the mapping is removed. We ignore errors here because Drop
        // cannot propagate them.
        let _ = self.sync();
        // SAFETY: `self.ptr` is a valid mmap'd region of length `self.len`
        // (maintained by the `BundleStore` invariant), and `self.fd` is a
        // valid open file descriptor. Both are released here on drop.
        unsafe {
            munmap(self.ptr as *mut c_void, self.len);
            close(self.fd);
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Hashing helpers
// ─────────────────────────────────────────────────────────────────

/// FNV-1a hash of a byte slice.
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// FNV-1a hash with ASCII lowercasing.
fn fnv1a_lower(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte.to_ascii_lowercase() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// FNV-1a hash of two byte slices separated by a space, with lowercasing.
fn fnv1a_lower_bigram(a: &[u8], b: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in a {
        hash ^= byte.to_ascii_lowercase() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= b' ' as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    for &byte in b {
        hash ^= byte.to_ascii_lowercase() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Derive `BITS_PER_TOKEN` distinct positions from a single hash.
/// Uses xorshift mixing to generate independent-looking values.
fn wrangle(hash: u64, index: usize) -> u64 {
    let mut h = hash ^ (index as u64).wrapping_mul(0x9E3779B97F4A7C15);
    // xorshift64
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    h
}

// ─────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdr_encoding_basic() {
        let sdr = encode_content("hello world");
        assert!(!sdr.active.is_empty());
        assert!(sdr.active.len() <= SDR_TARGET_ACTIVE);
        // All positions should be in the content range
        for &p in &sdr.active {
            assert!(p < CONTENT_BITS as u16, "position {p} out of content range");
        }
        // Positions should be sorted
        for i in 1..sdr.active.len() {
            assert!(sdr.active[i] > sdr.active[i - 1], "not sorted at {i}");
        }
    }

    #[test]
    fn test_sdr_determinism() {
        let a = encode_content("the quick brown fox");
        let b = encode_content("the quick brown fox");
        assert_eq!(a.active, b.active, "same text should produce same SDR");
    }

    #[test]
    fn test_sdr_similarity() {
        let a = encode_content("the quick brown fox jumps over the lazy dog");
        let b = encode_content("the quick brown fox jumps over the lazy cat");
        let c = encode_content("entropy thermodynamics statistical mechanics");

        let sim_ab = a.jaccard(&b);
        let sim_ac = a.jaccard(&c);

        assert!(
            sim_ab > sim_ac,
            "similar texts should have higher Jaccard similarity: {sim_ab} vs {sim_ac}"
        );
        assert!(
            sim_ab > 0.3,
            "similar texts should share >30% bits: {sim_ab}"
        );
        assert!(
            sim_ac < 0.1,
            "unrelated texts should share <10% bits: {sim_ac}"
        );
    }

    #[test]
    fn test_emotional_encoding() {
        let tag = [0.0f32; 12];
        let bits = encode_emotional(&tag);
        assert!(
            bits.is_empty(),
            "all-below-threshold should produce no bits"
        );

        let mut tag = [0.0f32; 12];
        tag[0] = 0.8; // DA high
        tag[6] = 0.7; // Cortisol high
        let bits = encode_emotional(&tag);
        assert_eq!(bits.len(), 2);
        assert_eq!(bits[0], EMOTIONAL_BASE as u16);
        assert_eq!(bits[1], (EMOTIONAL_BASE + 6) as u16);
    }

    #[test]
    fn test_temporal_encoding() {
        let bits1 = encode_temporal(1_000_000);
        let bits2 = encode_temporal(1_000_000);
        assert_eq!(bits1, bits2, "same timestamp should produce same bits");

        let bits3 = encode_temporal(999_999_999_999);
        // Should produce up to 6 bits, all in temporal range
        for &b in &bits3 {
            assert!(b >= TEMPORAL_BASE as u16 && b < (TEMPORAL_BASE + TEMPORAL_BITS) as u16);
        }
    }

    #[test]
    fn test_episode_encoding_combines_all() {
        let mut tag = [0.0f32; 12];
        tag[0] = 0.8;
        let sdr = encode_episode("hello world", &tag, 1_000_000);

        // Should have content bits, emotional bits, and temporal bits
        assert!(
            sdr.active.iter().any(|&p| p < CONTENT_BITS as u16),
            "should have content bits"
        );
        assert!(
            sdr.active.iter().any(|&p| p >= EMOTIONAL_BASE as u16),
            "should have emotional bits"
        );
        assert!(
            sdr.active
                .iter()
                .any(|&p| p >= TEMPORAL_BASE as u16 && p < (TEMPORAL_BASE + TEMPORAL_BITS) as u16),
            "should have temporal bits"
        );
        // Should be sorted and deduplicated
        for i in 1..sdr.active.len() {
            assert!(sdr.active[i] > sdr.active[i - 1]);
        }
    }

    #[test]
    fn test_code_assignment() {
        let code1 = episode_id_to_code(1);
        let code2 = episode_id_to_code(2);
        assert_ne!(code1, code2, "different IDs should have different codes");

        // Code for ID 0 should be all zeros
        let code0 = episode_id_to_code(0);
        assert!(
            code0.iter().all(|&b| b == 0),
            "code for 0 should be all zeros"
        );

        // Code for ID 1 should have digit 1 in the last position
        assert_eq!(code_digit(&code1, LOGHD_N - 1), 1);
        assert_eq!(code_digit(&code2, LOGHD_N - 1), 2);
    }

    #[test]
    fn test_expected_activation() {
        let code = episode_id_to_code(0);
        let act = code_to_expected_activation(&code);
        assert!(
            act.iter().all(|&a| a == 0.0),
            "code 0 should have all-zero activation"
        );

        let code = episode_id_to_code(1);
        let act = code_to_expected_activation(&code);
        // Last digit is 1, so g(1) = 1/3
        assert!((act[LOGHD_N - 1] - 1.0 / 3.0).abs() < 1e-6);
        assert!(act[..LOGHD_N - 1].iter().all(|&a| a == 0.0));
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time is before UNIX_EPOCH in test")
            .as_nanos();
        std::env::temp_dir().join(format!("genesis_sdr_{name}_{pid}_{nanos}.bundles"))
    }

    #[test]
    fn test_bundle_store_create_and_query() {
        let path = tmp_path("create");
        let mut store = BundleStore::create(&path).expect("create bundle store in temp dir");

        assert_eq!(store.episode_count(), 0);
        assert_eq!(store.next_episode_id(), 1);

        // Add an episode
        let sdr = encode_content("hello world");
        let code = episode_id_to_code(1);
        store.add_episode(&code, &sdr);

        assert_eq!(store.episode_count(), 1);
        assert_eq!(store.next_episode_id(), 2);

        // Query with the same text should produce high activation
        let query = encode_content("hello world");
        let activation = store.query_activation(&query);

        // At least some bundles should have non-zero activation
        assert!(
            activation.iter().any(|&a| a > 0.0),
            "query with same text should produce non-zero activation"
        );

        // Query with unrelated text should produce lower activation
        let query2 = encode_content("quantum entanglement physics");
        let activation2 = store.query_activation(&query2);

        let max1 = activation.iter().cloned().fold(0.0f32, f32::max);
        let max2 = activation2.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max1 >= max2,
            "matching query should have higher activation: {max1} vs {max2}"
        );

        drop(store);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bundle_store_persistence() {
        let path = tmp_path("persist");

        // Create and add an episode
        {
            let mut store = BundleStore::create(&path).expect("create bundle store in temp dir");
            let sdr = encode_content("persistent memory test");
            let code = episode_id_to_code(1);
            store.add_episode(&code, &sdr);
            store.sync().expect("sync bundle store to disk");
        }

        // Reopen and verify
        {
            let store = BundleStore::open(&path).expect("reopen persisted bundle store");
            assert_eq!(store.episode_count(), 1);
            assert_eq!(store.next_episode_id(), 2);

            let query = encode_content("persistent memory test");
            let activation = store.query_activation(&query);
            assert!(
                activation.iter().any(|&a| a > 0.0),
                "persisted bundles should still respond to queries"
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_sdr_empty_text() {
        let sdr = encode_content("");
        // Empty text has no tokens, so no content bits — but the SDR
        // should still be valid (empty or near-empty)
        assert!(
            sdr.active.len() < 20,
            "empty text should produce few or no bits"
        );
    }

    #[test]
    fn test_sdr_unicode() {
        let sdr = encode_content("héllo wörld 日本語");
        assert!(
            !sdr.active.is_empty(),
            "unicode text should still produce an SDR"
        );
    }
}
