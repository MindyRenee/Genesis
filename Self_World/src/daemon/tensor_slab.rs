//! Named multidimensional tensor slab — shared numeric state between
//! the subcognitive daemon and the cognitive mind.
//!
//! The IPC commands `PUT_TENSOR` / `GET_TENSOR` / `LIST_TENSORS` /
//! `DROP_TENSOR` let either side publish and read dense tensors
//! (world frames, salience fields, embeddings, motor fields) without
//! the data passing through `CoreState`'s mmap layout. Slab contents
//! are process-local runtime state — a field named `world.frame`
//! describes a moment and is meaningless across restarts, so it lives
//! beside the shared body-state statics rather than inside the
//! checksummed state schema.
//!
//! Wire layout (all integers little-endian, data in C row-major
//! order):
//!
//! ```text
//! tensor        := [u8 dtype] [u8 ndim] [ndim × u32 dims] [payload]
//! put request   := [u8 name_len] [name utf8] tensor
//! get request   := [u8 name_len] [name utf8]
//! get response  := [u8 found] (tensor iff found == 1)
//! list response := [u16 count] (per entry: [u8 name_len] [name]
//!                  [u8 dtype] [u8 ndim] [ndim × u32 dims])
//! drop request  := [u8 name_len] [name utf8]
//! ```

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Element dtypes shared across the wire. The codes are part of the
/// protocol — `python/genesis_client/tensor.py` mirrors them.
pub mod dtype {
    /// One byte per element (grid cells, masks, action ids).
    pub const U8: u8 = 0;
    /// Signed 32-bit integers (counts, discrete state).
    pub const I32: u8 = 1;
    /// 32-bit floats (fields, scores, salience maps).
    pub const F32: u8 = 2;
    /// 64-bit floats (high-precision latents).
    pub const F64: u8 = 3;

    /// Byte width of one element, or `None` for an unknown code.
    pub fn width(code: u8) -> Option<usize> {
        match code {
            U8 => Some(1),
            I32 | F32 => Some(4),
            F64 => Some(8),
            _ => None,
        }
    }

    /// Stable display name for diagnostics.
    pub fn name(code: u8) -> &'static str {
        match code {
            U8 => "u8",
            I32 => "i32",
            F32 => "f32",
            F64 => "f64",
            _ => "?",
        }
    }
}

/// Maximum tensor rank — grids, volumes, and stacked feature maps fit
/// comfortably inside eight axes.
pub const MAX_DIMS: usize = 8;
/// Names stay short enough to prefix telemetry lines.
pub const MAX_NAME_LEN: usize = 128;
/// One tensor may not exceed 4 MiB of payload.
pub const MAX_TENSOR_BYTES: usize = 4 << 20;
/// Bound on the number of live named tensors.
pub const MAX_TENSORS: usize = 256;
/// Bound on total slab payload bytes.
pub const MAX_SLAB_BYTES: usize = 32 << 20;

/// Why a tensor operation was rejected. `ipc.rs` maps these onto the
/// shared `error` module's byte codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorError {
    /// The request ran out of bytes mid-structure.
    PayloadTooShort,
    /// A field held a value the format forbids (bad dtype, rank over
    /// `MAX_DIMS`, empty/oversized name, data length not matching
    /// `dtype_width × prod(dims)`).
    InvalidValue,
    /// A new name arrived while the slab held `MAX_TENSORS` tensors,
    /// or total payload crossed `MAX_SLAB_BYTES`.
    SlabFull,
}

/// A dense, C-order tensor: dtype tag, shape, and raw little-endian
/// element bytes. Strides are implied — consumers that need
/// transposed views reshape after fetch.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Element dtype code (`dtype::*`).
    pub dtype: u8,
    /// One extent per axis; `shape.len() == rank`.
    pub shape: Vec<u32>,
    /// `dtype_width(dtype) × prod(shape)` bytes of element data.
    pub data: Vec<u8>,
}

impl Tensor {
    /// Build a tensor, validating dtype, rank, and payload length.
    pub fn new(dtype: u8, shape: Vec<u32>, data: Vec<u8>) -> Result<Self, TensorError> {
        let width = dtype::width(dtype).ok_or(TensorError::InvalidValue)?;
        if shape.is_empty() || shape.len() > MAX_DIMS {
            return Err(TensorError::InvalidValue);
        }
        let elements = checked_elements(&shape)?;
        let expected = elements
            .checked_mul(width)
            .ok_or(TensorError::InvalidValue)?;
        if expected != data.len() {
            return Err(TensorError::InvalidValue);
        }
        if data.len() > MAX_TENSOR_BYTES {
            return Err(TensorError::InvalidValue);
        }
        Ok(Self { dtype, shape, data })
    }

    /// Number of elements (`prod(shape)`; zero when any extent is 0).
    pub fn elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Serialize as `[dtype][ndim][dims][data]` for the wire.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + self.shape.len() * 4 + self.data.len());
        out.push(self.dtype);
        out.push(self.shape.len() as u8);
        for &d in &self.shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&self.data);
        out
    }

    /// Parse a tensor from the front of `buf`; returns the tensor and
    /// the number of bytes consumed so callers can append fields.
    pub fn decode(buf: &[u8]) -> Result<(Self, usize), TensorError> {
        if buf.len() < 2 {
            return Err(TensorError::PayloadTooShort);
        }
        let dtype = buf[0];
        let width = dtype::width(dtype).ok_or(TensorError::InvalidValue)?;
        let ndim = buf[1] as usize;
        if ndim == 0 || ndim > MAX_DIMS {
            return Err(TensorError::InvalidValue);
        }
        let header = 2 + ndim * 4;
        if buf.len() < header {
            return Err(TensorError::PayloadTooShort);
        }
        let shape: Vec<u32> = (0..ndim)
            .map(|i| {
                u32::from_le_bytes(
                    buf[2 + i * 4..6 + i * 4]
                        .try_into()
                        .expect("slice is 4 bytes; buf.len() >= header checked above"),
                )
            })
            .collect();
        let nbytes = checked_elements(&shape)?
            .checked_mul(width)
            .ok_or(TensorError::InvalidValue)?;
        if nbytes > MAX_TENSOR_BYTES {
            return Err(TensorError::InvalidValue);
        }
        let end = header + nbytes;
        if buf.len() < end {
            return Err(TensorError::PayloadTooShort);
        }
        let data = buf[header..end].to_vec();
        Ok((Self { dtype, shape, data }, end))
    }
}

/// Product of extents with overflow/upper-bound checks. An extent of
/// zero is legal (an empty tensor); multiplication stays in `u64`
/// until the bound check so `usize` width never matters.
fn checked_elements(shape: &[u32]) -> Result<usize, TensorError> {
    let mut total: u64 = 1;
    for &d in shape {
        total = total
            .checked_mul(d as u64)
            .ok_or(TensorError::InvalidValue)?;
        if total > (MAX_TENSOR_BYTES as u64) {
            // No dtype widens below 1 byte, so an element count above
            // the byte cap can never be a legal tensor.
            return Err(TensorError::InvalidValue);
        }
    }
    Ok(total as usize)
}

/// The process-local slab. Same pattern as `interoception`'s shared
/// statics: the registry outlives any one IPC connection and never
/// touches the mmap file.
#[derive(Default)]
pub struct TensorSlab {
    tensors: HashMap<String, Tensor>,
    payload_bytes: usize,
}

impl TensorSlab {
    /// Insert or overwrite a tensor. Existing names always update —
    /// field refresh is the dominant pattern; only *new* names run
    /// into the count and total-bytes caps.
    pub fn put(&mut self, name: String, tensor: Tensor) -> Result<(), TensorError> {
        if name.is_empty() || name.len() > MAX_NAME_LEN {
            return Err(TensorError::InvalidValue);
        }
        if let Some(old) = self.tensors.get(&name) {
            self.payload_bytes -= old.data.len();
        } else if self.tensors.len() >= MAX_TENSORS {
            return Err(TensorError::SlabFull);
        }
        if self.payload_bytes + tensor.data.len() > MAX_SLAB_BYTES {
            return Err(TensorError::SlabFull);
        }
        self.payload_bytes += tensor.data.len();
        self.tensors.insert(name, tensor);
        Ok(())
    }

    /// Fetch a tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Remove a tensor; absent names are a no-op (idempotent remove).
    pub fn remove(&mut self, name: &str) {
        if let Some(old) = self.tensors.remove(name) {
            self.payload_bytes -= old.data.len();
        }
    }

    /// `(name, dtype, shape)` for every live tensor, sorted by name so
    /// the wire listing is deterministic.
    pub fn list(&self) -> Vec<(String, u8, Vec<u32>)> {
        let mut out: Vec<_> = self
            .tensors
            .iter()
            .map(|(n, t)| (n.clone(), t.dtype, t.shape.clone()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

/// The shared slab instance.
static SLAB: OnceLock<Mutex<TensorSlab>> = OnceLock::new();

/// Access the daemon's tensor slab.
pub fn slab() -> &'static Mutex<TensorSlab> {
    SLAB.get_or_init(|| Mutex::new(TensorSlab::default()))
}

/// Parse `[u8 name_len][name]` from the front of `buf`; returns the
/// name and the offset just past it (where the tensor header begins
/// for PUT, or where the request ends for GET/DROP).
pub fn decode_name(buf: &[u8]) -> Result<(String, usize), TensorError> {
    if buf.is_empty() {
        return Err(TensorError::PayloadTooShort);
    }
    let len = buf[0] as usize;
    if len == 0 || len > MAX_NAME_LEN {
        return Err(TensorError::InvalidValue);
    }
    if buf.len() < 1 + len {
        return Err(TensorError::PayloadTooShort);
    }
    let name = String::from_utf8_lossy(&buf[1..1 + len]).into_owned();
    Ok((name, 1 + len))
}

/// Encode the slab listing as `[u16 count][entries…]` — names and
/// metadata only, never payload.
pub fn encode_list(entries: &[(String, u8, Vec<u32>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (name, dtype, shape) in entries {
        out.push(name.len() as u8);
        out.extend_from_slice(name.as_bytes());
        out.push(*dtype);
        out.push(shape.len() as u8);
        for &d in shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_tensor(shape: &[u32], fill: f32) -> Tensor {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let mut data = Vec::with_capacity(n * 4);
        for _ in 0..n {
            data.extend_from_slice(&fill.to_le_bytes());
        }
        Tensor::new(dtype::F32, shape.to_vec(), data).unwrap()
    }

    #[test]
    fn tensor_roundtrip_multidim() {
        // 3-D tensor: shape survives encode→decode intact.
        let t = f32_tensor(&[2, 3, 4], 1.5);
        assert_eq!(t.elements(), 24);
        let bytes = t.encode();
        let (back, used) = Tensor::decode(&bytes).unwrap();
        assert_eq!(used, bytes.len());
        assert_eq!(back, t);
    }

    #[test]
    fn tensor_decode_rejects_truncated_payload() {
        let t = f32_tensor(&[4, 4], 0.0);
        let mut bytes = t.encode();
        bytes.truncate(bytes.len() - 3);
        assert_eq!(Tensor::decode(&bytes), Err(TensorError::PayloadTooShort));
    }

    #[test]
    fn tensor_rejects_dtype_shape_mismatch() {
        // 4 f32 elements declared, only 3 present.
        let data = vec![0u8; 12];
        assert!(Tensor::new(dtype::F32, vec![4], data).is_err());
    }

    #[test]
    fn tensor_allows_zero_extent() {
        let t = Tensor::new(dtype::U8, vec![0, 8], vec![]).unwrap();
        assert_eq!(t.elements(), 0);
    }

    #[test]
    fn tensor_rejects_unknown_dtype_and_excess_rank() {
        assert!(Tensor::new(9, vec![2], vec![0; 2]).is_err());
        let shape = vec![1u32; MAX_DIMS + 1];
        assert!(Tensor::new(dtype::U8, shape, vec![0]).is_err());
    }

    #[test]
    fn slab_put_get_drop() {
        let mut slab = TensorSlab::default();
        let t = f32_tensor(&[2, 2], 0.25);
        slab.put("a".into(), t.clone()).unwrap();
        assert_eq!(slab.get("a"), Some(&t));
        slab.remove("a");
        assert!(slab.get("a").is_none());
        slab.remove("a"); // idempotent
    }

    #[test]
    fn slab_overwrite_updates_byte_count() {
        let mut slab = TensorSlab::default();
        slab.put("f".into(), f32_tensor(&[4], 0.0)).unwrap();
        let big = f32_tensor(&[8], 0.0);
        slab.put("f".into(), big.clone()).unwrap();
        assert_eq!(slab.payload_bytes, 32);
        // Overwrite is allowed even when a new name would be refused.
        for i in 0..MAX_TENSORS - 1 {
            slab.put(format!("t{i}"), f32_tensor(&[1], 0.0)).unwrap();
        }
        assert_eq!(
            slab.put("overflow".into(), f32_tensor(&[1], 0.0)),
            Err(TensorError::SlabFull)
        );
        // …but overwriting an existing name still works at the cap.
        slab.put("f".into(), f32_tensor(&[8], 1.0)).unwrap();
    }

    #[test]
    fn decode_name_checks_bounds() {
        assert!(decode_name(&[]).is_err());
        assert!(decode_name(&[5, b'a']).is_err()); // declared 5, gave 1
        let (name, off) = decode_name(&[3, b'f', b'o', b'o', 0xAA]).unwrap();
        assert_eq!(name, "foo");
        assert_eq!(off, 4);
    }

    #[test]
    fn encode_list_carries_metadata_only() {
        let entries = vec![
            ("a".to_string(), dtype::U8, vec![64, 64]),
            ("b".to_string(), dtype::F32, vec![7]),
        ];
        let bytes = encode_list(&entries);
        assert_eq!(&bytes[..2], &2u16.to_le_bytes());
        // Entry a: len + name + dtype + ndim + 2 dims.
        let a_end = 2 + 1 + 1 + 2 + 8;
        assert_eq!(bytes[2], 1);
        assert_eq!(&bytes[3..4], b"a");
        assert_eq!(bytes[4], dtype::U8);
        assert_eq!(bytes[5], 2);
        assert_eq!(&bytes[a_end..a_end + 1], &[1u8]);
    }
}
