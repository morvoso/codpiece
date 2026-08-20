//! GGUF v2/v3 reader. Parses the header, metadata KVs, and tensor directory
//! without touching tensor data (header-only I/O), and validates that every
//! tensor's [offset, offset+size) lies inside the file's data section.
//!
//! Format reference: ggml/docs/gguf.md (magic "GGUF", little-endian).

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" LE
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Hard caps against corrupt/hostile files (largest real file we own is ~31 GB
/// with ~1.1k tensors and ~40 KVs; caps are 100x+ above that).
const MAX_TENSORS: u64 = 1 << 20;
const MAX_KVS: u64 = 1 << 20;
const MAX_STRING_LEN: u64 = 256 * 1024 * 1024; // chat templates are ~10 KB; token lists are big but < 256 MB
const MAX_DIMS: u32 = 8;

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    BadMagic(u32),
    UnsupportedVersion(u32),
    Malformed(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io error: {e}"),
            GgufError::BadMagic(m) => write!(f, "bad magic 0x{m:08x} (not a GGUF file)"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            GgufError::Malformed(s) => write!(f, "malformed GGUF: {s}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self {
        GgufError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, GgufError>;

fn malformed(msg: impl Into<String>) -> GgufError {
    GgufError::Malformed(msg.into())
}

// ---------------------------------------------------------------------------
// Metadata values
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Short human rendering; arrays are elided to length + first elements.
    pub fn render(&self, max_elems: usize) -> String {
        match self {
            Value::String(s) => {
                if s.len() <= 120 {
                    format!("{s:?}")
                } else {
                    format!("<string, {} bytes>", s.len())
                }
            }
            Value::Array(v) => {
                let head: Vec<String> =
                    v.iter().take(max_elems).map(|e| e.render(2)).collect();
                let ell = if v.len() > max_elems { ", …" } else { "" };
                format!("[{}{}] ({} elems)", head.join(", "), ell, v.len())
            }
            Value::U8(v) => v.to_string(),
            Value::I8(v) => v.to_string(),
            Value::U16(v) => v.to_string(),
            Value::I16(v) => v.to_string(),
            Value::U32(v) => v.to_string(),
            Value::I32(v) => v.to_string(),
            Value::F32(v) => v.to_string(),
            Value::Bool(v) => v.to_string(),
            Value::U64(v) => v.to_string(),
            Value::I64(v) => v.to_string(),
            Value::F64(v) => v.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor types (ggml_type ids as serialized in GGUF)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorType(pub u32);

pub struct TypeTraits {
    pub name: &'static str,
    /// Elements per block.
    pub block: u64,
    /// Bytes per block.
    pub size: u64,
}

impl TensorType {
    /// Traits for the ggml types we recognize. Unknown ids (future types)
    /// return None; callers must degrade gracefully, not error.
    pub fn traits(self) -> Option<TypeTraits> {
        let t = |name, block, size| Some(TypeTraits { name, block, size });
        match self.0 {
            0 => t("F32", 1, 4),
            1 => t("F16", 1, 2),
            2 => t("Q4_0", 32, 18),
            3 => t("Q4_1", 32, 20),
            6 => t("Q5_0", 32, 22),
            7 => t("Q5_1", 32, 24),
            8 => t("Q8_0", 32, 34),
            9 => t("Q8_1", 32, 36),
            10 => t("Q2_K", 256, 84),
            11 => t("Q3_K", 256, 110),
            12 => t("Q4_K", 256, 144),
            13 => t("Q5_K", 256, 176),
            14 => t("Q6_K", 256, 210),
            15 => t("Q8_K", 256, 292),
            16 => t("IQ2_XXS", 256, 66),
            17 => t("IQ2_XS", 256, 74),
            18 => t("IQ3_XXS", 256, 98),
            19 => t("IQ1_S", 256, 50),
            20 => t("IQ4_NL", 32, 18),
            21 => t("IQ3_S", 256, 110),
            22 => t("IQ2_S", 256, 82),
            23 => t("IQ4_XS", 256, 136),
            24 => t("I8", 1, 1),
            25 => t("I16", 1, 2),
            26 => t("I32", 1, 4),
            27 => t("I64", 1, 8),
            28 => t("F64", 1, 8),
            29 => t("IQ1_M", 256, 56),
            30 => t("BF16", 1, 2),
            34 => t("TQ1_0", 256, 54),
            35 => t("TQ2_0", 256, 66),
            39 => t("MXFP4", 32, 17),
            _ => None,
        }
    }

    pub fn name(self) -> String {
        match self.traits() {
            Some(tr) => tr.name.to_string(),
            None => format!("type#{}", self.0),
        }
    }

    /// Row-padded byte size for `n_elems` elements, or None for unknown types.
    pub fn byte_size(self, n_elems: u64) -> Option<u64> {
        let tr = self.traits()?;
        if n_elems % tr.block != 0 {
            return None; // ggml requires row size to be a multiple of the block
        }
        Some(n_elems / tr.block * tr.size)
    }
}

// ---------------------------------------------------------------------------
// Tensor directory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dims in ggml order (ne[0] fastest-varying).
    pub dims: Vec<u64>,
    pub ty: TensorType,
    /// Offset relative to the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elems(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn byte_size(&self) -> Option<u64> {
        self.ty.byte_size(self.n_elems())
    }
}

// ---------------------------------------------------------------------------
// File reader
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    /// Absolute file offset where the tensor data section begins.
    pub data_start: u64,
    pub file_len: u64,
    pub kvs: BTreeMap<String, Value>,
    /// Tensors in file order.
    pub tensors: Vec<TensorInfo>,
}

struct Reader<R> {
    r: R,
    /// GGUF v2 serialized some counts/lengths as u32; v3 uses u64.
    wide: bool,
}

impl<R: Read> Reader<R> {
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.r.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.r.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    /// Length/count field: u64 in v3+, u32 in v1/v2.
    fn len(&mut self) -> Result<u64> {
        if self.wide {
            self.u64()
        } else {
            Ok(self.u32()? as u64)
        }
    }
    fn string(&mut self) -> Result<String> {
        let n = self.len()?;
        if n > MAX_STRING_LEN {
            return Err(malformed(format!("string length {n} exceeds cap")));
        }
        let mut buf = vec![0u8; n as usize];
        self.r.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| malformed(format!("invalid utf-8 string: {e}")))
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                if depth > 2 {
                    return Err(malformed("array nesting too deep"));
                }
                let elem_ty = self.u32()?;
                let n = self.len()?;
                if n > MAX_STRING_LEN {
                    return Err(malformed(format!("array length {n} exceeds cap")));
                }
                let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
                for _ in 0..n {
                    v.push(self.value(elem_ty, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(malformed(format!("unknown metadata value type {other}"))),
        })
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile> {
        let f = File::open(path)?;
        let file_len = f.metadata()?.len();
        let mut br = BufReader::with_capacity(1 << 20, f);

        // Header
        let mut hdr = Reader { r: &mut br, wide: true };
        let magic = hdr.u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = hdr.u32()?;
        if !(1..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let mut rd = Reader { r: &mut br, wide: version >= 3 };
        let n_tensors = rd.len()?;
        let n_kvs = rd.len()?;
        if n_tensors > MAX_TENSORS || n_kvs > MAX_KVS {
            return Err(malformed(format!(
                "implausible counts: {n_tensors} tensors, {n_kvs} kvs"
            )));
        }

        // Metadata
        let mut kvs = BTreeMap::new();
        for _ in 0..n_kvs {
            let key = rd.string()?;
            let ty = rd.u32()?;
            let val = rd.value(ty, 0)?;
            kvs.insert(key, val);
        }

        let alignment = kvs
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(malformed(format!("bad alignment {alignment}")));
        }

        // Tensor directory
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = rd.string()?;
            let n_dims = rd.u32()?;
            if n_dims > MAX_DIMS {
                return Err(malformed(format!("tensor {name}: {n_dims} dims")));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(rd.len()?);
            }
            let ty = TensorType(rd.u32()?);
            let offset = rd.u64()?;
            tensors.push(TensorInfo { name, dims, ty, offset });
        }

        // Data section starts at the next alignment boundary after the directory.
        let pos = br.stream_position()?;
        let data_start = pos.div_ceil(alignment) * alignment;

        let gguf = GgufFile { version, alignment, data_start, file_len, kvs, tensors };
        gguf.validate()?;
        Ok(gguf)
    }

    /// Cheap structural validation: offsets aligned, in-bounds, non-overlapping
    /// (in file order), and the last tensor ends within the file.
    fn validate(&self) -> Result<()> {
        let data_len = self
            .file_len
            .checked_sub(self.data_start)
            .ok_or_else(|| malformed("data section starts past EOF"))?;
        let mut prev_end = 0u64;
        for t in &self.tensors {
            if t.offset % self.alignment != 0 {
                return Err(malformed(format!("tensor {} offset not aligned", t.name)));
            }
            if t.offset < prev_end {
                return Err(malformed(format!("tensor {} overlaps previous", t.name)));
            }
            if let Some(sz) = t.byte_size() {
                let end = t.offset.checked_add(sz).ok_or_else(|| {
                    malformed(format!("tensor {} size overflows", t.name))
                })?;
                if end > data_len {
                    return Err(malformed(format!(
                        "tensor {} [{}..{}] exceeds data section ({} bytes)",
                        t.name, t.offset, end, data_len
                    )));
                }
                prev_end = end;
            } else {
                // Unknown type: can't size it; trust the next tensor's offset check.
                prev_end = t.offset;
            }
        }
        Ok(())
    }

    pub fn kv(&self, key: &str) -> Option<&Value> {
        self.kvs.get(key)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.kv("general.architecture").and_then(Value::as_str)
    }

    /// Sum of sized tensor bytes (excludes unknown types).
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors.iter().filter_map(TensorInfo::byte_size).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    struct W(Vec<u8>);
    impl W {
        fn u32(&mut self, v: u32) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn u64(&mut self, v: u64) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn s(&mut self, s: &str) {
            self.u64(s.len() as u64);
            self.0.extend_from_slice(s.as_bytes());
        }
    }

    /// Build a minimal v3 file: 2 KVs, 1 F32 tensor [4, 2], data section present.
    fn mini_gguf() -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u32(GGUF_MAGIC);
        w.u32(3); // version
        w.u64(1); // tensors
        w.u64(2); // kvs
        // kv 1: general.architecture = "testarch"
        w.s("general.architecture");
        w.u32(8);
        w.s("testarch");
        // kv 2: test.layers = u32 7
        w.s("test.layers");
        w.u32(4);
        w.u32(7);
        // tensor: "t0", dims [4,2], F32, offset 0
        w.s("t0");
        w.u32(2);
        w.u64(4);
        w.u64(2);
        w.u32(0); // F32
        w.u64(0);
        // pad to 32-byte alignment, then 32 bytes of data (8 f32)
        while w.0.len() % 32 != 0 {
            w.0.push(0);
        }
        w.0.extend_from_slice(&[0u8; 32]);
        w.0
    }

    #[test]
    fn parses_mini_file() {
        let bytes = mini_gguf();
        let dir = std::env::temp_dir().join(format!("tandem-gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mini.gguf");
        let mut f = File::create(&path).unwrap();
        f.write_all(&bytes).unwrap();
        drop(f);

        let g = GgufFile::open(&path).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("testarch"));
        assert_eq!(g.kv("test.layers").and_then(Value::as_u64), Some(7));
        assert_eq!(g.tensors.len(), 1);
        assert_eq!(g.tensors[0].n_elems(), 8);
        assert_eq!(g.tensors[0].byte_size(), Some(32));
        assert_eq!(g.tensor_bytes(), 32);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_bad_magic() {
        let dir = std::env::temp_dir().join(format!("tandem-gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.gguf");
        std::fs::write(&path, b"NOPE0000").unwrap();
        assert!(matches!(GgufFile::open(&path), Err(GgufError::BadMagic(_))));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn q8_0_sizing() {
        assert_eq!(TensorType(8).byte_size(32), Some(34));
        assert_eq!(TensorType(8).byte_size(33), None); // not block-aligned
        assert_eq!(TensorType(14).byte_size(512), Some(420)); // Q6_K
    }
}
