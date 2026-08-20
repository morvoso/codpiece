//! Weight loading: GGUF file → ggml tensors resident in a backend buffer.
//!
//! The loader is architecture-agnostic; graph builders (qwen35) consume the
//! tensor map by name. Every tensor's size is cross-checked against ggml's
//! own arithmetic — the GGUF directory is never trusted blindly.

use std::collections::HashMap;
use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use tandem_ggml_sys as ffi;
use tandem_gguf::GgufFile;

pub mod meta;
pub mod qwen35;
pub mod split;

/// Copy chunk size for streaming weights into backend buffers (bounds RSS).
const COPY_CHUNK: usize = 32 << 20;

#[derive(Debug)]
pub enum ModelError {
    Gguf(tandem_gguf::GgufError),
    Io(std::io::Error),
    Load(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Gguf(e) => write!(f, "gguf: {e}"),
            ModelError::Io(e) => write!(f, "io: {e}"),
            ModelError::Load(s) => write!(f, "load: {s}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<tandem_gguf::GgufError> for ModelError {
    fn from(e: tandem_gguf::GgufError) -> Self {
        ModelError::Gguf(e)
    }
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::Io(e)
    }
}

/// Where weights live and compute runs.
#[derive(Clone, Debug)]
pub enum Device {
    Cpu,
    /// Single CUDA device; requires the `cuda` feature.
    Cuda(i32),
    /// Tensor-parallel across several CUDA devices via ggml's meta device:
    /// every matrix is sliced, both GPUs work on every token, and ggml
    /// inserts the all-reduce. This is what production llama.cpp uses
    /// (`-sm tensor`).
    CudaTensorParallel(Vec<i32>),
    /// Layer-split across several CUDA devices: contiguous layer ranges are
    /// assigned per device and a ggml scheduler moves activations between
    /// them. Required for models larger than one card (the 27B is 29.3 GiB).
    /// Transfers ride PCIe host-bounce — this box has no NVLink and GeForce
    /// P2P is driver-disabled, so the split point is chosen to cross the bus
    /// exactly once per token.
    CudaSplit(Vec<i32>),
}

/// Weights resident on one or more backends, addressable by GGUF name.
pub struct Weights {
    pub gguf: GgufFile,
    /// kept alive because the meta device holds a raw pointer to it
    split_ctx: Option<Box<crate::meta::SplitCtx>>,
    /// one context+buffer per backend (single-device models use index 0)
    ctxs: Vec<*mut ffi::ggml_context>,
    buffers: Vec<ffi::ggml_backend_buffer_t>,
    backends: Vec<ffi::ggml_backend_t>,
    /// scheduler over all backends; None for single-backend models, which
    /// use the faster raw-backend + gallocr path
    sched: Option<ffi::ggml_backend_sched_t>,
    tensors: HashMap<String, *mut ffi::ggml_tensor>,
    /// per-tensor backend index, for reporting the split
    pub bytes_per_backend: Vec<u64>,
    pub bytes_loaded: u64,
    pub device: Device,
}

impl Weights {
    pub fn load(path: &Path, device: Device) -> Result<Weights, ModelError> {
        let gguf = GgufFile::open(path)?;
        let mut file = File::open(path)?;

        // How many layers does this file have? Used to spread layers across
        // devices; non-layer tensors land on device 0.
        let n_layer = gguf
            .tensors
            .iter()
            .filter_map(|t| t.name.strip_prefix("blk."))
            .filter_map(|r| r.split_once('.'))
            .filter_map(|(n, _)| n.parse::<usize>().ok())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        unsafe {
            // Tensor parallel: one meta backend that internally owns both
            // GPUs. From here on the model looks single-device, so it keeps
            // the fast path (raw backend + gallocr + cached decode graph).
            let mut split_ctx: Option<Box<crate::meta::SplitCtx>> = None;
            let backends = match &device {
                Device::CudaTensorParallel(ids) => {
                    let hp = crate::qwen35::Hparams::from_gguf(&gguf)?;
                    let blck: HashMap<String, i64> = gguf
                        .tensors
                        .iter()
                        .map(|t| {
                            let b = t.ty.traits().map(|tr| tr.block as i64).unwrap_or(1);
                            (t.name.clone(), b)
                        })
                        .collect();
                    let ctx = Box::new(crate::meta::SplitCtx {
                        hp,
                        n_devices: ids.len(),
                        blck,
                    });
                    let (backend, kept) = crate::meta::make_meta_backend(ids, ctx)
                        .map_err(ModelError::Load)?;
                    split_ctx = Some(kept);
                    vec![backend]
                }
                _ => init_backends(&device)?,
            };
            let n_backends = backends.len();
            // set when a CPU fallback is appended for the scheduler
            let mut backends_all: Vec<ffi::ggml_backend_t> = Vec::new();

            // Assign every tensor to a backend index.
            let assign = |name: &str| -> usize {
                if n_backends == 1 {
                    return 0;
                }
                match layer_index(name) {
                    // spread layers evenly; ceil so device 0 never overfills
                    Some(il) => (il * n_backends / n_layer.max(1)).min(n_backends - 1),
                    None => 0,
                }
            };

            let mut ctxs = Vec::with_capacity(n_backends);
            for _ in 0..n_backends {
                let params = ffi::ggml_init_params {
                    mem_size: (gguf.tensors.len() + 8) * ffi::ggml_tensor_overhead(),
                    mem_buffer: std::ptr::null_mut(),
                    no_alloc: true,
                };
                let ctx = ffi::ggml_init(params);
                if ctx.is_null() {
                    return Err(ModelError::Load("ggml_init failed".into()));
                }
                ctxs.push(ctx);
            }

            // Create tensor metadata in the context of its assigned device.
            let mut tensors = HashMap::with_capacity(gguf.tensors.len());
            let mut where_ = HashMap::with_capacity(gguf.tensors.len());
            for info in &gguf.tensors {
                let bi = assign(&info.name);
                let ne: Vec<i64> = info.dims.iter().map(|&d| d as i64).collect();
                let t = ffi::ggml_new_tensor(
                    ctxs[bi],
                    info.ty.0 as ffi::ggml_type,
                    ne.len() as std::os::raw::c_int,
                    ne.as_ptr(),
                );
                if t.is_null() {
                    return Err(ModelError::Load(format!("tensor create: {}", info.name)));
                }
                let cname = CString::new(info.name.as_str())
                    .map_err(|_| ModelError::Load(format!("NUL in tensor name {:?}", info.name)))?;
                ffi::ggml_set_name(t, cname.as_ptr());

                // Cross-check our size arithmetic against ggml's.
                let ours = info.byte_size();
                let theirs = ffi::ggml_nbytes(t) as u64;
                if ours != Some(theirs) {
                    return Err(ModelError::Load(format!(
                        "size mismatch for {}: gguf {:?} vs ggml {}",
                        info.name, ours, theirs
                    )));
                }
                tensors.insert(info.name.clone(), t);
                where_.insert(info.name.clone(), bi);
            }

            let mut buffers = Vec::with_capacity(n_backends);
            for (bi, &ctx) in ctxs.iter().enumerate() {
                let buf = ffi::ggml_backend_alloc_ctx_tensors(ctx, backends[bi]);
                if buf.is_null() {
                    return Err(ModelError::Load(format!(
                        "backend buffer alloc failed for device {bi}"
                    )));
                }
                buffers.push(buf);
            }

            // Stream weight data from disk into the backend buffers.
            let mut scratch = vec![0u8; COPY_CHUNK];
            let mut total = 0u64;
            let mut per_backend = vec![0u64; n_backends];
            for info in &gguf.tensors {
                let t = tensors[&info.name];
                let size = ffi::ggml_nbytes(t);
                file.seek(SeekFrom::Start(gguf.data_start + info.offset))?;
                let mut done = 0usize;
                while done < size {
                    let n = (size - done).min(COPY_CHUNK);
                    file.read_exact(&mut scratch[..n])?;
                    ffi::ggml_backend_tensor_set(t, scratch.as_ptr().cast(), done, n);
                    done += n;
                }
                total += size as u64;
                per_backend[where_[&info.name]] += size as u64;
            }

            // Multi-backend models execute through a scheduler, which places
            // each node and inserts the cross-device copies. ggml requires the
            // LAST backend in the list to be the CPU (it is the fallback for
            // ops no accelerator claims), so append one for scheduling only —
            // no weights are placed there.
            let sched = if n_backends > 1 {
                let mut sched_backends = backends.clone();
                sched_backends.push(ffi::ggml_backend_cpu_init());
                if sched_backends.last().map(|b| b.is_null()).unwrap_or(true) {
                    return Err(ModelError::Load("cpu fallback backend init".into()));
                }
                let s = ffi::ggml_backend_sched_new(
                    sched_backends.as_ptr() as *mut ffi::ggml_backend_t,
                    std::ptr::null_mut(),
                    sched_backends.len() as std::os::raw::c_int,
                    8192,
                    false,
                    false,
                );
                // keep the fallback alive for the scheduler's lifetime
                backends_all = sched_backends;
                if s.is_null() {
                    return Err(ModelError::Load("sched_new failed".into()));
                }
                Some(s)
            } else {
                None
            };

            Ok(Weights {
                gguf,
                split_ctx,
                ctxs,
                buffers,
                // own every backend we created, including the CPU fallback
                backends: if backends_all.is_empty() { backends } else { backends_all },
                sched,
                tensors,
                bytes_per_backend: per_backend,
                bytes_loaded: total,
                device,
            })
        }
    }

    pub fn tensor(&self, name: &str) -> Option<*mut ffi::ggml_tensor> {
        self.tensors.get(name).copied()
    }

    /// Primary backend (device 0). Single-device models compute on it
    /// directly; multi-device models use `sched()` instead.
    pub fn backend(&self) -> ffi::ggml_backend_t {
        self.backends[0]
    }

    /// Scheduler for multi-device execution, if this model is split.
    pub fn sched(&self) -> Option<ffi::ggml_backend_sched_t> {
        self.sched
    }

    /// Devices holding weights (excludes the CPU fallback the scheduler
    /// requires).
    pub fn n_backends(&self) -> usize {
        self.bytes_per_backend.len()
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.device, Device::Cpu)
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Read a tensor back out of the backend (for verification/debug).
    pub fn tensor_bytes(&self, name: &str) -> Option<Vec<u8>> {
        let t = self.tensor(name)?;
        unsafe {
            let n = ffi::ggml_nbytes(t);
            let mut out = vec![0u8; n];
            ffi::ggml_backend_tensor_get(t, out.as_mut_ptr().cast(), 0, n);
            Some(out)
        }
    }
}

impl Drop for Weights {
    fn drop(&mut self) {
        unsafe {
            if let Some(s) = self.sched {
                ffi::ggml_backend_sched_free(s);
            }
            for &b in &self.buffers {
                ffi::ggml_backend_buffer_free(b);
            }
            for &c in &self.ctxs {
                ffi::ggml_free(c);
            }
            for &b in &self.backends {
                ffi::ggml_backend_free(b);
            }
        }
    }
}

/// `blk.<N>.…` → N
fn layer_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .and_then(|(n, _)| n.parse().ok())
}

unsafe fn init_backends(device: &Device) -> Result<Vec<ffi::ggml_backend_t>, ModelError> {
    let mut out = Vec::new();
    match device {
        Device::Cpu => out.push(ffi::ggml_backend_cpu_init()),
        #[cfg(feature = "cuda")]
        Device::Cuda(i) => out.push(ffi::ggml_backend_cuda_init(*i)),
        #[cfg(feature = "cuda")]
        Device::CudaSplit(ids) => {
            if ids.is_empty() {
                return Err(ModelError::Load("CudaSplit with no devices".into()));
            }
            for &i in ids {
                out.push(ffi::ggml_backend_cuda_init(i));
            }
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) | Device::CudaSplit(_) => {
            return Err(ModelError::Load("built without the cuda feature".into()))
        }
        // built by load() itself: the meta device needs model hparams
        Device::CudaTensorParallel(_) => {
            return Err(ModelError::Load("tensor-parallel handled in load()".into()))
        }
    }
    if out.iter().any(|b| b.is_null()) {
        return Err(ModelError::Load("backend init failed".into()));
    }
    Ok(out)
}
