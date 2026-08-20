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

pub mod qwen35;

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

/// A backend to place weights (and compute) on.
#[derive(Clone, Copy, Debug)]
pub enum Device {
    Cpu,
    /// CUDA device index; requires the `cuda` feature.
    Cuda(i32),
}

/// Weights resident on a backend, addressable by GGUF tensor name.
pub struct Weights {
    pub gguf: GgufFile,
    ctx: *mut ffi::ggml_context,
    buffer: ffi::ggml_backend_buffer_t,
    backend: ffi::ggml_backend_t,
    tensors: HashMap<String, *mut ffi::ggml_tensor>,
    pub bytes_loaded: u64,
    pub device: Device,
}

impl Weights {
    pub fn load(path: &Path, device: Device) -> Result<Weights, ModelError> {
        let gguf = GgufFile::open(path)?;
        let mut file = File::open(path)?;

        unsafe {
            let backend = match device {
                Device::Cpu => ffi::ggml_backend_cpu_init(),
                #[cfg(feature = "cuda")]
                Device::Cuda(i) => ffi::ggml_backend_cuda_init(i),
                #[cfg(not(feature = "cuda"))]
                Device::Cuda(_) => {
                    return Err(ModelError::Load(
                        "built without the cuda feature".into(),
                    ))
                }
            };
            if backend.is_null() {
                return Err(ModelError::Load("backend init failed".into()));
            }

            let params = ffi::ggml_init_params {
                mem_size: gguf.tensors.len() * ffi::ggml_tensor_overhead(),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                ffi::ggml_backend_free(backend);
                return Err(ModelError::Load("ggml_init failed".into()));
            }

            // Create tensor metadata mirroring the GGUF directory.
            let mut tensors = HashMap::with_capacity(gguf.tensors.len());
            for info in &gguf.tensors {
                let ne: Vec<i64> = info.dims.iter().map(|&d| d as i64).collect();
                let t = ffi::ggml_new_tensor(
                    ctx,
                    info.ty.0 as ffi::ggml_type,
                    ne.len() as std::os::raw::c_int,
                    ne.as_ptr(),
                );
                if t.is_null() {
                    ffi::ggml_free(ctx);
                    ffi::ggml_backend_free(backend);
                    return Err(ModelError::Load(format!("tensor create: {}", info.name)));
                }
                let cname = CString::new(info.name.as_str())
                    .map_err(|_| ModelError::Load(format!("NUL in tensor name {:?}", info.name)))?;
                ffi::ggml_set_name(t, cname.as_ptr());

                // Cross-check our size arithmetic against ggml's.
                let ours = info.byte_size();
                let theirs = ffi::ggml_nbytes(t) as u64;
                if ours != Some(theirs) {
                    ffi::ggml_free(ctx);
                    ffi::ggml_backend_free(backend);
                    return Err(ModelError::Load(format!(
                        "size mismatch for {}: gguf {:?} vs ggml {}",
                        info.name, ours, theirs
                    )));
                }
                tensors.insert(info.name.clone(), t);
            }

            let buffer = ffi::ggml_backend_alloc_ctx_tensors(ctx, backend);
            if buffer.is_null() {
                ffi::ggml_free(ctx);
                ffi::ggml_backend_free(backend);
                return Err(ModelError::Load("backend buffer alloc failed".into()));
            }

            // Stream weight data from disk into the backend buffer.
            let mut scratch = vec![0u8; COPY_CHUNK];
            let mut total = 0u64;
            for info in &gguf.tensors {
                let t = tensors[&info.name];
                let size = ffi::ggml_nbytes(t);
                file.seek(SeekFrom::Start(gguf.data_start + info.offset))?;
                let mut done = 0usize;
                while done < size {
                    let n = (size - done).min(COPY_CHUNK);
                    file.read_exact(&mut scratch[..n])?;
                    ffi::ggml_backend_tensor_set(
                        t,
                        scratch.as_ptr().cast(),
                        done,
                        n,
                    );
                    done += n;
                }
                total += size as u64;
            }

            Ok(Weights { gguf, ctx, buffer, backend, tensors, bytes_loaded: total, device })
        }
    }

    pub fn tensor(&self, name: &str) -> Option<*mut ffi::ggml_tensor> {
        self.tensors.get(name).copied()
    }

    /// The backend the weights live on (also used to compute graphs).
    pub fn backend(&self) -> ffi::ggml_backend_t {
        self.backend
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
            ffi::ggml_backend_buffer_free(self.buffer);
            ffi::ggml_free(self.ctx);
            ffi::ggml_backend_free(self.backend);
        }
    }
}
