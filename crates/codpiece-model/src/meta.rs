//! Tensor-parallel execution via ggml's meta device.
//!
//! The meta device wraps several real devices and presents itself as ONE
//! backend. Every matrix is sliced across the members; ggml inserts the
//! all-reduce where a row-split (axis 0) product needs summing. This is the
//! mode production llama.cpp runs (`-sm tensor`), and unlike a layer split it
//! keeps both GPUs busy on every token instead of alternating.
//!
//! On this machine all inter-GPU traffic crosses PCIe host-bounce (no NVLink
//! bridge, and GeForce P2P is driver-disabled), so the all-reduce is the
//! expensive part of the design — but it is *one* reduction per split point
//! rather than a full activation handoff per layer.
//!
//! codpiece supplies one callback: given a tensor, how is it split? The
//! classification lives in `split.rs` and is unit-tested offline against the
//! production 27B's real shapes, because a wrong split does not error — it
//! silently computes the wrong thing.

use std::collections::HashMap;
use std::ffi::CStr;

use codpiece_ggml_sys as ffi;

use crate::qwen35::Hparams;
use crate::split::{self, Axis};

/// Passed to the C callback as `userdata`. Boxed and kept alive for as long
/// as the meta device exists.
pub struct SplitCtx {
    pub hp: Hparams,
    pub n_devices: usize,
    /// tensor name -> quantization block size, for granularity lookups of
    /// axis-0 reference tensors (see split::Split::axis0_ref)
    pub blck: HashMap<String, i64>,
}

fn axis_to_ffi(a: Axis) -> ffi::ggml_backend_meta_split_axis {
    match a {
        Axis::Axis0 => ffi::ggml_backend_meta_split_axis_GGML_BACKEND_SPLIT_AXIS_0,
        Axis::Axis1 => ffi::ggml_backend_meta_split_axis_GGML_BACKEND_SPLIT_AXIS_1,
        Axis::Axis2 => ffi::ggml_backend_meta_split_axis_GGML_BACKEND_SPLIT_AXIS_2,
        Axis::Mirrored => ffi::ggml_backend_meta_split_axis_GGML_BACKEND_SPLIT_AXIS_MIRRORED,
        Axis::Partial => ffi::ggml_backend_meta_split_axis_GGML_BACKEND_SPLIT_AXIS_PARTIAL,
    }
}

/// C callback. Must never panic across the FFI boundary, so every failure
/// path degrades to MIRRORED (replicate everywhere), which is always
/// arithmetically valid — just not memory-optimal.
pub unsafe extern "C" fn get_split_state(
    tensor: *const ffi::ggml_tensor,
    userdata: *mut std::ffi::c_void,
) -> ffi::ggml_backend_meta_split_state {
    let mut out = ffi::ggml_backend_meta_split_state {
        axis: ffi::ggml_backend_meta_split_axis_GGML_BACKEND_SPLIT_AXIS_MIRRORED,
        ne: [0; 256],
        nr: [0; 16],
        n_segments: 1,
    };
    out.nr[0] = 1;

    if tensor.is_null() || userdata.is_null() {
        return out;
    }
    let ctx = &*(userdata as *const SplitCtx);
    let name = match CStr::from_ptr((*tensor).name.as_ptr()).to_str() {
        Ok(n) => n,
        Err(_) => return out,
    };

    let cls = split::classify(name, &ctx.hp);
    let axis_idx = match cls.axis {
        Axis::Axis0 => 0usize,
        Axis::Axis1 => 1usize,
        Axis::Axis2 => 2usize,
        _ => return out,
    };
    let extent = (*tensor).ne[axis_idx];

    // Granularity comes from the axis-0 reference tensor's block size when we
    // know it (keeps a column split and its row-split partner aligned).
    let blck = cls
        .axis0_ref
        .and_then(|suffix| {
            let prefix = name.split_once('.').and_then(|_| {
                name.strip_prefix("blk.")
                    .and_then(|r| r.split_once('.'))
                    .map(|(n, _)| format!("blk.{n}."))
            });
            prefix.and_then(|p| ctx.blck.get(&format!("{p}{suffix}")).copied())
        })
        .or_else(|| ctx.blck.get(name).copied())
        .unwrap_or(1);

    let st = split::split_state(name, &ctx.hp, extent, blck, ctx.n_devices);
    // CODPIECE_TRACE_SPLIT names any tensor whose split ggml rejects: the meta
    // backend asserts that the per-device extents sum to the tensor's own
    // extent, and its abort does not say which tensor failed.
    if std::env::var("CODPIECE_TRACE_SPLIT").is_ok() {
        let sum: i64 = (0..st.n_segments)
            .map(|is| {
                let per: i64 = (0..ctx.n_devices).map(|d| st.ne[is * ctx.n_devices + d]).sum();
                per * st.nr[is] as i64
            })
            .sum();
        eprintln!(
            "[split] {name}: axis {:?} ne[{axis_idx}]={extent} segs={} sum={sum}{}",
            st.axis,
            st.n_segments,
            if sum == extent { "" } else { "  <-- MISMATCH" }
        );
    }
    out.axis = axis_to_ffi(st.axis);
    out.n_segments = st.n_segments as u32;
    let n = st.ne.len().min(out.ne.len());
    out.ne[..n].copy_from_slice(&st.ne[..n]);
    for (i, v) in st.nr.iter().take(out.nr.len()).enumerate() {
        out.nr[i] = *v;
    }
    out
}

/// Build a meta device over the given CUDA device indices.
/// Returns (device, backend, boxed userdata kept alive by the caller).
pub unsafe fn make_meta_backend(
    cuda_ids: &[i32],
    ctx: Box<SplitCtx>,
) -> Result<(ffi::ggml_backend_t, Box<SplitCtx>), String> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (cuda_ids, ctx);
        return Err("built without the cuda feature".into());
    }
    #[cfg(feature = "cuda")]
    {
        if cuda_ids.len() < 2 {
            return Err("tensor parallel needs >= 2 devices".into());
        }
        // Resolve ggml devices for the requested CUDA indices by matching the
        // CUDA backend's own device list order.
        let mut devs: Vec<ffi::ggml_backend_dev_t> = Vec::with_capacity(cuda_ids.len());
        for &id in cuda_ids {
            let b = ffi::ggml_backend_cuda_init(id);
            if b.is_null() {
                return Err(format!("cuda_init({id}) failed"));
            }
            let d = ffi::ggml_backend_get_device(b);
            // the temporary backend was only a handle to reach the device
            ffi::ggml_backend_free(b);
            if d.is_null() {
                return Err(format!("no ggml device for cuda {id}"));
            }
            devs.push(d);
        }

        let ud = Box::into_raw(ctx);
        let dev = ffi::ggml_backend_meta_device(
            devs.as_mut_ptr(),
            devs.len(),
            Some(get_split_state),
            ud as *mut std::ffi::c_void,
        );
        if dev.is_null() {
            let ctx = Box::from_raw(ud);
            drop(ctx);
            return Err("ggml_backend_meta_device returned null".into());
        }
        let backend = ffi::ggml_backend_dev_init(dev, std::ptr::null());
        if backend.is_null() {
            let ctx = Box::from_raw(ud);
            drop(ctx);
            return Err("meta backend init failed".into());
        }
        Ok((backend, Box::from_raw(ud)))
    }
}
