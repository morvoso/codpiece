//! Raw FFI bindings to vendored ggml (llama.cpp pin b10423).
//! Everything here is `unsafe extern "C"`; safe wrappers live in higher crates.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
