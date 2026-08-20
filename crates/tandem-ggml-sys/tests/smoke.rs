//! End-to-end FFI smoke test: build a tiny graph, run it on the ggml CPU
//! backend, check the arithmetic. Proves: cmake vendor build, static linking,
//! struct layout, and the backend API surface tandem-runtime will use.

use tandem_ggml_sys::*;

#[test]
fn cpu_matmul_end_to_end() {
    unsafe {
        let params = ggml_init_params {
            mem_size: ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = ggml_init(params);
        assert!(!ctx.is_null(), "ggml_init");

        // A: 3 rows x 2 cols (ne = [2, 3]), B: 2 rows x 2 cols (ne = [2, 2]).
        // ggml_mul_mat(A, B)[i, j] = dot(A.row(i), B.row(j)) -> ne = [3, 2].
        let a = ggml_new_tensor_2d(ctx, ggml_type_GGML_TYPE_F32, 2, 3);
        let b = ggml_new_tensor_2d(ctx, ggml_type_GGML_TYPE_F32, 2, 2);
        let c = ggml_mul_mat(ctx, a, b);

        let gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);

        let backend = ggml_backend_cpu_init();
        assert!(!backend.is_null(), "cpu backend");

        let buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert!(!buf.is_null(), "alloc ctx tensors");

        let a_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: [f32; 4] = [1.0, 1.0, 2.0, 2.0];
        ggml_backend_tensor_set(a, a_data.as_ptr().cast(), 0, size_of_val(&a_data));
        ggml_backend_tensor_set(b, b_data.as_ptr().cast(), 0, size_of_val(&b_data));

        let status = ggml_backend_graph_compute(backend, gf);
        assert_eq!(status, ggml_status_GGML_STATUS_SUCCESS, "graph compute");

        let mut out = [0.0f32; 6];
        ggml_backend_tensor_get(c, out.as_mut_ptr().cast(), 0, size_of_val(&out));
        assert_eq!(out, [3.0, 7.0, 11.0, 6.0, 14.0, 22.0]);

        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
}

#[test]
fn vendored_ops_present() {
    // The two ops tandem's qwen35 graph cannot live without. Their presence at
    // this pin is the whole point of vendoring (ARCHITECTURE.md §1).
    assert_ne!(ggml_op_GGML_OP_GATED_DELTA_NET, ggml_op_GGML_OP_NONE);
    assert_ne!(ggml_op_GGML_OP_SSM_CONV, ggml_op_GGML_OP_NONE);
}
