use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let vendor = manifest
        .join("../../third_party/llama.cpp/ggml")
        .canonicalize()
        .expect("vendored ggml missing — run scripts/fetch-deps.sh first");

    let cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let build_dir = out.join("ggml-build");
    std::fs::create_dir_all(&build_dir).unwrap();

    // llama.cpp's vendored ggml omits ggml.pc.in, but ggml's standalone cmake
    // configure_file()s it unconditionally. Provide the trivial template.
    let pc_in = vendor.join("ggml.pc.in");
    if !pc_in.exists() {
        std::fs::write(
            &pc_in,
            "prefix=@CMAKE_INSTALL_PREFIX@\n\
             includedir=@CMAKE_INSTALL_FULL_INCLUDEDIR@\n\
             libdir=@CMAKE_INSTALL_FULL_LIBDIR@\n\n\
             Name: ggml\n\
             Description: ggml (tandem vendored build)\n\
             Version: @GGML_VERSION@\n\
             Cflags: -I${includedir}\n\
             Libs: -L${libdir} -lggml\n",
        )
        .expect("write ggml.pc.in shim");
    }

    // Configure. Static libs, no extras. Job count is capped by the caller's
    // environment (SAFETY.md rule 5: nice + bounded jobs on shared boxes).
    let mut cfg = Command::new("cmake");
    cfg.arg("-S").arg(&vendor)
        .arg("-B").arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DGGML_BUILD_TESTS=OFF")
        .arg("-DGGML_BUILD_EXAMPLES=OFF")
        .arg("-DGGML_CCACHE=OFF")
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON");
    // Portable-by-default: binaries built on the workstation must run on
    // llm-host (i9-12900KF = AVX2/FMA/F16C, no AVX512). -march=native once
    // shipped an illegal-instruction binary; never again. Opt back in with
    // TANDEM_NATIVE_CPU=1 for machine-local perf experiments.
    if env::var("TANDEM_NATIVE_CPU").ok().as_deref() != Some("1") {
        cfg.arg("-DGGML_NATIVE=OFF")
            .arg("-DGGML_AVX=ON")
            .arg("-DGGML_AVX2=ON")
            .arg("-DGGML_FMA=ON")
            .arg("-DGGML_F16C=ON")
            .arg("-DGGML_BMI2=ON");
    }
    if cuda {
        cfg.arg("-DGGML_CUDA=ON")
            .arg("-DCMAKE_CUDA_ARCHITECTURES=86")
            // standalone ggml defaults this OFF (llama.cpp's root cmake turns
            // it on); without it the whole CUDA-graph capture path — the
            // mechanism tandem's cached decode graph exists to exploit — is
            // compiled out
            .arg("-DGGML_CUDA_GRAPHS=ON");
    }
    run(&mut cfg, "cmake configure");

    let jobs = env::var("TANDEM_BUILD_JOBS").unwrap_or_else(|_| {
        std::thread::available_parallelism()
            .map(|n| n.get().min(8).to_string())
            .unwrap_or_else(|_| "4".into())
    });
    run(
        Command::new("cmake")
            .arg("--build").arg(&build_dir)
            .arg("--config").arg("Release")
            .arg("-j").arg(&jobs),
        "cmake build",
    );

    // Static archives land under build/src (and per-backend subdirs).
    println!("cargo:rustc-link-search=native={}", build_dir.join("src").display());
    for backend in ["ggml-cpu", "ggml-blas", "ggml-cuda"] {
        let d = build_dir.join("src").join(backend);
        if d.exists() {
            println!("cargo:rustc-link-search=native={}", d.display());
        }
    }

    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
        println!("cargo:rustc-link-lib=dylib=cuda");
        // ggml-cuda @ b10423 uses NCCL for its multi-GPU comm layer
        println!("cargo:rustc-link-lib=dylib=nccl");
    }
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=m");

    // Bindings.
    let mut builder = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", vendor.join("include").display()))
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_var("GGML_.*")
        .allowlist_var("GGUF_.*")
        .use_core()
        .derive_default(true)
        .generate_comments(false)
        .layout_tests(false);
    if cuda {
        builder = builder.clang_arg("-DTANDEM_CUDA");
    }
    let bindings = builder.generate().expect("bindgen failed");
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    // Track the vendored compute sources: without this cargo never re-runs cmake after
    // a ggml edit, so a patched backend would silently not make it into the binary.
    for sub in ["ggml/src", "ggml/include", "ggml/CMakeLists.txt"] {
        println!("cargo:rerun-if-changed=../../third_party/llama.cpp/{sub}");
    }
    println!("cargo:rerun-if-changed=../../scripts/fetch-deps.sh");
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd.status().unwrap_or_else(|e| panic!("{what}: spawn failed: {e}"));
    if !status.success() {
        panic!("{what} failed with {status}");
    }
}
