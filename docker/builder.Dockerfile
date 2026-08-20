# codpiece builder for llm-host: CUDA toolchain + rust + libclang (bindgen).
# Build:  docker build -t codpiece-builder -f docker/builder.Dockerfile docker/
# Use:    docker run --rm --runtime nvidia -v <repo>:/src -v codpiece-cargo:/root/.cargo/registry \
#           -w /src codpiece-builder nice -n 19 cargo build --release --features cuda
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      cmake git clang libclang-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH
