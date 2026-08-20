#!/usr/bin/env bash
# Backtrace a codpiece crash under tensor parallelism. Needs the codpiece-builder-gdb
# image: docker build from "FROM codpiece-builder / RUN apt-get install -y gdb".
# Edit the codpiece args below to reproduce whatever is crashing.
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
P="<|im_start|>user
Write a Python function that merges two sorted lists.<|im_end|>
<|im_start|>assistant
"
timeout 1200 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
  -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
  --entrypoint bash codpiece-builder-gdb -lc '
    gdb -q -batch -ex run -ex "bt 25" -ex "info registers rip" \
      --args /src/target/release/codpiece fused '"$M"' -p "'"$P"'" \
      -n 160 -c 4096 --depth auto --max-depth 5 --tp 0,1
  ' 2>&1 | grep -vE "CUDA graph warmup|^\[New Thread|^\[Thread .* exited" | tail -45
