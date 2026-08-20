#!/usr/bin/env bash
# Context is the second priority after accuracy, and everything so far has been
# measured at -c 4096. This model should make long context nearly free: only 16 of its
# 64 layers are attention, GQA 4x256, so the KV cache is ~4 KiB/token and 196K tokens
# is under a gigabyte, while the other 48 layers carry constant-size recurrent state.
# Check the implementation behaves that way — memory, prefill, and above all whether
# decode slows down as the cache fills.
# Runs INSIDE bench-window.sh.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf

run() {
  timeout 3600 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models \
    -v $HOME/llm/tandem:/work \
    --entrypoint /src/target/release/tandem tandem-builder "$@" >/tmp/lc_out.txt 2>/tmp/lc_err.txt
}
report() { # $1 label
  if grep -q "decode:" /tmp/lc_err.txt; then
    echo "  $1: $(grep -oE 'prefill:.*' /tmp/lc_err.txt | tail -1)"
  else
    echo "  $1: FAILED: $(grep -viE 'ggml_cuda_init|Device [0-9]|CUDA graph' /tmp/lc_err.txt | tail -3 | tr '\n' ' ')"
  fi
}

for CHARS in 4000 40000 110000; do
  head -c $CHARS $HOME/llm/tandem/wiki.test.raw > $HOME/llm/tandem/lc_prompt.txt
  echo "### prompt of $CHARS chars"
  run gen "$M" -f /work/lc_prompt.txt -n 32 -c 32768 --tp 0,1
  report "plain   "
  run fused "$M" -f /work/lc_prompt.txt -n 32 -c 32768 --depth 3 --tp 0,1
  report "fused d3"
  echo "  VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
done
