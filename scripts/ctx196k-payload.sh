#!/usr/bin/env bash
# Production serves this model at 196K context. Everything tandem has been measured at
# so far is 4K, and M5's gate assumes 196K works. Run it: ~194K tokens of real text,
# then decode, sampling VRAM while it runs.
#
# The interesting part is that this should be cheap. Only 16 of the 64 layers are
# attention (GQA 4x256), so the KV cache is ~4 KiB/token — under a gigabyte at 196K —
# and the other 48 layers carry constant-size recurrent state regardless of context.
# Runs INSIDE bench-window.sh.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
head -c 800000 $HOME/llm/tandem/wiki.test.raw > $HOME/llm/tandem/lc_prompt.txt
echo "prompt: $(wc -c < $HOME/llm/tandem/lc_prompt.txt) chars (~194K tokens at the 4.13 chars/token measured earlier)"

# sample VRAM while the run is in flight; the earlier payload sampled after exit and
# only ever saw an empty card
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ' >> /tmp/vram.txt
    sleep 5
  done ) & SAMPLER=$!
: > /tmp/vram.txt

run() {
  timeout 3600 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -e TANDEM_PREFILL_CHUNK="${TANDEM_PREFILL_CHUNK:-}" \
    -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models \
    -v $HOME/llm/tandem:/work \
    --entrypoint /src/target/release/tandem tandem-builder "$@" >/tmp/c_out.txt 2>/tmp/c_err.txt
}
report() {
  if grep -q "decode:" /tmp/c_err.txt; then
    echo "  $1: $(grep -oE 'prefill:.*' /tmp/c_err.txt | tail -1)"
  else
    echo "  $1: FAILED"
    grep -iE "assert|abort|error|failed|overflow|out of memory" /tmp/c_err.txt | head -4 | sed 's/^/      /'
    grep -viE "ggml_cuda_init|Device [0-9]|CUDA graph|^/lib/" /tmp/c_err.txt | tail -3 | sed 's/^/      /' 
  fi
}

for CHUNK in 128 32; do
  export TANDEM_PREFILL_CHUNK=$CHUNK
  echo "-- prefill chunk $CHUNK --"
  run gen "$M" -f /work/lc_prompt.txt -n 32 -c 200704 --tp 0,1
  report "plain   "
  run fused "$M" -f /work/lc_prompt.txt -n 32 -c 200704 --depth 3 --tp 0,1
  report "fused d3"
done

kill $SAMPLER 2>/dev/null
echo "  peak VRAM (MiB, per card): $(sort -k1 -n /tmp/vram.txt | tail -1)"
