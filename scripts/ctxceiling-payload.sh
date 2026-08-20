#!/usr/bin/env bash
# What context does tandem actually serve end to end? Allocation alone reaches 200K,
# and a 27K-token prompt runs; 194K runs the cards completely out of memory. Find the
# ceiling by running real prompts of increasing length.
#
# Budget per card: 14.65 GiB of weights, plus KV at 64 KiB/token halved across the two
# cards, plus the compute buffers.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
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
    echo "    $1: $(grep -oE 'prefill:.*' /tmp/c_err.txt | tail -1)"
  else
    echo "    $1: FAILED — $(grep -iE 'out of memory|assert|CUDA error' /tmp/c_err.txt | head -1)"
  fi
}

# chars -> roughly tokens/4.13
for CHARS in 620000 700000; do
  head -c $CHARS $HOME/llm/tandem/wiki.test.raw > $HOME/llm/tandem/lc_prompt.txt
  CTX=$(( (CHARS/4 + 2048) / 256 * 256 ))
  echo "### $CHARS chars (~$((CHARS/4130))K tokens), -c $CTX"
  run fused "$M" -f /work/lc_prompt.txt -n 16 -c $CTX --depth 3 --tp 0,1
  report "fused d3"
done
