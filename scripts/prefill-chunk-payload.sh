#!/usr/bin/env bash
# Prefill throughput against chunk size, at exactly the config db2.py ran.
#
# I dropped the prefill chunk to 64 so a 196K context would fit in memory, and scaled it
# by n_ctx — which means a 131072-token context gets 64 whatever the prompt is. Prefill
# is compute-bound, so small batches waste the GPU. This is the measurement I should
# have taken when I made that change.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
head -c 140000 $HOME/llm/tandem/wiki.test.raw > $HOME/llm/tandem/lc_prompt.txt

run() { # $1 = chunk
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -e TANDEM_PREFILL_CHUNK="$1" \
    -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models \
    -v $HOME/llm/tandem:/work \
    --entrypoint /src/target/release/tandem tandem-builder "${@:2}" >/dev/null 2>/tmp/pc.txt
  grep -oE "prefill: [0-9]+ tok in [0-9.]+s \([0-9.]+ tok/s\)" /tmp/pc.txt | tail -1
}

echo "fused (speculative) prefill, -c 131072:"
for C in 64 128 256 512; do
  echo "  chunk $C: $(run $C fused "$M" -f /work/lc_prompt.txt -n 8 -c 131072 --depth 3 --tp 0,1)"
done
echo "plain prefill, -c 131072 (no draft head at all):"
for C in 64 512; do
  echo "  chunk $C: $(run $C gen "$M" -f /work/lc_prompt.txt -n 8 -c 131072 --tp 0,1)"
done
