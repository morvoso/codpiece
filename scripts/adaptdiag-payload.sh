#!/usr/bin/env bash
# Adaptive depth diagnostic: run at several --max-depth values and show the exit code
# and whatever the failure said, not just the timing. Bounding --max-depth bounds how
# many distinct graph shapes exist, which is what has separated working from failing.
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
for RAW in "Explain in three sentences why the sky is blue." \
           "Write a Python function that merges two sorted lists." \
           "List the first ten prime numbers and their sum."; do
P="<|im_start|>user
$RAW<|im_end|>
<|im_start|>assistant
"
echo "=== $RAW ==="
for MD in 3; do
  timeout 900 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder \
    fused "$M" -p "$P" -n 160 -c 4096 --depth auto --max-depth $MD --tp 0,1 \
    >/tmp/ad_out.txt 2>/tmp/ad_err.txt
  rc=$?
  echo "--- max-depth $MD: rc=$rc ---"
  if grep -q "decode:" /tmp/ad_err.txt; then
    grep -oE "decode:.*" /tmp/ad_err.txt | tail -1
  else
    grep -viE "cuda graph warmup|CUDA Graph id|ggml_cuda_init|Device [0-9]" /tmp/ad_err.txt | tail -8
  fi
done
done
