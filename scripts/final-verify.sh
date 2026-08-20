#!/usr/bin/env bash
# Final gate for the cached-graph work: with no --path flag (so, the shipped
# default), does the fused speculative round still produce exactly what plain
# greedy decoding produces? Speed is reported, but the verdict is the diff.
# Runs INSIDE bench-window.sh.
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
run() {
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder "$@" >/tmp/out.txt 2>/tmp/n.txt
}
stat() { grep -oE "decode:.*" /tmp/n.txt | sed 's/decode: //'; }

for PROMPT in "Explain in three sentences why the sky is blue." \
              "Write a Python function that merges two sorted lists." \
              "List the first ten prime numbers and their sum."; do
  P="<|im_start|>user
$PROMPT<|im_end|>
<|im_start|>assistant
"
  echo "### $PROMPT"
  run gen "$M" -p "$P" -n 128 -c 4096 --tp 0,1; cp /tmp/out.txt /tmp/ref.txt
  echo "  plain greedy:  $(stat)"
  for D in 1 2 3 auto; do
    run fused "$M" -p "$P" -n 128 -c 4096 --depth $D --tp 0,1
    if diff -q /tmp/ref.txt /tmp/out.txt >/dev/null; then V=IDENTICAL; else V="DIFFERS <<<"; fi
    echo "  fused depth $D [$V]: $(stat)"
  done
done
