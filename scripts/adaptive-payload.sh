#!/usr/bin/env bash
# Adaptive draft depth versus every fixed depth, on prompts whose predictability
# differs a lot. The bar is not "adaptive is fastest somewhere" — it is that adaptive
# lands at or above the best fixed depth on EACH prompt without being told which.
# Losslessness is checked against plain greedy for every run.
# Runs INSIDE bench-window.sh.
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
run() {
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/tandem tandem-builder "$@" >/tmp/out.txt 2>/tmp/n.txt
}
rate() { grep -oE "\([0-9.]+ tok/s\)" /tmp/n.txt | tail -1 | tr -d '()'; }
extra() { grep -oE "adaptive \[.*" /tmp/n.txt; }
verdict() { diff -q /tmp/ref.txt /tmp/out.txt >/dev/null && echo same || echo DIFF; }

for PROMPT in "Explain in three sentences why the sky is blue." \
              "Write a Python function that merges two sorted lists." \
              "List the first ten prime numbers and their sum."; do
  P="<|im_start|>user
$PROMPT<|im_end|>
<|im_start|>assistant
"
  echo "### $PROMPT"
  run gen "$M" -p "$P" -n 160 -c 4096 --tp 0,1; cp /tmp/out.txt /tmp/ref.txt
  echo "  plain greedy      : $(rate)"
  for D in 1 2 3 4 5; do
    run fused "$M" -p "$P" -n 160 -c 4096 --depth $D --tp 0,1
    echo "  fixed depth $D     : $(rate)  [vs greedy: $(verdict)]"
  done
  run fused "$M" -p "$P" -n 160 -c 4096 --depth auto --max-depth 5 --tp 0,1
  echo "  ADAPTIVE          : $(rate)  [vs greedy: $(verdict)]"
  echo "                      $(extra)"
done
