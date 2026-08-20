#!/usr/bin/env bash
# Adaptive depth with an optimism bonus and a flat (non-decaying) prior, against the
# fixed depths it has to beat.
#
# The fourth prompt is the one that matters. The first three each have fairly uniform
# predictability, so one fixed depth suits the whole generation and adaptation has
# little to find. The mixed prompt deliberately swings — prose, then code, then a
# table — which is where a per-round choice should beat every fixed depth, and where
# the earlier version could not because its prior assumed acceptance decays along the
# chain when in code it rises.
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

bench() { # $1 = tokens, $2 = prompt
  local N=$1 PROMPT=$2
  local P="<|im_start|>user
$PROMPT<|im_end|>
<|im_start|>assistant
"
  echo "### ($N tok) $PROMPT"
  run gen "$M" -p "$P" -n $N -c 4096 --tp 0,1; cp /tmp/out.txt /tmp/ref.txt
  echo "  plain greedy : $(rate)"
  for D in 1 2 3; do
    run fused "$M" -p "$P" -n $N -c 4096 --depth $D --tp 0,1
    echo "  fixed depth $D: $(rate)  [$(verdict)]"
  done
  run fused "$M" -p "$P" -n $N -c 4096 --depth auto --tp 0,1
  echo "  ADAPTIVE     : $(rate)  [$(verdict)]  $(extra)"
}

bench 128 "Explain in three sentences why the sky is blue."
bench 128 "Write a Python function that merges two sorted lists."
bench 128 "List the first ten prime numbers and their sum."
bench 320 "Explain in two sentences what binary search is, then give a complete Python implementation with docstring and type hints, then a markdown table of its time and space complexity for best, average and worst case."
