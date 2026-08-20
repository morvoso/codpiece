#!/usr/bin/env bash
# How much does one more chain step cost, and does that change with context?
#
# The adaptive picker assumes a constant 6.05 ms per chain step, measured at ~180
# tokens of context. On a 320-token generation the implied cost was closer to 9-13 ms,
# which is enough to invert the ranking of depth 2 against depth 3. Measure it directly:
# the stats line reports rounds and decode seconds, so ms/round is exact, and the
# difference between adjacent depths at the same length is the per-step cost.
# Runs INSIDE bench-window.sh.
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
P="<|im_start|>user
Explain in two sentences what binary search is, then give a complete Python implementation with docstring and type hints, then a markdown table of its time and space complexity for best, average and worst case.<|im_end|>
<|im_start|>assistant
"
run() {
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder "$@" >/dev/null 2>/tmp/n.txt
  # "<tokens> <seconds> <rounds>" straight from the stats line
  sed -n 's/.*decode: \([0-9]*\) tok in \([0-9.]*\)s.*, \([0-9]*\) rounds.*/\1 \2 \3/p' /tmp/n.txt | tail -1
}
echo "tokens depth ms/round"
for N in 128 256 512; do
  for D in 1 2 3; do
    read -r tok secs rounds <<<"$(run fused "$M" -p "$P" -n $N -c 4096 --depth $D --tp 0,1)"
    if [ -n "$rounds" ] && [ "$rounds" -gt 0 ]; then
      echo "$N $D $(awk -v s="$secs" -v r="$rounds" 'BEGIN{printf "%.1f", 1000*s/r}')"
    else
      echo "$N $D FAILED"
    fi
  done
done
