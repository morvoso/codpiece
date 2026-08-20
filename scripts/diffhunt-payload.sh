#!/usr/bin/env bash
# The Python prompt makes fused decoding diverge from plain greedy. Find out where,
# and whether it is specific to the cached graph path or to speculation itself.
# Runs INSIDE bench-window.sh.
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
P="<|im_start|>user
Write a Python function that merges two sorted lists.<|im_end|>
<|im_start|>assistant
"
run() {
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder "$@" >"$1.txt" 2>/tmp/n.txt
}
gen_to() { # $1 = outfile, rest = args
  local out=$1; shift
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder "$@" >"$out" 2>/tmp/n.txt
  grep -oE "decode:.*" /tmp/n.txt
}

gen_to /tmp/d_ref.txt   gen   "$M" -p "$P" -n 128 -c 4096 --tp 0,1
gen_to /tmp/d_ref2.txt  gen   "$M" -p "$P" -n 128 -c 4096 --tp 0,1
echo "plain greedy vs ITSELF rerun: $(diff -q /tmp/d_ref.txt /tmp/d_ref2.txt >/dev/null && echo stable || echo NONDETERMINISTIC)"

gen_to /tmp/d_cached.txt  fused "$M" -p "$P" -n 128 -c 4096 --depth 1 --tp 0,1
echo "fused d1 cached : $(diff -q /tmp/d_ref.txt /tmp/d_cached.txt >/dev/null && echo IDENTICAL || echo DIFFERS)"
gen_to /tmp/d_rebuild.txt fused "$M" -p "$P" -n 128 -c 4096 --depth 1 --tp 0,1 --path rebuild
echo "fused d1 rebuild: $(diff -q /tmp/d_ref.txt /tmp/d_rebuild.txt >/dev/null && echo IDENTICAL || echo DIFFERS)"

echo "=== first divergence, plain greedy vs fused cached ==="
diff /tmp/d_ref.txt /tmp/d_cached.txt | head -20
echo "=== plain greedy, first 6 lines ==="
head -6 /tmp/d_ref.txt
echo "=== fused cached, first 6 lines ==="
head -6 /tmp/d_cached.txt
