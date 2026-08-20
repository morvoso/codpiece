#!/usr/bin/env bash
# Samplers on the real model.
#
# The equivalence that matters: `--top-k 1 --temp 1` narrows the candidate set to the
# argmax, so it must reproduce greedy decoding token for token — while going through the
# host-side sampling path, with a full vocabulary of logits read back every step. If
# those two agree, the readback, the candidate construction and the draw are all wired
# up correctly; if they disagree, one of them is wrong.
# Runs INSIDE bench-window.sh.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
P="<|im_start|>user
Explain in three sentences why the sky is blue.<|im_end|>
<|im_start|>assistant
"
run() { # $1 = outfile, rest = args
  local out=$1; shift
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder \
    gen "$M" -p "$P" -n 96 -c 4096 --tp 0,1 "$@" >"$out" 2>/tmp/sm_err.txt
  grep -oE "decode:.*" /tmp/sm_err.txt | tail -1
}

echo "== greedy reference (in-graph argmax) =="
echo "  $(run /tmp/g_ref.txt)"

echo "== --temp 0 (must still take the greedy path) =="
r=$(run /tmp/g_t0.txt --temp 0)
echo "  $r  [$(diff -q /tmp/g_ref.txt /tmp/g_t0.txt >/dev/null && echo IDENTICAL || echo DIFFERS)]"

echo "== --top-k 1 --temp 1 (host sampling, must equal greedy) =="
r=$(run /tmp/g_k1.txt --top-k 1 --temp 1.0)
echo "  $r  [$(diff -q /tmp/g_ref.txt /tmp/g_k1.txt >/dev/null && echo IDENTICAL || echo DIFFERS)]"

echo "== --temp 0.7 --top-p 0.8 --top-k 20, seeds =="
run /tmp/g_s1.txt --temp 0.7 --top-p 0.8 --top-k 20 --seed 1234 >/dev/null
run /tmp/g_s1b.txt --temp 0.7 --top-p 0.8 --top-k 20 --seed 1234 >/dev/null
run /tmp/g_s2.txt --temp 0.7 --top-p 0.8 --top-k 20 --seed 9999 >/dev/null
echo "  same seed twice : $(diff -q /tmp/g_s1.txt /tmp/g_s1b.txt >/dev/null && echo REPRODUCIBLE || echo 'NOT reproducible <<<')"
echo "  different seed  : $(diff -q /tmp/g_s1.txt /tmp/g_s2.txt >/dev/null && echo 'identical <<< (sampling not varying)' || echo differs, as it should)"
echo "  vs greedy       : $(diff -q /tmp/g_ref.txt /tmp/g_s1.txt >/dev/null && echo 'identical <<<' || echo differs, as it should)"
echo "--- sampled text (seed 1234) ---"
head -c 400 /tmp/g_s1.txt
echo
