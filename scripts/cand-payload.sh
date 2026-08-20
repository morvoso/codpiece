#!/usr/bin/env bash
# Candidate-restricted drafting: the draft head projects onto a shortlist instead of
# all 248,320 vocabulary rows. Verification always uses the full vocabulary, so this
# can only cost acceptance, never correctness.
#
# The point of the sweep is to separate the two effects the shortlist has. Acceptance
# is printed alongside throughput: if acceptance holds and throughput still drops, the
# gather costs more than the projection saves and the idea is dead.
# Runs INSIDE bench-window.sh.
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
P="<|im_start|>user
Explain in three sentences why the sky is blue.<|im_end|>
<|im_start|>assistant
"
run() {
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -e CODPIECE_CAND_DUMMY="${CODPIECE_CAND_DUMMY:-}" \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder "$@" >/tmp/out.txt 2>/tmp/n.txt
}
stat() { grep -oE "decode:.*" /tmp/n.txt | sed 's/decode: //'; }
verdict() { diff -q /tmp/ref.txt /tmp/out.txt >/dev/null && echo LOSSLESS || echo MISMATCH; }

run gen "$M" -p "$P" -n 160 -c 4096 --tp 0,1; cp /tmp/out.txt /tmp/ref.txt
echo "plain decode:              $(stat)"

run fused "$M" -p "$P" -n 160 -c 4096 --depth 2 --tp 0,1 --path rebuild
echo "d2 full vocab [$(verdict)]: $(stat)"

CODPIECE_CAND_DUMMY=1 run fused "$M" -p "$P" -n 160 -c 4096 --depth 2 --tp 0,1 --path rebuild --cand 4096
echo "d2 static 4096 [$(verdict)]: $(stat)"

for C in 2048 4096 16384; do
  run fused "$M" -p "$P" -n 160 -c 4096 --depth 2 --tp 0,1 --path rebuild --cand $C
  echo "d2 ctx+low $C [$(verdict)]: $(stat)"
done
