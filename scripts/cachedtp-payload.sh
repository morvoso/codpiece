#!/usr/bin/env bash
# Now that every graph a decode round touches has a session-lifetime context, can the
# meta backend replay them under tensor parallelism? The cached path skips rebuilding
# ~5000 nodes per round; the rebuild path is the control. Both must stay lossless.
# Runs INSIDE bench-window.sh.
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
P="<|im_start|>user
Explain in three sentences why the sky is blue.<|im_end|>
<|im_start|>assistant
"
run() {
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder "$@" >/tmp/out.txt 2>/tmp/n.txt
}
stat() { grep -oE "decode:.*" /tmp/n.txt | sed 's/decode: //'; }
verdict() { diff -q /tmp/ref.txt /tmp/out.txt >/dev/null && echo LOSSLESS || echo MISMATCH; }
report() { # $1 label
  if grep -q "decode:" /tmp/n.txt; then echo "$1 [$(verdict)]: $(stat)"
  else echo "$1 [FAILED]: $(grep -iE 'assert|abort|error|codpiece-fattn' /tmp/n.txt | head -2 | tr '\n' ' ')"; fi
}

run gen "$M" -p "$P" -n 160 -c 4096 --tp 0,1; cp /tmp/out.txt /tmp/ref.txt
echo "plain decode:        $(stat)"

# short run first: a cached-graph fault shows up in the first few rounds.
# No losslessness verdict here — the reference is 160 tokens, this is 12.
run fused "$M" -p "$P" -n 12 -c 4096 --depth 2 --tp 0,1 --path cached
if grep -q "decode:" /tmp/n.txt; then echo "smoke d2 cached (12 tok) [ran]: $(stat)"
else echo "smoke d2 cached [FAILED]: $(grep -iE 'assert|abort|error|codpiece-fattn' /tmp/n.txt | head -2 | tr '\n' ' ')"; fi

for D in 1 2 3; do
  run fused "$M" -p "$P" -n 160 -c 4096 --depth $D --tp 0,1 --path rebuild
  report "d$D rebuild"
  run fused "$M" -p "$P" -n 160 -c 4096 --depth $D --tp 0,1 --path cached
  report "d$D CACHED "
done
