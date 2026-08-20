#!/usr/bin/env bash
# 196K does not fit because two tensors are replicated on both cards rather than split:
# output.weight (BF16, 2.37 GiB) and token_embd (Q8_0, 1.26 GiB), costing 1.81 GiB per
# card. They are mirrored so that argmax and the embedding gather can happen inside the
# graph, which is what makes the fused speculative round possible under tensor
# parallelism.
#
# Plain decoding does not need that. CODPIECE_SPLIT_OUTPUT=1 splits the LM head and falls
# back to a host-side argmax — ~1 MB of logits per token over PCIe. If that is what
# stands between codpiece and production's context, it is a cheap trade.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
head -c 800000 $HOME/llm/codpiece/wiki.test.raw > $HOME/llm/codpiece/lc_prompt.txt

run() { # $1 = "split" to split the LM head, anything else leaves it mirrored
  local sp=$1; shift
  # The code tests only whether CODPIECE_SPLIT_OUTPUT is SET, so passing it as an empty
  # string still splits. It has to be absent entirely for the mirrored path.
  local extra=()
  [ "$sp" = "split" ] && extra=(-e CODPIECE_SPLIT_OUTPUT=1)
  timeout 3600 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 "${extra[@]}" \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    -v $HOME/llm/codpiece:/work \
    --entrypoint /src/target/release/codpiece codpiece-builder "$@" >/tmp/s_out.txt 2>/tmp/s_err.txt
}
report() {
  if grep -q "decode:" /tmp/s_err.txt; then
    echo "  $1: $(grep -oE 'prefill:.*' /tmp/s_err.txt | tail -1)"
  else
    echo "  $1: FAILED — $(grep -iE 'out of memory|assert|CUDA error|in-graph' /tmp/s_err.txt | head -1)"
  fi
}

echo "### 194K-token prompt, -c 200704"
run mirror gen "$M" -f /work/lc_prompt.txt -n 16 -c 200704 --tp 0,1
report "plain, mirrored head"
run split gen "$M" -f /work/lc_prompt.txt -n 16 -c 200704 --tp 0,1
report "plain, split head   "
run mirror fused "$M" -f /work/lc_prompt.txt -n 16 -c 200704 --depth 3 --tp 0,1
report "fused d3, capped MTP"

echo "### short prompt at 4K — the cap must not change anything here"
P="<|im_start|>user
Explain in three sentences why the sky is blue.<|im_end|>
<|im_start|>assistant
"
run mirror gen "$M" -p "$P" -n 128 -c 4096 --tp 0,1
report "plain               "
run mirror fused "$M" -p "$P" -n 128 -c 4096 --depth 3 --tp 0,1
report "fused d3            "
