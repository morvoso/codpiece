#!/usr/bin/env bash
# Two questions in one run.
#
# 1. Plain decode costs ~26 ms/token against a 15.6 ms bandwidth floor. Under tensor
#    parallelism the meta backend cuts the graph at every AllReduce and issues each
#    piece as its own per-device launch, so the count of those cuts is what sets the
#    fixed cost of a step. Count them.
# 2. A recent meta-backend change cost ~7% on every path including plain decode, which
#    replays a single graph and should never rebuild. Count the rebuilds to find out
#    whether it rebuilds anyway.
# Runs INSIDE bench-window.sh.
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
P="<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
"
go() { # $1 = label, rest = codpiece args
  local label=$1; shift
  timeout 900 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -e GGML_META_TRACE_SUBGRAPHS=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder "$@" >/dev/null 2>/tmp/sg.txt
  echo "=== $label: $(grep -c '\[meta\] rebuild' /tmp/sg.txt) rebuilds ==="
  grep "\[meta\] rebuild" /tmp/sg.txt | sort | uniq -c | sort -rn | head -5
}
go "plain decode, 20 tokens" gen "$M" -p "$P" -n 20 -c 4096 --tp 0,1
go "fused d2 cached, 20 tokens" fused "$M" -p "$P" -n 20 -c 4096 --depth 2 --tp 0,1 --path cached
