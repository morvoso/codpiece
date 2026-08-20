#!/usr/bin/env bash
# Does the larger prefill chunk still fit at production context? 512 was known to fail
# at 194K and 64 to work; the formula now picks 334 there, which is in between and
# therefore has to be measured rather than assumed.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
head -c 800000 $HOME/llm/tandem/wiki.test.raw > $HOME/llm/tandem/lc_prompt.txt
for C in "" 334; do
  lbl=${C:-auto}
  timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    ${C:+-e TANDEM_PREFILL_CHUNK=$C} \
    -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models -v $HOME/llm/tandem:/work \
    --entrypoint /src/target/release/tandem tandem-builder \
    gen "$M" -f /work/lc_prompt.txt -n 8 -c 200704 --tp 0,1 >/dev/null 2>/tmp/cc.txt
  if grep -q "decode:" /tmp/cc.txt; then
    echo "  chunk $lbl: $(grep -oE 'prefill:.*' /tmp/cc.txt | tail -1)"
  else
    echo "  chunk $lbl: FAILED — $(grep -iE 'out of memory|CUDA error' /tmp/cc.txt | head -1)"
  fi
done
