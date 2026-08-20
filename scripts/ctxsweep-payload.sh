#!/usr/bin/env bash
# Where does context stop working, and why? Short prompt so this measures allocation
# and graph construction rather than prefill. Budget per card at 196K is 14.65 GiB of
# weights + 6.0 GiB of KV = 20.65 of 24, so this should fit — if it does not, the error
# says what actually ran out.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
P="<|im_start|>user
Say hello.<|im_end|>
<|im_start|>assistant
"
for C in 32768 65536 131072 200704; do
  timeout 900 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/codpiece codpiece-builder \
    gen "$M" -p "$P" -n 8 -c $C --tp 0,1 >/tmp/s_out.txt 2>/tmp/s_err.txt
  rc=$?
  if grep -q "decode:" /tmp/s_err.txt; then
    echo "-c $C: OK  $(grep -oE 'decode:.*' /tmp/s_err.txt | tail -1)"
  else
    echo "-c $C: rc=$rc"
    grep -viE "ggml_cuda_init|Device [0-9]|CUDA graph|^/src/target|^/lib/" /tmp/s_err.txt | tail -5 | sed 's/^/    /'
  fi
done
