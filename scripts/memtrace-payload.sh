#!/usr/bin/env bash
# Where does the memory go at long context? Weights (14.65 GiB/card) and the KV cache
# (64 KiB/token, halved across the cards) are arithmetic; everything else is the compute
# buffer, which is what actually fails to allocate at 196K. Print it per graph, at
# several contexts, with a short prompt so the KV bucket is the only thing changing.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
P="<|im_start|>user
Say hello.<|im_end|>
<|im_start|>assistant
"
for C in 8192 65536 137216 200704; do
  echo "### -c $C  (KV per card: $(python3 -c "print(f'{$C*65536/2/2**30:.2f}')") GiB)"
  timeout 900 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 -e TANDEM_TRACE_MEM=1 \
    -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/tandem tandem-builder \
    gen "$M" -p "$P" -n 4 -c $C --tp 0,1 >/dev/null 2>/tmp/m_err.txt
  grep -E "^\[mem\]" /tmp/m_err.txt | sort -u | sed 's/^/  /'
  timeout 900 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 -e TANDEM_TRACE_MEM=1 \
    -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models \
    --entrypoint /src/target/release/tandem tandem-builder \
    fused "$M" -p "$P" -n 4 -c $C --depth 3 --tp 0,1 >/dev/null 2>/tmp/m_err.txt
  grep -E "^\[mem\]" /tmp/m_err.txt | sort -u | sed 's/^/  /'
done
