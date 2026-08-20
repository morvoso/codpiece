#!/usr/bin/env bash
# M2 bench payload: single-GPU CUDA correctness + decode speed, tandem vs
# llama.cpp b10423, Qwen3.5-0.8B-BF16. Runs INSIDE the locked bench window
# (bench-window.sh) — prod is stopped, GPUs are ours. Everything pins to one
# physical GPU via CUDA_VISIBLE_DEVICES so the comparison is apples-to-apples.
set -u

REPO=$HOME/llm/tandem/codpiece
GPU_SEL=${GPU_SEL:-1}
MODEL_C=/models/qwen35-small/Qwen3.5-0.8B-BF16.gguf

tdm() { # tandem (CUDA build) in the builder container
  docker run --rm --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all -e "CUDA_VISIBLE_DEVICES=$GPU_SEL" \
    -v "$REPO:/src" -v "$HOME/llm/models:/models" -v "$HOME/llm/tandem:/work" \
    --entrypoint /src/target/release/tandem \
    tandem-builder "$@"
}

oracle() { # llama-completion (b10423 CUDA) — same GPU
  docker run --rm --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all -e "CUDA_VISIBLE_DEVICES=$GPU_SEL" \
    -e LD_LIBRARY_PATH=/work/llama.cpp-b10423/build/bin \
    -v "$HOME/llm/tandem:/work" -v "$HOME/llm/models:/models" \
    --entrypoint /work/llama.cpp-b10423/build/bin/llama-completion \
    nvidia/cuda:13.0.0-devel-ubuntu24.04 "$@"
}

PROMPT="<|im_start|>user
The capital of France is<|im_end|>
<|im_start|>assistant
"

echo "== [1/4] tandem selftest on CUDA (session vs stateless, on-GPU) =="
tdm selftest "$MODEL_C" --gpu 0

echo "== [2/4] tandem ppl on CUDA, 4 chunks =="
echo "   CPU reference: [1]10.3754,[2]16.5382,[3]16.4713,[4]16.2159"
tdm ppl "$MODEL_C" -f /work/wiki.test.raw -c 512 --chunks 4 --gpu 0 2>/dev/null

echo "== [3/4] decode speed, 256 tokens, 2 reps each =="
for r in 1 2; do
  echo "-- tandem rep $r"
  tdm gen "$MODEL_C" -p "$PROMPT" -n 256 --ignore-eos --gpu 0 2>&1 >/dev/null | grep -E "prefill:"
done
for r in 1 2; do
  echo "-- oracle rep $r"
  oracle -m "$MODEL_C" -p "The capital of France is" -n 256 --temp 0 -ngl 999 \
    --no-warmup --ignore-eos --no-display-prompt 2>&1 >/dev/null \
    | grep -E "prompt eval time|[^a-z]eval time"
done

echo "== [4/4] 64-token generation parity on CUDA =="
T=$(tdm gen "$MODEL_C" -p "$PROMPT" -n 64 --gpu 0 2>/dev/null | sed 's/<|im_end|>//')
O=$(oracle -m "$MODEL_C" -p "The capital of France is" -n 64 --temp 0 -ngl 999 \
      --no-warmup --no-display-prompt 2>/dev/null | sed -e 's/> EOF by user//' -e 's/<|im_end|>//')
norm() { printf '%s' "$1" | sed -e 's/[[:space:]]*$//' | awk 'NF{b=0} !NF{b++} {l[NR]=$0; last=NR} END{while(last>0&&l[last]=="")last--; for(i=1;i<=last;i++)print l[i]}'; }
if [ "$(norm "$T")" == "$(norm "$O")" ]; then
  echo "GPU GEN PARITY: IDENTICAL"
else
  echo "GPU GEN PARITY: DIVERGENCE"
  echo "--- tandem ---"; norm "$T" | head -6
  echo "--- oracle ---"; norm "$O" | head -6
fi
