#!/usr/bin/env bash
# Long-context parity: a multi-thousand-token prompt exercises chunked
# prefill (512/chunk), position handling at depth, KV writes across chunk
# boundaries, and decode-graph bucket rollovers — none of which the short
# prompts touch. Compares tandem vs llama-completion (b10423), greedy.
#
# Usage: parity-longctx.sh [prompt_tokens] [gen_tokens]
# Env: TANDEM (binary), MODEL (gguf), GPU_SEL (empty = CPU), ORACLE_FA
set -u
TANDEM=${TANDEM:-/tmp/tandem}
MODEL=${MODEL:-$HOME/llm/models/qwen35-small/Qwen3.5-0.8B-BF16.gguf}
CORPUS=${CORPUS:-$HOME/llm/tandem/wiki.test.raw}
NPROMPT=${1:-2000}
NGEN=${2:-32}
ORACLE_FA=${ORACLE_FA:-on}
MODEL_IN_CONTAINER=${MODEL/#$HOME\/llm\/models/\/models}

# Deterministic prompt: first NPROMPT words of the corpus, no special tokens.
PROMPT_FILE=/tmp/longctx-prompt.txt
head -c 200000 "$CORPUS" | tr -s ' \n' ' ' | cut -d' ' -f1-"$NPROMPT" > "$PROMPT_FILE"
NTOK=$("$TANDEM" tokenize "$MODEL" < "$PROMPT_FILE" 2>&1 >/dev/null | grep -oE '[0-9]+')
echo "prompt: $(wc -c < "$PROMPT_FILE") bytes / $NTOK tokens; generating $NGEN"

gpu_args=()
[ -n "${GPU_SEL:-}" ] && gpu_args=(--gpu 0)

T=$(nice -n 19 "$TANDEM" gen "$MODEL" -p "$(cat "$PROMPT_FILE")" -n "$NGEN" \
      -c 8192 -t 8 "${gpu_args[@]}" 2>/dev/null)

O=$(docker run --rm --runtime nvidia \
      -e "CUDA_VISIBLE_DEVICES=${GPU_SEL:-}" \
      -e "LD_LIBRARY_PATH=/work/llama.cpp-b10423/build/bin" \
      -v "$HOME/llm/tandem:/work" -v "$HOME/llm/models:/models" -v /tmp:/host-tmp \
      --entrypoint /work/llama.cpp-b10423/build/bin/llama-completion \
      nvidia/cuda:13.0.0-devel-ubuntu24.04 \
      -m "$MODEL_IN_CONTAINER" -f /host-tmp/longctx-prompt.txt -n "$NGEN" \
      --temp 0 -t 8 -c 8192 -fa "$ORACLE_FA" -no-cnv \
      -ngl "$([ -n "${GPU_SEL:-}" ] && echo 999 || echo 0)" \
      --no-warmup --no-display-prompt --no-escape 2>/dev/null \
    | sed -e 's/> EOF by user//')

norm() { sed -e 's/<|im_end|>//' -e 's/[[:space:]]*$//' | awk 'NF{b=0} !NF{b++} {l[NR]=$0; last=NR} END{while(last>0&&l[last]=="")last--; for(i=1;i<=last;i++)print l[i]}'; }
TN=$(printf '%s' "$T" | norm)
ON=$(printf '%s' "$O" | norm)

if [ "$TN" == "$ON" ]; then
  echo "LONG-CTX PARITY: IDENTICAL ($(printf '%s' "$TN" | wc -c) chars)"
  exit 0
else
  echo "LONG-CTX PARITY: DIVERGENCE"
  echo "--- tandem ---"; printf '%s\n' "$TN" | head -5
  echo "--- oracle ---"; printf '%s\n' "$ON" | head -5
  exit 1
fi
