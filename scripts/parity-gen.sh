#!/usr/bin/env bash
# Generation-parity harness: tandem vs llama-completion (b10423), greedy,
# same chat-template-wrapped prompt, N tokens. Runs ON llm-host.
# Usage: parity-gen.sh [n_tokens]
# Env: TANDEM (binary), MODEL (gguf), ORACLE_BIN_DIR (llama.cpp build/bin)
set -u
TANDEM=${TANDEM:-/tmp/tandem}
MODEL=${MODEL:-$HOME/llm/models/qwen35-small/Qwen3.5-0.8B-BF16.gguf}
ORACLE_BIN=${ORACLE_BIN_DIR:-$HOME/llm/tandem/llama.cpp-b10423/build/bin}
N=${1:-64}

# oracle model path as seen inside the container
MODEL_IN_CONTAINER=${MODEL/#$HOME\/llm\/models/\/models}

oracle_gen() { # $1 = raw prompt
  docker run --rm --runtime nvidia -e CUDA_VISIBLE_DEVICES= \
    -e "LD_LIBRARY_PATH=/work/llama.cpp-b10423/build/bin" \
    -v "$HOME/llm/tandem:/work" -v "$HOME/llm/models:/models" \
    --entrypoint /work/llama.cpp-b10423/build/bin/llama-completion \
    nvidia/cuda:13.0.0-devel-ubuntu24.04 \
    -m "$MODEL_IN_CONTAINER" -p "$1" -n "$N" --temp 0 -t 8 -ngl 0 \
    -fa "${ORACLE_FA:-off}" --no-warmup --no-display-prompt 2>/dev/null \
  | sed -e 's/> EOF by user//' -e 's/<|im_end|>//'
}

tandem_gen() { # $1 = raw prompt
  local wrapped="<|im_start|>user
$1<|im_end|>
<|im_start|>assistant
"
  # TANDEM_SUBCMD=run (stateless reference) or gen (session/engine path)
  nice -n 19 "$TANDEM" "${TANDEM_SUBCMD:-run}" "$MODEL" -p "$wrapped" -n "$N" -t 8 2>/dev/null \
  | sed -e 's/<|im_end|>//'
}

norm() { # strip trailing whitespace / blank tail lines for comparison
  sed -e 's/[[:space:]]*$//' | awk 'NF {blank=0} !NF {blank++} {lines[NR]=$0; last=NR} END {while (last>0 && lines[last]=="") last--; for (i=1;i<=last;i++) print lines[i]}'
}

PASS=0; FAIL=0
run_case() { # $1 = label, $2 = prompt
  local o t
  o=$(oracle_gen "$2" | norm)
  t=$(tandem_gen "$2" | norm)
  if [ "$o" == "$t" ]; then
    echo "PASS [$1] ($(printf '%s' "$t" | wc -c) chars identical)"
    PASS=$((PASS+1))
  else
    echo "FAIL [$1]"
    echo "--- oracle ---"; printf '%s\n' "$o" | head -8
    echo "--- tandem ---"; printf '%s\n' "$t" | head -8
    FAIL=$((FAIL+1))
  fi
}

run_case "prose"        "The capital of France is"
run_case "code"         "Write a Python function that reverses a linked list."
run_case "multilingual" "日本の首都はどこですか？簡単に答えてください。"

echo "=================================================="
echo "gen-parity: $PASS pass, $FAIL fail (n=$N tokens, greedy, temp 0)"
[ "$FAIL" -eq 0 ]
