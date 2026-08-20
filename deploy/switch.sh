#!/bin/sh
# Switch the LLM engine on this box. Only one can run: each needs ~46 of the
# 48 GiB of VRAM. Both serve on port 8020 with model id "qwen3.8-27b", so no
# client (qwen CLI / pi / omp) needs any change when switching.
#
#   ./switch.sh vllm       fast engine   ~76 prose / ~90 code tok/s, 180K ctx, fp8 KV
#   ./switch.sh llamacpp   accurate      ~51 prose / ~67 code tok/s, 196K ctx, f16 KV
#   ./switch.sh codpiece   fast+warm     ~68 code tok/s, warm multi-turn, NO VISION
#   ./switch.sh status     show what is running
#
# Both stacks are declarative files in this directory; switching never edits
# them, so "switch back" always returns the EXACT config, byte for byte.
set -u
cd /home/morvoso/llm

LC=docker-compose.llamacpp.yml
VL=docker-compose.vllm.yml
TD=docker-compose.codpiece.yml

wait_ready() {
    what=$1
    i=0
    while [ $i -lt 90 ]; do
        if curl -sf -m 5 http://127.0.0.1:8020/v1/models >/dev/null 2>&1; then
            echo "$what is READY on :8020 after $((i*5))s"
            curl -s -m 5 http://127.0.0.1:8020/v1/models \
                | head -c 200 | sed 's/^/    /'
            echo
            return 0
        fi
        i=$((i+1)); sleep 5
    done
    echo "TIMEOUT waiting for $what — check: docker compose -f $2 logs --tail 40"
    return 1
}

case "${1:-status}" in
  vllm)
      echo "stopping llama.cpp / codpiece ..."
      docker compose -f $LC down >/dev/null 2>&1
      docker compose -f $TD down >/dev/null 2>&1
      echo "starting vLLM ..."
      docker compose -f $VL up -d || exit 2
      wait_ready "vLLM" $VL
      ;;
  llamacpp|llama|llama.cpp)
      echo "stopping vLLM / codpiece ..."
      docker compose -f $VL down >/dev/null 2>&1
      docker compose -f $TD down >/dev/null 2>&1
      echo "starting llama.cpp ..."
      docker compose -f $LC up -d || exit 2
      wait_ready "llama.cpp" $LC
      ;;
  codpiece)
      # codpiece: the from-scratch engine (~/llm/codpiece/codpiece). Faster on
      # text/code and warm on multi-turn, but NO VISION -- switch.sh llamacpp
      # restores vision. Rollback is always byte-exact; this script never edits
      # the compose files.
      echo "stopping llama.cpp / vLLM ..."
      docker compose -f $LC down >/dev/null 2>&1
      docker compose -f $VL down >/dev/null 2>&1
      echo "starting codpiece ..."
      docker compose -f $TD up -d || exit 2
      wait_ready "codpiece" $TD
      ;;
  status)
      for c in llama-qwen vllm-qwen vllm-proxy codpiece-qwen; do
          s=$(docker inspect -f '{{.State.Status}}' $c 2>/dev/null || echo absent)
          printf '  %-12s %s\n' "$c" "$s"
      done
      echo "  --- :8020 ---"
      curl -s -m 5 http://127.0.0.1:8020/v1/models | head -c 200 | sed 's/^/  /' \
          || echo "  not responding"
      echo
      nvidia-smi --query-gpu=index,memory.used,temperature.gpu,power.draw \
          --format=csv,noheader | sed 's/^/  gpu /'
      ;;
  *)
      echo "usage: $0 [vllm|llamacpp|codpiece|status]"; exit 1
      ;;
esac
