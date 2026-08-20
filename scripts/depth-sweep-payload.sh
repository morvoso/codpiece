#!/usr/bin/env bash
# Does declining to draft when the model is unsure pay?
#
# Production runs llama.cpp with `--spec-draft-p-min 0.75` and reports 0.78 acceptance
# at temperature 1.0; codpiece always drafts K and reports 0.35-0.42. Under the rejection
# rule an unlikely draft is rejected in proportion to how unlikely it is, so the chain
# steps behind it are spent for nothing — the gate is meant to stop paying for them.
#
# Sweep it on the box's own benchmark, at the depth where codpiece trails.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
NAME=codpiece-gate

for DEPTH in ${SPEC_DEPTHS:-1 2 3}; do
  docker rm -f $NAME >/dev/null 2>&1
  docker run -d --name $NAME --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    -p 8020:8020 --entrypoint /src/target/release/codpiece codpiece-builder \
    serve "$M" --host 0.0.0.0 --port 8020 -c 131072 --tp 0,1 --max-tokens 400 \
    --depth $DEPTH >/dev/null
  ok=0
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8020/health || true)" = "200" ] && { ok=1; break; }
    docker ps --format '{{.Names}}' | grep -qx $NAME || break
    sleep 3
  done
  [ "$ok" = 1 ] || { echo "depth $DEPTH: server failed"; docker logs $NAME 2>&1 | tail -5; continue; }
  echo "--- depth $DEPTH ---"
  ( cd $HOME/llm/bench && timeout 1800 python3 db2.py --label d$DEPTH --depths ${DEPTHS:-32000} --reps ${REPS:-2} --max-tokens 400 2>&1 | grep -E '^\s+\{' )
done
docker rm -f $NAME >/dev/null 2>&1
