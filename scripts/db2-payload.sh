#!/usr/bin/env bash
# Run the box's own benchmark against tandem, unmodified.
#
# db2.py hard-codes http://127.0.0.1:8020 and has no host flag, so tandem takes that
# port for the duration — prod is stopped inside the window anyway. The point is that
# the script is not adapted to tandem in any way: whatever it needs, tandem provides,
# or the run fails and that is the finding.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
NAME=tandem-db2
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
  -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models \
  -p 8020:8020 --entrypoint /src/target/release/tandem tandem-builder \
  serve "$M" --host 0.0.0.0 --port 8020 -c 131072 --tp 0,1 --max-tokens 400 >/dev/null

for _ in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8020/health || true)" = "200" ] && break
  docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "server exited:"; docker logs $NAME 2>&1 | tail -15; exit 1; }
  sleep 3
done
echo "tandem up on 8020"

cd $HOME/llm/bench && timeout 3000 python3 db2.py --label tandem --depths ${DEPTHS:-0,32000} --reps ${REPS:-1} --max-tokens 400
rc=$?
echo "db2 rc=$rc"
docker rm -f $NAME >/dev/null 2>&1
