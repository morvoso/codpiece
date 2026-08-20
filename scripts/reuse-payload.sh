#!/usr/bin/env bash
# Session reuse on the 27B: what does the next turn of a long conversation cost?
#
# Turn 1 prefills ~32K tokens cold — that wall time is also the baseline for what turn 2
# would cost without reuse, since its prompt is the same size to within a hundred
# tokens. Turns 2 and 3 extend the conversation the way a coding chat does; with reuse
# the engine prefills only each turn's suffix. llama.cpp's session cache is what gives
# prod its 1.3 s revisit in the box's own records; this is codpiece's equivalent.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
NAME=codpiece-reuse
head -c 132000 $HOME/llm/codpiece/wiki.test.raw > /tmp/reuse_seed.txt
tail -c 132000 $HOME/llm/codpiece/wiki.test.raw > /tmp/reuse_seed2.txt

docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
  -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
  -p 8020:8020 --entrypoint /src/target/release/codpiece codpiece-builder \
  serve "$M" --host 0.0.0.0 --port 8020 -c ${CTX:-32768} --tp 0,1 >/dev/null
for _ in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8020/health || true)" = "200" ] && break
  docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "server exited"; docker logs $NAME 2>&1 | tail -5; exit 1; }
  sleep 3
done

python3 <<'PY'
import json, time, urllib.request
def post(obj, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:8020/v1/completions",
                                 json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    return json.load(urllib.request.urlopen(req, timeout=timeout)), time.time() - t0

convoA = open("/tmp/reuse_seed.txt").read()
convoB = open("/tmp/reuse_seed2.txt").read()

# conversation A, turn 1: cold
convoA += "\n\nSummarise the last paragraph in one sentence:"
r, wall = post({"prompt": convoA, "max_tokens": 48})
print(f"A turn 1 (cold)   : {r['timings']['prompt_n']:6d} tok, wall {wall:6.1f}s")
convoA += r["choices"][0]["text"]

# conversation B: forces A out of the live session and into host RAM
convoB += "\n\nSummarise the last paragraph in one sentence:"
r, wall = post({"prompt": convoB, "max_tokens": 48})
print(f"B turn 1 (switch) : {r['timings']['prompt_n']:6d} tok, wall {wall:6.1f}s")

# back to A: must restore from host RAM, not re-prefill
convoA += "\n\nNow say it in exactly three words:"
r, wall = post({"prompt": convoA, "max_tokens": 32})
print(f"A turn 2 (RESTORE): {r['timings']['prompt_n']:6d} tok, wall {wall:6.1f}s")
print(f"reply: {r['choices'][0]['text'][:80]!r}")
PY
docker logs $NAME 2>&1 | grep -E "session slot|pool capped" | head -2
docker rm -f $NAME >/dev/null 2>&1
