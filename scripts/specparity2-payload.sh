#!/usr/bin/env bash
# Accuracy check, and the one that matters most for this engine.
#
# codpiece's speculative sampling at temperature is meant to be distribution-preserving:
# the draft head proposes its argmax, so its distribution is a point mass, and accepting
# with probability p(x0) — otherwise drawing from p with x0 removed — emits exactly p.
# A unit test asserts that over 200k draws of a toy distribution. This asserts it on the
# real 27B, which is where an implementation bug would actually live.
#
# The number only means something against a null: two independent samples from the SAME
# configuration differ too, and at a few thousand tokens spread over a thousand token
# types they differ a lot. So this measures spec-vs-spec (different seeds) as well as
# spec-vs-no-spec, and compares the two.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
NAME=codpiece-parity
N=${N:-200}
TOKENS=${TOKENS:-20}

start() { # $1 = depth (0 = no speculation)
  docker rm -f $NAME >/dev/null 2>&1
  docker run -d --name $NAME --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
    -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
    -p 8020:8020 --entrypoint /src/target/release/codpiece codpiece-builder \
    serve "$M" --host 0.0.0.0 --port 8020 -c 8192 --tp 0,1 --depth "$1" >/dev/null
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8020/health || true)" = "200" ] && return 0
    docker ps --format '{{.Names}}' | grep -qx $NAME || return 1
    sleep 3
  done
  return 1
}

collect() { # $1 = outfile, $2 = seed base
  OUT="$1" BASE="$2" NGEN="$N" NTOK="$TOKENS" python3 <<'PY'
import collections, json, os, urllib.request
out, base = os.environ["OUT"], int(os.environ["BASE"])
n, toks = int(os.environ["NGEN"]), int(os.environ["NTOK"])
counts = collections.Counter()
def post(path, obj, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:8020" + path,
                                 json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
for i in range(n):
    text = post("/v1/completions", {"prompt": "The weather today is",
                                    "max_tokens": toks, "temperature": 1.0,
                                    "seed": base + i})["choices"][0]["text"]
    # tokenise through the server so both sides count the same units
    counts.update(post("/tokenize", {"content": text, "add_special": False}, 60)["tokens"])
json.dump(counts, open(out, "w"))
print(f"  {sum(counts.values())} tokens over {n} generations, {len(counts)} distinct")
PY
}

echo "== speculation ON (depth 3), sample A =="
start 3 || { echo "server failed"; docker logs $NAME 2>&1 | tail -5; exit 1; }
collect /tmp/par_specA.json 1000
echo "== speculation ON (depth 3), sample B — the null =="
collect /tmp/par_specB.json 5000
echo "== speculation OFF =="
start 0 || { echo "server failed"; docker logs $NAME 2>&1 | tail -5; exit 1; }
collect /tmp/par_plain.json 1000
docker rm -f $NAME >/dev/null 2>&1

python3 <<'PY'
import json
def load(p): return {int(k): v for k, v in json.load(open(p)).items()}
A, B, C = load("/tmp/par_specA.json"), load("/tmp/par_specB.json"), load("/tmp/par_plain.json")
def tv(x, y):
    nx, ny = sum(x.values()), sum(y.values())
    return 0.5 * sum(abs(x.get(k, 0)/nx - y.get(k, 0)/ny) for k in set(x) | set(y))
null, test = tv(A, B), tv(A, C)
print(f"\nnull  (spec vs spec, different seeds): {null:.4f}")
print(f"test  (spec vs no speculation)      : {test:.4f}")
print(f"ratio test/null                     : {test/null:.3f}")
print()
print("=> no detectable difference: speculation preserves the distribution"
      if test <= null * 1.15 else
      "=> test exceeds the null; speculation is shifting the distribution <<<")
PY
