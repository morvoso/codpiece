#!/usr/bin/env bash
# On the Python prompt codpiece's speculative round diverges from its own single-token
# greedy decoding, identically on the cached and rebuilt paths. The suspicion is that
# batched verification is not bit-identical to one-token-at-a-time decoding, so a
# near-tied argmax can flip — inherent to speculative decoding rather than a codpiece
# bug. Test that claim against the incumbent: run llama.cpp greedy with and without
# its own MTP speculation and diff.
# Runs INSIDE bench-window.sh.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
RAW="Write a Python function that merges two sorted lists."
PROMPT=$'<|im_start|>user\n'"$RAW"$'<|im_end|>\n<|im_start|>assistant\n'

serve() { # $1 = extra args
  docker rm -f codpiece-ref-server >/dev/null 2>&1
  docker run -d --name codpiece-ref-server --runtime nvidia --ipc=host --shm-size=8g \
    -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 \
    -v "$HOME/llm/models:/models" -p 8031:8080 \
    ghcr.io/ggml-org/llama.cpp:server-cuda-b10423 \
    --host 0.0.0.0 --port 8080 -m "$M" -c 4096 -ngl 999 --fit off --no-warmup \
    -sm tensor -ts 50,50 -fa on --cache-type-k f16 --cache-type-v f16 -np 1 --jinja \
    $1 >/dev/null 2>&1
  for _ in $(seq 1 100); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:8031/health || true)" = "200" ] && return 0
    docker ps --format '{{.Names}}' | grep -qx codpiece-ref-server || { echo "  server died"; return 1; }
    sleep 3
  done
  echo "  server not healthy"; return 1
}

ask() { # $1 = outfile
  python3 - "$PROMPT" "$1" <<'PY'
import json, sys, urllib.request
prompt, out = sys.argv[1], sys.argv[2]
body = json.dumps({"prompt": prompt, "max_tokens": 128, "temperature": 0,
                   "top_k": 1, "n_predict": 128}).encode()
req = urllib.request.Request("http://localhost:8031/v1/completions", body,
                             {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.load(r)
open(out, "w").write(d["choices"][0]["text"])
print("  %.2f tok/s (%s predicted)" % (d.get("timings", {}).get("predicted_per_second", 0),
                                       d.get("timings", {}).get("predicted_n", 0)))
PY
}

echo "== llama.cpp, NO speculation, greedy =="
serve "" && ask /tmp/lc_plain.txt

echo "== llama.cpp, prod MTP speculation, greedy =="
serve "--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75" && ask /tmp/lc_spec.txt

docker rm -f codpiece-ref-server >/dev/null 2>&1

echo "== does llama.cpp's own speculation change its greedy output? =="
if diff -q /tmp/lc_plain.txt /tmp/lc_spec.txt >/dev/null 2>&1; then
  echo "  IDENTICAL — llama.cpp speculation is token-exact here"
else
  echo "  DIFFERS — llama.cpp speculation also changes greedy output"
  diff /tmp/lc_plain.txt /tmp/lc_spec.txt | head -12
fi
