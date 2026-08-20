#!/usr/bin/env bash
# Head-to-head at matched settings: llama.cpp's own server running production's
# MTP config, versus codpiece's MTP speculative decoding. Same model, same GPUs,
# same context, same prompt. Runs INSIDE a locked bench window.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
RAW="Explain in three sentences why the sky is blue."
PROMPT=$'<|im_start|>user\n'"$RAW"$'<|im_end|>\n<|im_start|>assistant\n'
NPRED=${NPRED:-96}

echo "== llama.cpp b10423 server, prod MTP config (-sm tensor, draft-mtp 3 / 0.75) =="
docker rm -f codpiece-ref-server >/dev/null 2>&1
# ipc: host + a real /dev/shm are what make -sm tensor work in a container:
# NCCL's shared-memory transport aborts in ncclGroupEnd() with Docker's
# default 64 MiB. Prod's compose sets exactly this (docker-compose.llamacpp.yml).
docker run -d --name codpiece-ref-server --runtime nvidia \
  --ipc=host --shm-size=8g \
  -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 \
  -v "$HOME/llm/models:/models" -p 8031:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b10423 \
  --host 0.0.0.0 --port 8080 -m "$M" -c 4096 -ngl 999 --fit off --no-warmup \
  -sm tensor -ts 50,50 -fa on --cache-type-k f16 --cache-type-v f16 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 --jinja \
  >/dev/null 2>&1

# Give up early if the container died: waiting out the full timeout for a
# crashed server holds the bench window (and therefore production) down for
# no reason.
ready=0
for _ in $(seq 1 100); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:8031/health || true)
    if [ "$code" = "200" ]; then ready=1; break; fi
    if ! docker ps --format '{{.Names}}' | grep -qx codpiece-ref-server; then
        echo "  reference server exited during startup:"
        docker logs codpiece-ref-server 2>&1 | grep -iE "error|abort|assert" | head -4
        break
    fi
    sleep 3
done
if [ "$ready" != "1" ]; then
    echo "  reference server did not become healthy; logs:"
    docker logs codpiece-ref-server 2>&1 | tail -5
else
    python3 - "$PROMPT" "$NPRED" <<'PY'
import json, sys, urllib.request
prompt, npred = sys.argv[1], int(sys.argv[2])
for rep in (1, 2, 3):
    body = json.dumps({"prompt": prompt, "max_tokens": npred,
                       "temperature": 0, "n_predict": npred}).encode()
    req = urllib.request.Request("http://localhost:8031/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        t = json.load(r).get("timings", {})
    print("  llama.cpp rep%d: %.2f tok/s (%s predicted)"
          % (rep, t.get("predicted_per_second", 0), t.get("predicted_n", 0)))
PY
fi
docker rm -f codpiece-ref-server >/dev/null 2>&1
sleep 5

codpiece() { # $@ = subcommand and flags
    timeout 1800 docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 \
      -v "$HOME/llm/codpiece/codpiece:/src" -v "$HOME/llm/models:/models" \
      --entrypoint /src/target/release/codpiece codpiece-builder "$@" \
      >/dev/null 2>/tmp/hh.txt
    grep -oE 'decode:.*' /tmp/hh.txt
}

echo "== codpiece, MTP speculative depth 3 (separate draft graphs) =="
for rep in 1 2 3; do
    echo "  codpiece spec-3   rep$rep: $(codpiece spec "$M" -p "$PROMPT" -n "$NPRED" -c 4096 --spec 3 --tp 0,1)"
done

echo "== codpiece, fused verify+draft chain, cached graphs =="
for D in 2 3; do
    for rep in 1 2 3; do
        echo "  codpiece fused-d$D rep$rep: $(codpiece fused "$M" -p "$PROMPT" -n "$NPRED" -c 4096 --depth $D --tp 0,1 --path cached)"
    done
done
