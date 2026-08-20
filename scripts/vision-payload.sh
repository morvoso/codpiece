#!/usr/bin/env bash
# Vision end-to-end gate: the same newspaper image through llama.cpp b10423
# (the production vision reference) and codpiece, both asked for the main
# headline. The image is mtmd's test-1.jpeg whose headline is known
# ("MEN WALK ON MOON" — the moon-landing front page), so the check greps for
# "walk on moon" in both answers. Runs INSIDE bench-window.sh on port 8031.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
MM=/models/qwen38/mmproj-BF16.gguf
IMG=$HOME/llm/codpiece/test-1.jpeg
PORT=8031
NAME=codpiece-vision-test

python3 - "$IMG" > /tmp/vision_body.json <<'PY'
import base64, json, sys
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
body = {
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        {"type": "text", "text": "What is the main headline of this newspaper page? Reply with the headline text only."},
    ]}],
    "max_tokens": 512, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
json.dump(body, open("/tmp/vision_body.json", "w"))
PY

ask() { # $1 = label; server must be healthy on $PORT
  curl -s --max-time 600 -H 'Content-Type: application/json' \
    --data-binary @/tmp/vision_body.json "http://127.0.0.1:$PORT/v1/chat/completions" \
    | python3 -c '
import json,sys
r=json.load(sys.stdin)
if "choices" not in r: print("ERROR:", json.dumps(r)[:300]); sys.exit(1)
m=r["choices"][0]["message"]
t=r.get("timings",{})
print("reply:", repr(m.get("content","")[:200]))
if t: print("timing: %.1f tok/s prefill %.1f tok/s decode, %d prompt tokens" % (
    t.get("prompt_per_second",0), t.get("predicted_per_second",0), t.get("prompt_n",0)))'
}

wait_up() {
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/health || true)" = "200" ] && return 0
    docker ps --format '{{.Names}}' | grep -qx $NAME || return 1
    sleep 3
  done
  return 1
}

echo "########## llama.cpp b10423 reference ##########"
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v $HOME/llm/models:/models -p $PORT:$PORT \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b10423 \
  -m "$M" --mmproj "$MM" --host 0.0.0.0 --port $PORT -c 8192 \
  -sm tensor -ts 50,50 -fa on --image-min-tokens 1024 >/dev/null
wait_up || { echo "llama.cpp server failed"; docker logs $NAME 2>&1 | tail -8; exit 1; }
ask llamacpp | tee /tmp/vision_ref.txt

echo "########## codpiece ##########"
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
  -v $HOME/llm/codpiece/codpiece:/src -v $HOME/llm/models:/models \
  -p $PORT:$PORT --entrypoint /src/target/release/codpiece codpiece-builder \
  serve "$M" --host 0.0.0.0 --port $PORT -c 8192 --tp 0,1 \
  --mmproj "$MM" --mmproj-gpu 0 >/dev/null
wait_up || { echo "codpiece server failed"; docker logs $NAME 2>&1 | tail -12; exit 1; }
ask codpiece | tee /tmp/vision_cp.txt

echo "== follow-up on the same image (session reuse path) =="
python3 - > /tmp/vision_body2.json <<'PY'
import base64, json
b64 = base64.b64encode(open("/home/morvoso/llm/codpiece/test-1.jpeg", "rb").read()).decode()
body = {
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        {"type": "text", "text": "Who is the author credited under the headline? Name only."},
    ]}],
    "max_tokens": 512, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
json.dump(body, open("/tmp/vision_body2.json", "w"))
PY
curl -s --max-time 600 -H 'Content-Type: application/json' \
  --data-binary @/tmp/vision_body2.json "http://127.0.0.1:$PORT/v1/chat/completions" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print("reply:", repr(r["choices"][0]["message"]["content"][:160]))' \
  | tee /tmp/vision_cp2.txt

echo "== text-only request still healthy on the same server =="
curl -s --max-time 300 -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":8}' \
  "http://127.0.0.1:$PORT/v1/completions" | head -c 200; echo

docker rm -f $NAME >/dev/null 2>&1
docker logs $NAME >/dev/null 2>&1 || true

echo "########## GATE ##########"
ok=1
grep -qi "walk on moon" /tmp/vision_ref.txt && echo "  llama.cpp read the headline" || { echo "  llama.cpp MISSED the headline <<<"; ok=0; }
grep -qi "walk on moon" /tmp/vision_cp.txt && echo "  codpiece read the headline" || { echo "  codpiece MISSED the headline <<<"; ok=0; }
grep -qi "wilford" /tmp/vision_cp2.txt && echo "  codpiece read the byline" || echo "  codpiece byline differs (informational)"
[ "$ok" = 1 ] && echo "  VISION GATE: PASS" || echo "  VISION GATE: FAIL <<<"
[ "$ok" = 1 ]
