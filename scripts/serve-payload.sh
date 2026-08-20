#!/usr/bin/env bash
# The server, end to end against the 27B.
#
# The gate that matters is the last one: a greedy request through HTTP must produce the
# same text as `tandem gen` with the same prompt. Everything between the socket and the
# sampler is new code, and that check exercises all of it at once.
# Runs INSIDE bench-window.sh.
set -u
M=/models/qwen38/Qwen3.8-27B-UD-Q8_K_XL.gguf
NAME=tandem-serve-test
PORT=8031

# The reference has to be taken before the server starts: both hold the whole model,
# and two copies do not fit on these cards.
PROMPT="<|im_start|>user
Explain in three sentences why the sky is blue.<|im_end|>
<|im_start|>assistant
"
docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
  -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models \
  --entrypoint /src/target/release/tandem tandem-builder \
  gen "$M" -p "$PROMPT" -n 64 -c 8192 --tp 0,1 >/tmp/srv_cli.txt 2>/dev/null
echo "cli reference: $(wc -c < /tmp/srv_cli.txt) bytes"

docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_P2P_DISABLE=1 \
  -v $HOME/llm/tandem/codpiece:/src -v $HOME/llm/models:/models \
  -p $PORT:$PORT --entrypoint /src/target/release/tandem tandem-builder \
  serve "$M" --host 0.0.0.0 --port $PORT -c 8192 --tp 0,1 >/dev/null

ready=0
for _ in $(seq 1 90); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:$PORT/health || true)" = "200" ]; then
    ready=1; break
  fi
  docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "server exited:"; docker logs $NAME 2>&1 | tail -15; break; }
  sleep 3
done
if [ "$ready" != 1 ]; then
  echo "server never became healthy"; docker logs $NAME 2>&1 | tail -20; docker rm -f $NAME >/dev/null 2>&1; exit 1
fi
echo "== server up =="
echo "  /health   : $(curl -s http://localhost:$PORT/health)"
echo "  /v1/models: $(curl -s http://localhost:$PORT/v1/models | head -c 160)"
echo "  /slots    : $(curl -s http://localhost:$PORT/slots | head -c 160)"

post() { curl -s --max-time 300 -H 'Content-Type: application/json' -d "$2" "http://localhost:$PORT$1"; }

echo "== /v1/completions, greedy, non-streaming =="
R=$(post /v1/completions '{"prompt":"The capital of France is","max_tokens":16}')
echo "  text    : $(echo "$R" | python3 -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))')"
echo "  timings : $(echo "$R" | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("%.1f tok/s" % t["predicted_per_second"])')"

echo "== /v1/chat/completions, greedy, non-streaming =="
R=$(post /v1/chat/completions '{"messages":[{"role":"user","content":"Reply with exactly the word: pineapple"}],"max_tokens":40}')
echo "  content : $(echo "$R" | python3 -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["message"]["content"][:120]))')"

echo "== streaming (SSE) =="
N=$(curl -s --max-time 300 -N -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count: one two three"}],"max_tokens":24,"stream":true}' \
  http://localhost:$PORT/v1/chat/completions | grep -c '^data: ')
echo "  received $N SSE events (last should be [DONE])"

echo "== sampling is reproducible by seed =="
B='{"prompt":"Once upon a time","max_tokens":24,"temperature":0.8,"top_p":0.9,"seed":%d}'
A1=$(post /v1/completions "$(printf "$B" 1234)" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])')
A2=$(post /v1/completions "$(printf "$B" 1234)" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])')
A3=$(post /v1/completions "$(printf "$B" 4321)" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])')
[ "$A1" = "$A2" ] && echo "  same seed : REPRODUCIBLE" || echo "  same seed : NOT reproducible <<<"
[ "$A1" = "$A3" ] && echo "  diff seed : identical <<<" || echo "  diff seed : differs, as it should"

echo "== speculation is on for both greedy and sampled requests =="
for BODY in '{"prompt":"The capital of France is","max_tokens":64}' \
            '{"prompt":"The capital of France is","max_tokens":64,"temperature":0.7,"top_p":0.9,"seed":5}'; do
  R=$(post /v1/completions "$BODY")
  echo "  $(echo "$BODY" | head -c 60)..."
  echo "    $(echo "$R" | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; a=t.get("acceptance"); print("%.1f tok/s, acceptance %s" % (t["predicted_per_second"], "n/a" if a is None else "%.3f"%a))')"
done

echo "== error handling =="
echo "  bad json  : $(curl -s -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d '{oops' http://localhost:$PORT/v1/completions)"
echo "  no route  : $(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/nope)"

echo "== GATE: greedy over HTTP must equal the CLI =="
# A here-string would append a newline to the prompt and change its tokenisation, so
# the prompt goes through the environment byte for byte.
PROMPT="$PROMPT" python3 -c 'import json,os; print(json.dumps({"prompt": os.environ["PROMPT"], "max_tokens": 64}))' > /tmp/srv_body.json
curl -s --max-time 300 -H 'Content-Type: application/json' --data-binary @/tmp/srv_body.json \
  http://localhost:$PORT/v1/completions \
  | python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin)["choices"][0]["text"])' > /tmp/srv_http.txt
# `tandem gen` prints its output with a trailing newline; the HTTP body has none.
printf '%s' "$(cat /tmp/srv_cli.txt)"  > /tmp/srv_cli_n.txt
printf '%s' "$(cat /tmp/srv_http.txt)" > /tmp/srv_http_n.txt
if diff -q /tmp/srv_http_n.txt /tmp/srv_cli_n.txt >/dev/null; then
  echo "  IDENTICAL"
else
  echo "  DIFFERS <<<"
  echo "  http $(wc -c < /tmp/srv_http_n.txt) bytes, cli $(wc -c < /tmp/srv_cli_n.txt) bytes"
  diff <(fold -w80 /tmp/srv_cli_n.txt) <(fold -w80 /tmp/srv_http_n.txt) | head -8
fi

docker rm -f $NAME >/dev/null 2>&1
