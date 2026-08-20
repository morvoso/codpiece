#!/usr/bin/env bash
# Locked GPU bench window for llm-host (SAFETY.md rules 4 + 11-14).
#
#   bench-window.sh <payload command...>
#
# Takes the sweep lock, verifies prod is idle, stops prod, runs the payload
# with a temperature watchdog, then ALWAYS restores prod and verifies health
# before releasing the lock. Payload stdout/stderr are teed to a log.
set -u

LOCK=/tmp/llama-sweep.lock
PROD_CONTAINER=llama-qwen
PROD_URL=http://localhost:8020
LOGDIR=$HOME/llm/codpiece/logs
TEMP_ABORT_C=83
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOGDIR/bench-$STAMP.log"

say() { echo "[bench-window] $*" | tee -a "$LOG"; }

exec 9>"$LOCK"
if ! flock -n 9; then
    say "ABORT: another sweep holds $LOCK"
    exit 1
fi

# prod must exist and be idle before we take it down
if ! docker ps --format '{{.Names}}' | grep -qx "$PROD_CONTAINER"; then
    say "ABORT: prod container $PROD_CONTAINER not running (nothing to stop; refusing odd state)"
    exit 1
fi
SLOTS=$(curl -s --max-time 5 "$PROD_URL/slots" || echo '[]')
if echo "$SLOTS" | grep -q '"is_processing": *true'; then
    say "ABORT: prod is mid-request (/slots busy)"
    exit 1
fi

restore() {
    say "restoring prod..."
    docker start "$PROD_CONTAINER" >/dev/null 2>&1
    for i in $(seq 1 90); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$PROD_URL/health" || true)
        if [ "$code" = "200" ]; then
            say "prod /health 200 after ${i}x2s"
            ok=$(curl -s --max-time 30 "$PROD_URL/v1/completions" \
                -H 'Content-Type: application/json' \
                -d '{"prompt":"ping","max_tokens":1}' | grep -c '"text"' || true)
            say "prod 1-token completion check: $([ "$ok" -ge 1 ] && echo OK || echo FAILED)"
            return
        fi
        sleep 2
    done
    say "WARNING: prod /health not 200 after 180s — INVESTIGATE (docker logs $PROD_CONTAINER)"
}

watchdog() { # $1 = payload pid to kill on thermal breach
    local pid=$1
    while kill -0 "$pid" 2>/dev/null; do
        line=$(nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader 2>/dev/null)
        echo "[temps $(date +%H:%M:%S)] $line" >> "$LOG"
        hot=$(echo "$line" | awk -F', ' -v t="$TEMP_ABORT_C" '$2+0 >= t {print $1}')
        if [ -n "$hot" ]; then
            echo "[watchdog] GPU $hot reached ${TEMP_ABORT_C}C — killing payload" | tee -a "$LOG"
            kill "$pid" 2>/dev/null
            return
        fi
        sleep 5
    done
}

say "stopping prod ($PROD_CONTAINER)"
docker stop "$PROD_CONTAINER" >>"$LOG" 2>&1
trap restore EXIT

say "payload: $*"
"$@" > >(tee -a "$LOG") 2>&1 &
PAYLOAD_PID=$!
watchdog "$PAYLOAD_PID" &
WATCHDOG_PID=$!

wait "$PAYLOAD_PID"
PAYLOAD_RC=$?
kill "$WATCHDOG_PID" 2>/dev/null

say "payload rc=$PAYLOAD_RC; log: $LOG"
# restore runs via trap
exit "$PAYLOAD_RC"
