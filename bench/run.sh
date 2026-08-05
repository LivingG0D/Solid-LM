#!/bin/bash
# Start an engine, wait until it is genuinely ready, measure, stop it.
# One engine resident at a time — 24 GB shared means two models would make
# every number a lie.
run_bench() {
  local label=$1 port=$2; shift 2
  pkill -f "port $port" 2>/dev/null; sleep 1
  "$@" > "/tmp/bench-$port.log" 2>&1 &
  local pid=$!
  local ok=0
  for _ in $(seq 1 400); do
    kill -0 $pid 2>/dev/null || break
    # llama-server answers /health with 503 {"status":"loading model"} while it
    # pages in, so a bare "did curl succeed" check calls it ready far too early.
    if curl -s -m 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    # MLX has no /health; a 200 on /v1/models is its ready signal.
    if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:$port/v1/models" 2>/dev/null)" = "200" ]; then ok=1; break; fi
    sleep 1
  done
  if [ $ok -eq 1 ]; then
    printf '%-36s %s\n' "$label" "$(python3 tps.py "$port" 3 2>&1 | tail -1)"
  else
    printf '%-36s FAILED to start — %s\n' "$label" \
      "$(grep -iE 'error|failed|unsupported' /tmp/bench-$port.log 2>/dev/null | tail -1 | cut -c1-90)"
  fi
  kill -TERM $pid 2>/dev/null
  for _ in $(seq 1 80); do kill -0 $pid 2>/dev/null || break; sleep 0.5; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 3
}
