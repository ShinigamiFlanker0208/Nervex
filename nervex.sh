#!/usr/bin/env bash
# Boot Nervex in one command: Neo4j -> API -> UI, then print the URLs.
# Ctrl-C stops everything this script started (and only that).
#
#   ./nervex.sh           API + UI
#   ./nervex.sh --no-ui   API only
#
# Env overrides: API_HOST (127.0.0.1), API_PORT (8000), UI_PORT (5173)
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd); LOGS="$ROOT/.nervex-logs"; mkdir -p "$LOGS"
API_HOST=${API_HOST:-127.0.0.1}; API_PORT=${API_PORT:-8000}; UI_PORT=${UI_PORT:-5173}
WITH_UI=1; [ "${1:-}" = "--no-ui" ] && WITH_UI=0

if [ -t 1 ]; then B=$'\e[1m'; D=$'\e[2m'; R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; N=$'\e[0m'; else B= D= R= G= Y= N=; fi
say() { printf "  %s%-8s%s %s\n" "$D" "$1" "$N" "$2"; }
die() { printf "  %s✗%s %s\n" "$R" "$N" "$1"; exit 1; }

PIDS=()          # process-group leaders we started
STOPPING=0
cleanup() {
  [ "$STOPPING" = 1 ] && return; STOPPING=1
  [ ${#PIDS[@]} -eq 0 ] && return
  echo; say stop "shutting down"
  # each server runs in its own process group (setsid), so killing the group also kills
  # the real server under the uv / node wrappers, not just the wrapper
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    alive=0; for p in "${PIDS[@]}"; do kill -0 -- "-$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break; sleep .5
  done
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
  say done "stopped"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
wait_for() {    # url, seconds, name, pid
  local t=0
  until curl -sf -m 2 -o /dev/null "$1"; do
    kill -0 "$4" 2>/dev/null || die "$3 exited during startup. Last lines of its log:\n$(tail -n 15 "$5")"
    sleep 1; t=$((t + 1))
    if [ "$t" -ge "$2" ]; then die "$3 did not answer within $2s (log: $5)"; fi
  done
  return 0
}
start() {       # logfile, command... ; runs in a new session, stdin closed
  setsid "${@:2}" </dev/null >"$1" 2>&1 &
  PIDS+=("$!")
}

printf "\n  %sNervex%s  %sa reading machine for research papers%s\n\n" "$B" "$N" "$D" "$N"
command -v uv     >/dev/null || die "uv not found: https://docs.astral.sh/uv/"
command -v docker >/dev/null || die "docker not found"
command -v curl   >/dev/null || die "curl not found"
if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; say env "created .env from .env.example ${Y}(add your LLM key)${N}"; fi

# 1. Neo4j: reuse the container if it exists
if docker ps --format '{{.Names}}' | grep -qx nervex-neo4j; then say neo4j "already running"
elif docker ps -a --format '{{.Names}}' | grep -qx nervex-neo4j; then docker start nervex-neo4j >/dev/null && say neo4j "started"
else docker run -d --name nervex-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5 >/dev/null && say neo4j "created"
fi
docker image inspect soulgraph-runtime:latest >/dev/null 2>&1 \
  || say sandbox "${Y}image missing${N}: docker build -f sandbox/Dockerfile.sandbox -t soulgraph-runtime:latest ."

# 2. API
port_busy "$API_PORT" && die "port $API_PORT is in use (Nervex already running? set API_PORT=...)"
uv sync -q || die "uv sync failed"
API_HOST=$API_HOST API_PORT=$API_PORT start "$LOGS/api.log" uv run python main.py
say api "starting  ${D}log: .nervex-logs/api.log${N}"
wait_for "http://$API_HOST:$API_PORT/" 120 "API" "${PIDS[-1]}" "$LOGS/api.log"

# 3. UI
if [ $WITH_UI = 1 ]; then
  command -v npm >/dev/null || die "npm not found (Node 18+), or use --no-ui"
  port_busy "$UI_PORT" && die "port $UI_PORT is in use (set UI_PORT=...)"
  if [ ! -d frontend/node_modules ]; then say ui "installing packages (first run)"; (cd frontend && npm install --silent) || die "npm install failed"; fi
  start "$LOGS/ui.log" "$ROOT/frontend/node_modules/.bin/vite" "$ROOT/frontend" --host 127.0.0.1 --port "$UI_PORT" --strictPort
  say ui "starting  ${D}log: .nervex-logs/ui.log${N}"
  wait_for "http://127.0.0.1:$UI_PORT/" 60 "UI" "${PIDS[-1]}" "$LOGS/ui.log"
fi

H=$(curl -s -m 5 "http://$API_HOST:$API_PORT/")
st=$(printf '%s' "$H" | sed -n 's/.*"status":"\([a-z]*\)".*/\1/p')
llm=$(printf '%s' "$H" | sed -n 's/.*"provider":"\([^"]*\)".*/\1/p')
msg=$(printf '%s' "$H" | sed -n 's/.*"message":"\([^"]*\)".*/\1/p')
echo
[ $WITH_UI = 1 ] && printf "  %s➜  App%s       http://127.0.0.1:%s\n" "$G$B" "$N" "$UI_PORT"
printf "  %s➜  API%s       http://%s:%s/api/v1\n" "$G" "$N" "$API_HOST" "$API_PORT"
printf "  %s➜  API docs%s  http://%s:%s/docs\n" "$G" "$N" "$API_HOST" "$API_PORT"
printf "  %s➜  Neo4j%s     http://localhost:7474\n\n" "$G" "$N"
if [ "$st" = online ]; then say status "${G}online${N} · llm: ${llm:-?}"; else say status "${Y}${st:-unknown}${N} · ${msg}"; fi
say stop "press Ctrl-C"; echo

# stay in the foreground; if any server dies, say so and stop the rest
while :; do
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null || die "a server stopped unexpectedly; see .nervex-logs/"; done
  sleep 2
done
