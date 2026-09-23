#!/usr/bin/env bash
# Start the Jarvis stack on the Mac mini. Logs go to <repo>/.logs so they can be read from the repo.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${JARVIS_ENV_FILE:-$HOME/.jarvis.env}"
LOGS="$REPO/.logs"
mkdir -p "$LOGS/traces"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy .env.example there and fill it in."; exit 1
fi
set -a; source "$ENV_FILE"; set +a
: "${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY must be set in $ENV_FILE}"
: "${DATABASE_URL:?DATABASE_URL must be set in $ENV_FILE}"

cd "$REPO"
echo "Starting Jarvis from $REPO"

echo "  LiteLLM on :4000"
nohup uv run litellm --config "$REPO/configs/litellm.yaml" --port 4000 > "$LOGS/litellm.log" 2>&1 &
echo $! > "$LOGS/litellm.pid"
sleep 8

echo "  Gateway on :4001"
nohup uv run python -u -m jarvis.gateway > "$LOGS/gateway.log" 2>&1 &
echo $! > "$LOGS/gateway.pid"

echo "  RAG watcher"
nohup uv run python -u -m jarvis.rag_watcher > "$LOGS/rag_watcher.log" 2>&1 &
echo $! > "$LOGS/rag_watcher.pid"

# Open WebUI is installed separately (it pins its own heavy dependency set). Use it if present.
if [[ -x "$HOME/venvs/ai/bin/open-webui" ]]; then
  echo "  Open WebUI on :8080"
  nohup "$HOME/venvs/ai/bin/open-webui" serve > "$LOGS/openwebui.log" 2>&1 &
  echo $! > "$LOGS/openwebui.pid"
elif command -v open-webui >/dev/null; then
  echo "  Open WebUI on :8080"
  nohup open-webui serve > "$LOGS/openwebui.log" 2>&1 &
  echo $! > "$LOGS/openwebui.pid"
else
  echo "  Open WebUI not found, skipping"
fi

echo; echo "Model health check (LiteLLM):"
sleep 15
for model in classifier reasoning coding general; do
  response=$(curl -s -X POST http://localhost:4000/v1/chat/completions \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with a short greeting.\"}], \"max_tokens\": 1500}" \
    --max-time 120 2>/dev/null || true)
  content=$(printf '%s' "$response" | uv run python -c '
import sys, json, re
try:
    c = json.load(sys.stdin)["choices"][0]["message"]["content"]
    c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL)
    c = re.sub(r"<think>.*$", "", c, flags=re.DOTALL)
    print(c.strip()[:50])
except Exception:
    print("")
' 2>/dev/null)
  if [[ -n "$content" ]]; then echo "  ok      $model"; else echo "  EMPTY   $model (hung, corrupted, or out of tokens while thinking)"; fi
done

echo
echo "Jarvis is up."
echo "  LiteLLM UI   http://localhost:4000/ui"
echo "  Gateway      http://localhost:4001/v1   (health: /health)"
echo "  Open WebUI   http://localhost:8080"
echo "  Logs         $LOGS/*.log     traces: $LOGS/traces/"
