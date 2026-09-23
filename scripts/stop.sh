#!/usr/bin/env bash
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$REPO/.logs"
for name in litellm gateway rag_watcher openwebui; do
  pidfile="$LOGS/$name.pid"
  if [[ -f "$pidfile" ]]; then
    pid=$(cat "$pidfile")
    if kill "$pid" 2>/dev/null; then echo "stopped $name ($pid)"; else echo "$name not running"; fi
    rm -f "$pidfile"
  fi
done
echo "done"
