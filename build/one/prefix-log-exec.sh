#!/bin/bash
# Prefix each output line, then exec into the real process so supervisord's SIGTERM
# reaches uvicorn / encoder / rabbit / qdrant (same PID after exec chain).
# Usage: prefix-log-exec.sh <prefix> <command> [args...]

set -euo pipefail

PREFIX="$1"
shift
export PREFIX

exec > >(stdbuf -oL bash -c 'while IFS= read -r line || [ -n "$line" ]; do echo "[$PREFIX] $line"; done')
exec 2>&1
exec "$@"
