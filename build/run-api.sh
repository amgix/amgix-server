#!/bin/sh

# Use AMGIX_LOG_LEVEL for gunicorn, default to info
LOG_LEVEL=$(echo "${AMGIX_LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')

# gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8234 src.api.main:app \
#   --timeout 120 --graceful-timeout 30 --keep-alive 5 \
#   --log-level "${LOG_LEVEL}"

uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8234 \
    --loop uvloop \
    --http httptools \
    --workers 1 \
    --log-level "${LOG_LEVEL}"  