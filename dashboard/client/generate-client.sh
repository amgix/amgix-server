#!/usr/bin/env bash
# Generate the TypeScript client from the FastAPI app (no Docker, no HTTP server).
# OpenAPI is produced by importing src.api.main:app and calling app.openapi().
# Requires: Python 3 with API deps on PATH (e.g. source amgix-server/venv/bin/activate),
# and openapi-generator-cli (global npm install or npx @openapitools/openapi-generator-cli).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SERVER_ROOT="$(cd "$ROOT/../.." && pwd)"
export PYTHONPATH="$SERVER_ROOT"
export AMGIX_VERSION="${AMGIX_VERSION:-1.2.0-dash}"
PACKAGE_VERSION="${PACKAGE_VERSION:-$AMGIX_VERSION}"

OPENAPI_31_TMP="$(mktemp "${TMPDIR:-/tmp}/amgix-openapi-31.XXXXXX")"
OPENAPI_30_LOCAL="$(mktemp "${TMPDIR:-/tmp}/amgix-openapi-30.XXXXXX")"

cleanup() {
  rm -f "${OPENAPI_31_TMP:-}" "${OPENAPI_30_LOCAL:-}"
}
trap cleanup EXIT

echo "Exporting OpenAPI from FastAPI (PYTHONPATH=${SERVER_ROOT}) ..."
python3 "${ROOT}/export_openapi.py" >"$OPENAPI_31_TMP"

echo "Downgrading spec (OpenAPI 3.1 → 3.0-compatible) ..."
python3 "${ROOT}/openapi-downgrade.py" "$OPENAPI_31_TMP" -o "$OPENAPI_30_LOCAL" -f json

if [[ ! -s "$OPENAPI_30_LOCAL" ]]; then
  echo "error: downgrade produced empty or missing output" >&2
  exit 1
fi

echo "Package version: ${PACKAGE_VERSION}"
echo "Setting client.yaml packageVersion to ${PACKAGE_VERSION} ..."
sed -i.bak "s/^  packageVersion: .*/  packageVersion: ${PACKAGE_VERSION}/" client.yaml
rm -f client.yaml.bak

if [[ -f "${ROOT}/src/package.json" ]]; then
  echo "Setting src/package.json version to ${PACKAGE_VERSION} ..."
  sed -i.bak "s|^  \"version\": \".*\"|  \"version\": \"${PACKAGE_VERSION}\"|" "${ROOT}/src/package.json"
  rm -f "${ROOT}/src/package.json.bak"
fi

echo "Generating TypeScript client ..."
openapi-generator-cli generate -c client.yaml -i "$OPENAPI_30_LOCAL" --skip-validate-spec

echo "Done. Review changes under ${ROOT}/src and commit if correct."
