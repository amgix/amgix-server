#!/usr/bin/env python3
"""
Emit FastAPI OpenAPI JSON to stdout (no HTTP server).

Requires:
  - PYTHONPATH set to the amgix-server repository root (parent of `src/`).
  - API dependencies installed (see amgix-server/src/api/requirements.txt), e.g. in a venv.

Optional:
  - AMGIX_VERSION — written into the spec info.version (default: v1.2.0-dash).
"""
from __future__ import annotations

import json
import os
import sys

_SERVER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

os.environ.setdefault("AMGIX_VERSION", "v1.2.0-dash")


def main() -> None:
    try:
        from src.api.main import app
    except ImportError as e:
        print(
            "Failed to import the API app. Set PYTHONPATH to the amgix-server root "
            f"(expected: {_SERVER_ROOT}) and install src/api/requirements.txt.\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    spec = app.openapi()
    json.dump(spec, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
