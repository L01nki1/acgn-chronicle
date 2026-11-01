"""Run the G1a query page locally: `uv run python -m acgn_rag.web`.

Owner: C | Spec: docs/ROADMAP.md G1a
"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ACGN Chronicle G1a query page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="reload on source changes")
    args = parser.parse_args()
    uvicorn.run(
        "acgn_rag.web.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
