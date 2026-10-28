"""CLI wrapper: export or verify the KnowledgePort JSON Schema.

Usage:
    python scripts/export_schema.py            # write contracts/knowledge_port.schema.json
    python scripts/export_schema.py --check    # verify only, exit code 1 on drift (used by CI)

The logic lives in acgn_rag/contracts_export.py; this file only owns the exit code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.contracts_export import check, write


def main() -> int:
    parser = argparse.ArgumentParser(description="Export or verify the KnowledgePort JSON Schema")
    parser.add_argument("--check", action="store_true", help="verify only, do not write")
    args = parser.parse_args()

    if args.check:
        ok, message = check()
        print(message)
        return 0 if ok else 1

    path = write()
    print(f"WROTE: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
