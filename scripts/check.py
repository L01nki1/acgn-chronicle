"""Run the four CI gates locally, in the same order CI runs them.

Usage:
    python scripts/check.py          # gate only: stop at the first failure
    python scripts/check.py --fix    # apply ruff fixes first, then gate

Why this exists: CI runs four separate steps and stops at the first red one, but people tend to
run `pytest` (or `ruff check`) and push. A green local pytest says nothing about formatting or
about the contract schema, and the failure that motivated this script was exactly that gap -
`ruff check` green, `ruff format --check` red, CI stalled before the tests ever ran.

Owner: D | Spec: docs/ONBOARDING.md 4.1
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(label: str, args: list[str], hint: str) -> bool:
    print(f"\n== {label} ==", flush=True)
    print("   $ " + " ".join(args), flush=True)
    # Fixed argv built from this file, never from user input; no shell involved.
    completed = subprocess.run(args, cwd=ROOT, check=False)
    if completed.returncode == 0:
        print("   OK", flush=True)
        return True
    print(f"   FAILED - fix: {hint}", file=sys.stderr, flush=True)
    return False


def main() -> int:
    do_fix = "--fix" in sys.argv[1:]

    if do_fix:
        # Auto-fixable gates first; the gates then run unchanged, so --fix can never mask a failure.
        _run("ruff check --fix", [PY, "-m", "ruff", "check", "--fix", "."], "-")
        _run("ruff format", [PY, "-m", "ruff", "format", "."], "-")

    gates = [
        ("1/4 lint", [PY, "-m", "ruff", "check", "."], "fix the reported lines by hand"),
        (
            "2/4 format",
            [PY, "-m", "ruff", "format", "--check", "."],
            "python scripts/check.py --fix   (or: ruff format .)",
        ),
        (
            "3/4 contract schema",
            [PY, str(ROOT / "scripts" / "export_schema.py"), "--check"],
            "python scripts/export_schema.py   and commit contracts/knowledge_port.schema.json",
        ),
        (
            "4/4 tests",
            [PY, "-m", "pytest", "-q"],
            "read the failing test; never mark it network to hide it",
        ),
    ]

    for label, args, hint in gates:
        if not _run(label, args, hint):
            # CI stops at the first red step, so this does too: later gates would report noise.
            print("\nStopped at the first failing gate, exactly like CI.", file=sys.stderr)
            return 1

    print("\nAll four gates passed - safe to push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
