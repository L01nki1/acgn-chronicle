"""Point git at the tracked hook directory (`.githooks/`). Run once per clone.

Usage:
    python scripts/install_hooks.py           # install
    python scripts/install_hooks.py --check   # report whether hooks are active (CI-free)

Why `core.hooksPath` instead of copying files into `.git/hooks`:
  - `.git/` is not versioned, so copied hooks can never be updated by `git pull`. A member would
    silently keep running a hook from three weeks ago.
  - `.githooks/` is tracked and reviewed like any other file, so the gate improves for everyone
    at the same time.

The setting is local to the clone (not committed), because `core.hooksPath` is a repository
configuration, not a tracked file. That is the price of doing this without a framework.

Owner: D | Spec: docs/ONBOARDING.md 4.2
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = ".githooks"
HOOKS = ("pre-commit", "pre-push")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _current() -> str:
    return _git("config", "--get", "core.hooksPath").stdout.strip()


def main() -> int:
    if not (ROOT / HOOKS_DIR).is_dir():
        print(f"ERROR: {HOOKS_DIR}/ is missing", file=sys.stderr)
        return 1

    missing = [name for name in HOOKS if not (ROOT / HOOKS_DIR / name).is_file()]
    if missing:
        print(f"ERROR: missing hook file(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    if "--check" in sys.argv[1:]:
        current = _current()
        if current == HOOKS_DIR:
            print(f"OK: core.hooksPath = {HOOKS_DIR}")
            return 0
        print(f"NOT INSTALLED: core.hooksPath = {current or '(unset)'}")
        print("Run: python scripts/install_hooks.py")
        return 1

    result = _git("config", "core.hooksPath", HOOKS_DIR)
    if result.returncode != 0:
        print(f"ERROR: git config failed: {result.stderr.strip()}", file=sys.stderr)
        return 1

    print(f"core.hooksPath = {HOOKS_DIR}")
    print(
        "  pre-commit : lint + format on staged Python files, contract check when ports.py changes"
    )
    print("  pre-push   : all four CI gates (python scripts/check.py)")
    print("\nThis setting is local to this clone; every new clone needs the same one command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
