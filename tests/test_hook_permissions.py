"""The tracked git hooks must stay executable and LF-only in the index.

Why this test exists: a hook file added from Windows can be recorded as mode 100644. Git on macOS
then refuses to run it ("permission denied") while `git commit` still succeeds in a clone that has
`core.hooksPath` unset - so the gate silently stops existing for five of the six members. Likewise a
CRLF shell script dies as "bad interpreter". Both are invisible until someone relies on the hook.

Owner: D | Spec: docs/ONBOARDING.md 4.2
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ("pre-commit", "pre-push")

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


def _index_modes() -> dict[str, str]:
    """Return {path: mode} for .githooks/ as recorded in the index."""
    completed = subprocess.run(
        ["git", "ls-files", "-s", "--", ".githooks"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    modes: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        # format: <mode> <object> <stage>\t<path>
        meta, _, path = line.partition("\t")
        modes[path] = meta.split()[0]
    return modes


def test_hooks_are_tracked() -> None:
    modes = _index_modes()
    for name in HOOKS:
        assert f".githooks/{name}" in modes, f".githooks/{name} is not tracked"


def test_hooks_are_executable_in_the_index() -> None:
    modes = _index_modes()
    for name in HOOKS:
        assert modes.get(f".githooks/{name}") == "100755", (
            f".githooks/{name} must be mode 100755 (it is {modes.get(f'.githooks/{name}')}). "
            f"Fix: git update-index --chmod=+x .githooks/{name}"
        )


def test_hooks_have_lf_endings_and_a_shebang() -> None:
    for name in HOOKS:
        raw = (ROOT / ".githooks" / name).read_bytes()
        assert b"\r\n" not in raw, f".githooks/{name} contains CRLF; macOS will fail to execute it"
        assert raw.startswith(b"#!"), f".githooks/{name} needs a shebang"
