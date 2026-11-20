"""The run records what code actually ran, including an uncommitted worktree.

`git rev-parse HEAD` alone cannot say that: a run started from a dirty tree files its results
under the parent commit. That happened in #116 - one adopted-answer-line run is recorded against
the commit that preceded the rule it was testing - which makes it impossible for a reviewer to
tell from the archive whether to trust the provenance.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval  # noqa: E402


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_clean_tree_records_the_bare_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if "rev-parse" in command:
            return _completed("abc1234\n")
        return _completed("")

    monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

    assert run_eval._git_sha() == "abc1234"
    # The dirty probe must be scoped to the directories that can change behaviour.
    assert calls[1][-3:] == ["src", "scripts", "tests"]


def test_modified_source_is_marked_with_a_count(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **_kwargs):
        if "rev-parse" in command:
            return _completed("abc1234\n")
        return _completed(" M src/acgn_rag/core/ground/policy.py\n M scripts/run_eval.py\n")

    monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

    assert run_eval._git_sha() == "abc1234+dirty:2"


def test_untracked_scratch_files_do_not_invalidate_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--untracked-files=no` is what keeps a stray local script out of the identity."""

    def fake_run(command, **_kwargs):
        if "rev-parse" in command:
            return _completed("abc1234\n")
        assert "--untracked-files=no" in command
        return _completed("")

    monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

    assert run_eval._git_sha() == "abc1234"


def test_a_failed_dirty_probe_never_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Losing the provenance detail is acceptable; aborting a long run over it is not."""

    def fake_run(command, **_kwargs):
        if "rev-parse" in command:
            return _completed("abc1234\n")
        raise OSError("git vanished")

    monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

    assert run_eval._git_sha() == "abc1234+dirty:unknown"


def test_no_git_at_all_reports_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **_kwargs):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(run_eval.subprocess, "run", fake_run)

    assert run_eval._git_sha() == "unknown"
