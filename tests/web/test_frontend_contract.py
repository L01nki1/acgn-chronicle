"""Run the front-end contract rules under Node's built-in test runner.

The rules that must not regress - code-point clipping, "no URL, no render", the cache tuple and the
empty-vs-refused split - live in `static/contract.mjs`, deliberately free of DOM dependencies.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "tests" / "web" / "contract.test.mjs"
APP_FILE = ROOT / "src" / "acgn_rag" / "web" / "static" / "app.mjs"
NODE = shutil.which("node")


def test_page_reads_the_snapshot_from_the_request_body() -> None:
    """The render wrapper stores the wire body under `body`; `request.snapshot_id` is undefined.

    A browser run rendered `快照 undefined` in the result meta while this test was missing.
    """
    source = APP_FILE.read_text(encoding="utf-8")
    assert "request.snapshot_id" not in source
    assert "request.body.snapshot_id" in source


@pytest.mark.skipif(
    NODE is None, reason="Node.js is not installed; skipping the .mjs contract tests"
)
def test_contract_module_passes_its_node_suite() -> None:
    completed = subprocess.run(
        [NODE, "--test", str(TEST_FILE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.skipif(
    NODE is None, reason="Node.js is not installed; skipping the .mjs contract tests"
)
def test_contract_module_exports_the_client_guardrails() -> None:
    completed = subprocess.run(
        [NODE, "--input-type=module", "-e", "import('./src/acgn_rag/web/static/contract.mjs')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
