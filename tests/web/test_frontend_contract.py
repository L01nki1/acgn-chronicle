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
NODE = shutil.which("node")


@pytest.mark.skipif(
    NODE is None, reason="Node.js is not installed; skipping the .mjs contract tests"
)
def test_contract_module_passes_its_node_suite() -> None:
    completed = subprocess.run(
        [NODE, "--test", str(TEST_FILE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
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
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
