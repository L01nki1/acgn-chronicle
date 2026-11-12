"""Run the front-end contract rules under Node's built-in test runner.

The rules that must not regress - code-point clipping, "no URL, no render", the cache tuple and the
empty-vs-refused split - live in `static/contract.mjs`, deliberately free of DOM dependencies. The
request/response coordination rule lives in `static/queue.mjs` for the same reason.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NODE_TEST_FILES = (
    ROOT / "tests" / "web" / "contract.test.mjs",
    ROOT / "tests" / "web" / "queue.test.mjs",
)
APP_FILE = ROOT / "src" / "acgn_rag" / "web" / "static" / "app.mjs"
NODE = shutil.which("node")


def test_page_reads_the_snapshot_from_the_request_body() -> None:
    """The render wrapper stores the wire body under `body`; `request.snapshot_id` is undefined.

    A browser run rendered `快照 undefined` in the result meta while this test was missing.
    """
    source = APP_FILE.read_text(encoding="utf-8")
    assert "request.snapshot_id" not in source
    assert "request.body.snapshot_id" in source


def test_an_action_during_an_in_flight_request_is_queued_not_dropped() -> None:
    """A busy flag that returns early left the new question next to the old response.

    The page must queue the newest intent and say so, and it must mark the visible response as
    belonging to the previous request once the form changes.
    """
    source = APP_FILE.read_text(encoding="utf-8")
    assert "if (state.busy)" not in source
    assert "queryQueue.begin()" in source
    assert "queryQueue.queue(runQuery)" in source
    assert "markFormChanged" in source


def test_citation_url_renders_as_a_safe_link() -> None:
    """G2: the citation URL is a real anchor, not text plus a "coming later" note.

    `isRenderableCitation` has already rejected non-http(s) URLs before a card is built, so the
    anchor may carry the original URL; `noopener noreferrer` keeps the opened page out of
    `window.opener`.
    """
    source = APP_FILE.read_text(encoding="utf-8")
    assert "跳转将在 G2 启用" not in source
    assert 'element("a", {' in source
    assert "href: citation.url" in source
    assert 'target: "_blank"' in source
    assert 'rel: "noopener noreferrer"' in source


@pytest.mark.skipif(
    NODE is None, reason="Node.js is not installed; skipping the .mjs contract tests"
)
@pytest.mark.parametrize("test_file", NODE_TEST_FILES, ids=lambda path: path.name)
def test_static_module_passes_its_node_suite(test_file: Path) -> None:
    completed = subprocess.run(
        [NODE, "--test", str(test_file)],
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
