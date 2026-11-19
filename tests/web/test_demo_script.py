"""The one-command demo transcript reads the wire shapes the page renders.

Owner: C | Spec: docs/ROADMAP.md G2, issue #5 (D's W3 assignment, item 1)

`scripts/demo.py` is the grader-facing reproduce command, so its output must stay honest: no
citation the page cannot attribute is counted as renderable, the 200-code-point budget is checked
against the payload rather than assumed, and the refusal state is reported separately from an empty
result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import demo


def test_transcript_renders_only_citations_that_carry_a_url() -> None:
    report = demo.transcript()

    assert report["retrieve"]["status"] == 200
    assert report["retrieve"]["excerpts"] > 0
    assert report["retrieve"]["dropped_unattributable"] == 0
    for item in report["retrieve"]["rendered"]:
        assert item["url"].startswith("https://")
        assert item["publisher"]
        assert item["source_type"] in {"official", "media", "metadata"}


def _excerpt(*, url: str = "https://example.com/a", publisher: str = "示例媒体") -> dict:
    chunk_id = "20261105-demo::A::1"
    return {
        "chunk_id": chunk_id,
        "quote": "第二季将于 2027 年春季开播。",
        "citation": {
            "source_type": "media",
            "publisher": publisher,
            "title": "示例报道",
            "url": url,
            "published_at": "2026-11-01T00:00:00Z",
            "chunk_id": chunk_id,
        },
    }


@pytest.mark.parametrize(
    "dropped",
    [
        _excerpt(url="example.com/a"),  # scheme-less URL: the page's `new URL()` check rejects it
        _excerpt(url="ftp://example.com/a"),
        _excerpt(publisher="   "),
    ],
)
def test_a_citation_the_page_would_drop_is_not_counted(dropped: dict) -> None:
    """The transcript counts what the page draws, so a dropped citation cannot inflate the tally."""
    assert demo.is_renderable_citation(dropped["citation"]) is False

    rendered, dropped_count = demo.split_renderable_excerpts([_excerpt(), dropped])

    assert len(rendered) == 1
    assert dropped_count == 1


def test_transcript_quotes_stay_inside_the_shared_budget() -> None:
    report = demo.transcript()

    assert report["retrieve"]["quote_budget_ok"] is True
    assert all(
        item["quote_codepoints"] <= demo.MAX_QUOTE_CODEPOINTS
        for item in report["retrieve"]["rendered"]
    )


def test_transcript_reports_answer_and_refusal_as_distinct_states() -> None:
    answered = demo.transcript()["ask"]
    refused = demo.transcript(question="这个作品的制作人住址在哪里？")["ask"]  # noqa: RUF001

    assert answered["status"] == 200
    assert answered["abstained"] is False
    assert answered["citation_count"] > 0

    assert refused["status"] == 200
    assert refused["abstained"] is True
    assert refused["abstain_reason"]
    assert refused["citation_count"] == 0


def test_a_non_positive_top_k_is_rejected_before_the_request() -> None:
    assert demo.main(["--top-k", "0"]) == 2


@pytest.mark.parametrize("top_k", [1, 5])
def test_transcript_honours_top_k(top_k: int) -> None:
    assert demo.transcript(top_k=top_k)["retrieve"]["excerpts"] == top_k
