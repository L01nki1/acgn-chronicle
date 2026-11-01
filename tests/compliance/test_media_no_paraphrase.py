"""Snapshot the C4 policy text required by Issue #4; runtime prompt inclusion is W2 work."""

from __future__ import annotations

from acgn_rag.core.ground.policy import MEDIA_SOURCE_RULE


def test_media_source_rule_snapshot() -> None:
    assert MEDIA_SOURCE_RULE == (
        "If a source's `source_type` is `media`, extract only verifiable facts and figures; "
        "never paraphrase its commentary, reviews or speculation. If the source is `official`, "
        "official statements and facts may be restated."
    )
