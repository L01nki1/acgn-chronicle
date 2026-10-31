"""W1 grounded-prompt policy text; the W2 prompt builder must include it verbatim."""

from __future__ import annotations

MEDIA_SOURCE_RULE = (
    "If a source's `source_type` is `media`, extract only verifiable facts and figures; "
    "never paraphrase its commentary, reviews or speculation. If the source is `official`, "
    "official statements and facts may be restated."
)

__all__ = ["MEDIA_SOURCE_RULE"]
