"""L0 fact extraction: batched LLM calls, one flat schema, strict parse, degrade-never-drop.

Owner: F | Spec: 01 §4.4, RUNBOOK §3.6

Design constraints (RUNBOOK §3.6, measured on the pilot corpus):

  - Batches of <= 10 numbered announcements per LLM call; the number <-> chunk mapping lives in
    this code, never in the model's answer (a model asked to echo ids invents them, RUNBOOK §3).
  - One flat Pydantic model - no nesting, no unions - because deeply structured output is a known
    vendor failure mode. Strict parse plus 2 retries; after that the whole batch degrades to
    plain-text chunks: a document is never dropped because the LLM returned bad JSON.
  - Corporate/IR releases are filtered out BEFORE extraction (they cost calls and carry no ACGN
    facts). The category tag is ingest-side metadata only and never a FactUpdate field
    (interface issue #17, option 1).
  - max_tokens >= 1024 on every call (DSV4-flash reasoning-token quirk, RUNBOOK §3.4), and the
    system prompt is a single constant so provider-side prompt caching can act on it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field

from acgn_rag.config import LLMConfig
from acgn_rag.core.ports import FactUpdate

#: Announcements per LLM call (RUNBOOK §3.6: ~9.5k L0 items => ~950 calls, not 9,500).
BATCH_SIZE = 10
#: Floor for max_tokens on every call (RUNBOOK §3.4: DSV4-flash returns empty content below it).
MIN_MAX_TOKENS = 1024
#: Literal the prompt tells the model to use for values the text does not state.
UNKNOWN = "unknown"

#: One system prompt, identical within and across batches (RUNBOOK §3.6), so any provider-side
#: prompt caching can act on it. It names the flat schema once; per-batch counts stay in the
#: user prompt.
SYSTEM_PROMPT = (
    "You extract structured facts from ACGN industry announcements (anime, manga, games, "
    "novels, related merchandise and events). For every announcement you return exactly one "
    "flat JSON object with these fields: "
    '"title": string, the work or product the announcement is about; '
    '"release_date": string, the announced release or publication date, or "unknown"; '
    '"platforms": array of strings, the platforms or media it is released on, or an empty '
    "array; "
    '"event_type": string, the kind of announcement (release, update, event, collaboration, '
    'cancellation, and so on), or "unknown"; '
    '"relevance": number from 0 to 1, how concrete and checkable the announcement is '
    "(1.0 = a dated, concrete ACGN fact). "
    "Use only what the text states; never invent titles, dates or platforms; no nested objects "
    "and no extra keys. Respond with a JSON array holding one object per numbered announcement, "
    "in the same order, and nothing else."
)

# Corporate-finance keywords (RUNBOOK §3.6). "IR" is matched as a stand-alone Latin token so
# English words like "third" or "spirit" never trip the filter; the CJK terms are plain
# substring matches. 決算短信 is subsumed by 決算 but kept so the list visibly matches the
# RUNBOOK. This classifier runs BEFORE extraction: ~30-40 % of the Capcom volume never reaches
# the LLM at all.
IR_KEYWORDS: tuple[str, ...] = ("決算", "株主", "決算短信", "事業報告")
IR_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z])[Ii][Rr](?![A-Za-z])")

_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def is_ir_release(text: str) -> bool:
    """True when the text is a corporate/IR release rather than ACGN content.

    The result is recorded ingest-side as ``ingest_category="corporate"`` and is deliberately
    NOT a FactUpdate field: ``FactUpdate`` has fixed fields and ``official/corporate`` is not a
    SourceType (interface issue #17, option 1).
    """
    return any(keyword in text for keyword in IR_KEYWORDS) or bool(IR_TOKEN_PATTERN.search(text))


class AnnouncementChunk(BaseModel):
    """One announcement plus the provenance every FactUpdate row must carry (G1b)."""

    chunk_id: str
    text: str
    url: str
    publisher: str


class AnnouncementFacts(BaseModel):
    """Flat extraction schema - deliberately no nesting and no unions (RUNBOOK §3.6)."""

    title: str
    release_date: str | None = None
    platforms: list[str] = Field(default_factory=list)
    event_type: str | None = None
    relevance: float = Field(..., ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class FactBatch:
    """Outcome of one extraction batch; nothing is ever silently dropped.

    ``extracted[i]`` describes the i-th CANDIDATE chunk of the batch (the chunks that survived
    the IR prefilter), in their original order - pair them with their chunks via
    ``expand_to_fact_updates``.
    """

    extracted: list[AnnouncementFacts]
    degraded_chunk_ids: list[str]
    ir_skipped_chunk_ids: list[str]
    calls: int


class LLMClient(Protocol):
    """One completion call; implemented by the OpenAI adapter and by offline test fakes."""

    def complete(self, system: str, user: str, *, max_tokens: int) -> str: ...


def make_openai_client(config: LLMConfig, *, http_client: Any = None) -> LLMClient:
    """Build an OpenAI-compatible chat client from the extraction settings.

    The ``openai`` import stays inside this function so importing the module does not require
    the optional ``[llm]`` extra. CI installs that extra and tests the client with mocked HTTP;
    it makes no live API calls. Swapping the vendor stays a ``.env`` change (D5).

    ``http_client`` exists for tests: openai >= 3 ships its own HTTP stack (the ``httpx2``
    package), which respx's global patch does not intercept, so the respx-mocked vendor test
    injects ``httpx.Client(transport=MockTransport(respx_router.handler))`` here. Production
    calls leave it None and use the SDK default.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - CI runs never reach this branch
        raise ImportError(
            "the extraction client needs the [llm] extra: pip install -e '.[llm]'"
        ) from exc

    kwargs: dict[str, Any] = {
        "base_url": config.extract_base_url or None,
        "api_key": config.extract_api_key or None,
    }
    if http_client is not None:
        kwargs["http_client"] = http_client
    vendor = OpenAI(**kwargs)

    class _Adapter:
        """LLMClient over the vendor chat API; enforces the max_tokens floor at call time."""

        def complete(self, system: str, user: str, *, max_tokens: int) -> str:
            # Raise rather than clamp (decision, NOTES): a silently clamped budget hides the
            # misconfiguration behind a slower call, while the failure it causes - empty
            # content on a reasoning-token deployment (RUNBOOK §3.4) - is much harder to trace
            # back to config.
            if max_tokens < MIN_MAX_TOKENS:
                raise ValueError(
                    f"max_tokens={max_tokens} is below the {MIN_MAX_TOKENS} floor: "
                    "reasoning-token deployments (DSV4-flash) return empty content below it "
                    "(RUNBOOK §3.4)"
                )
            response = vendor.chat.completions.create(
                model=config.extract_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=config.temperature,
            )
            content = response.choices[0].message.content
            # Empty content (reasoning tokens ate the budget) flows through the normal
            # parse-fail -> retry -> degrade path instead of crashing the run.
            return content if content is not None else ""

    return _Adapter()


def build_batch_prompt(chunks: Sequence[AnnouncementChunk]) -> tuple[str, str]:
    """Number the announcements 1..n and demand n flat objects back.

    The system prompt is the module constant; the user prompt carries only the numbered texts
    and the count. The number <-> chunk mapping stays in code: the model is never asked to echo
    ids, because a model asked for ids invents them (RUNBOOK §3).
    """
    if not 1 <= len(chunks) <= BATCH_SIZE:
        raise ValueError(f"a batch holds 1..{BATCH_SIZE} announcements, got {len(chunks)}")
    numbered = "\n\n".join(f"{index}. {chunk.text}" for index, chunk in enumerate(chunks, 1))
    user = (
        f"Extract the facts for the {len(chunks)} numbered announcements below.\n\n"
        f"{numbered}\n\n"
        f"Respond with a JSON array of exactly {len(chunks)} objects, in this order. Each "
        "object has the flat fields title, release_date, platforms, event_type, relevance; use "
        'the string "unknown" for any value the announcement does not state. Output only the '
        "JSON array."
    )
    return SYSTEM_PROMPT, user


def extract_facts(
    chunks: Sequence[AnnouncementChunk],
    client: LLMClient,
    *,
    max_retries: int = 2,
    max_tokens: int = MIN_MAX_TOKENS,
) -> FactBatch:
    """Extract facts for one batch (<= BATCH_SIZE chunks) following the RUNBOOK §3.6 SOP.

    IR-prefiltered chunks never reach the client. On a parse or validation failure the WHOLE
    batch is retried ``max_retries`` times with the same prompt; after that every candidate
    chunk in the batch degrades to a plain-text chunk (listed in ``degraded_chunk_ids``) - a
    document is never dropped because the LLM returned bad JSON.
    """
    ir_skipped = [chunk.chunk_id for chunk in chunks if is_ir_release(chunk.text)]
    candidates = [chunk for chunk in chunks if not is_ir_release(chunk.text)]
    if not candidates:
        return FactBatch(
            extracted=[], degraded_chunk_ids=[], ir_skipped_chunk_ids=ir_skipped, calls=0
        )

    system, user = build_batch_prompt(candidates)
    calls = 0
    extracted: list[AnnouncementFacts] = []
    degraded: list[str] = []
    for _attempt in range(1 + max_retries):
        raw = client.complete(system, user, max_tokens=max_tokens)
        calls += 1
        try:
            # json.JSONDecodeError and pydantic.ValidationError both subclass ValueError, so a
            # bad payload and a schema violation take the same path: retry the whole batch.
            extracted = _parse_batch(raw, len(candidates))
            break
        except ValueError:
            extracted = []
    else:
        extracted = []
        degraded = [chunk.chunk_id for chunk in candidates]
    return FactBatch(
        extracted=extracted,
        degraded_chunk_ids=degraded,
        ir_skipped_chunk_ids=ir_skipped,
        calls=calls,
    )


def expand_to_fact_updates(
    batch_items: Sequence[AnnouncementFacts], chunks: Sequence[AnnouncementChunk]
) -> list[FactUpdate]:
    """Expand extracted facts into wire FactUpdate rows, pairing slot i with chunks[i].

    Only known values produce rows (``unknown`` became None at parse time); a platform list
    yields one row per platform. Every row carries chunk_id, url and publisher from its source
    chunk - attribution is structural, not a convention (G1b requirement).
    """
    if len(batch_items) != len(chunks):
        raise ValueError(
            f"{len(batch_items)} fact object(s) cannot be paired with {len(chunks)} chunk(s); "
            "facts align positionally with the candidate chunks of their batch"
        )
    rows: list[FactUpdate] = []
    for facts, chunk in zip(batch_items, chunks, strict=True):
        provenance = {
            "media": facts.title,
            "chunk_id": chunk.chunk_id,
            "url": chunk.url,
            "publisher": chunk.publisher,
        }
        if facts.release_date is not None:
            rows.append(FactUpdate(field="release_date", value=facts.release_date, **provenance))
        for platform in facts.platforms:
            rows.append(FactUpdate(field="platform", value=platform, **provenance))
        if facts.event_type is not None:
            rows.append(FactUpdate(field="event_type", value=facts.event_type, **provenance))
    return rows


def _strip_json_fence(text: str) -> str:
    """Tolerate a fenced ```json block: many chat models wrap the array despite the prompt."""
    fenced = _FENCED_BLOCK.search(text)
    return fenced.group(1).strip() if fenced else text.strip()


def _known_or_none(value: object) -> str | None:
    """Map the prompt's "unknown" literal (and blanks) to None so no FactUpdate carries it."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and value.lower() != UNKNOWN else None


def _clean_platforms(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("platforms must be an array of strings")
    cleaned = (platform for platform in (_known_or_none(item) for item in value) if platform)
    return list(dict.fromkeys(cleaned))


def _to_announcement(item: object) -> AnnouncementFacts:
    if not isinstance(item, dict):
        raise ValueError("every element of the returned array must be a JSON object")
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("an announcement object is missing its title")
    return AnnouncementFacts.model_validate(
        {
            "title": title.strip(),
            "release_date": _known_or_none(item.get("release_date")),
            "platforms": _clean_platforms(item.get("platforms")),
            "event_type": _known_or_none(item.get("event_type")),
            "relevance": item.get("relevance"),
        }
    )


def _parse_batch(raw: str, expected: int) -> list[AnnouncementFacts]:
    payload = json.loads(_strip_json_fence(raw))
    if not isinstance(payload, list):
        raise ValueError("the model did not return a JSON array")
    if len(payload) != expected:
        raise ValueError(f"expected {expected} fact object(s), got {len(payload)}")
    return [_to_announcement(item) for item in payload]


__all__ = [
    "BATCH_SIZE",
    "IR_KEYWORDS",
    "IR_TOKEN_PATTERN",
    "MIN_MAX_TOKENS",
    "SYSTEM_PROMPT",
    "UNKNOWN",
    "AnnouncementChunk",
    "AnnouncementFacts",
    "FactBatch",
    "LLMClient",
    "build_batch_prompt",
    "expand_to_fact_updates",
    "extract_facts",
    "is_ir_release",
    "make_openai_client",
]
