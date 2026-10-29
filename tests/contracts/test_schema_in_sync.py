"""Contract consistency (invariant 5 of spec 02 / 05 §5): editing ports without re-exporting fails here.

This test is the CI contract gate itself - do not turn it into a skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from acgn_rag import contracts_export
from acgn_rag.contracts_export import SCHEMA_PATH, build_schema, check
from acgn_rag.core import ports


def test_schema_matches_ports() -> None:
    ok, message = check()
    assert ok, f"{message} (fix: python scripts/export_schema.py)"


def test_schema_file_is_valid_json() -> None:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == build_schema()["schema_version"]
    assert "CorpusSnapshot" in payload["models"]


def test_export_writes_identical_lf_bytes_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "knowledge_port.schema.json"
    monkeypatch.setattr(contracts_export, "SCHEMA_PATH", output)
    assert contracts_export.write() == output
    assert output.read_bytes() == contracts_export.dump().encode("utf-8")


def test_wire_models_are_exactly_what_is_exported() -> None:
    exported = set(build_schema()["models"])
    assert exported == set(ports.WIRE_MODELS)


def test_internal_types_never_reach_the_schema() -> None:
    """`RetrievedChunk` carries the full body; a consumer must not be able to find it.

    Keeping it out of the schema is what makes "the contract has no full-text field" a fact
    rather than a promise.
    """
    models = build_schema()["models"]
    assert "RetrievedChunk" not in models
    assert "ChunkExcerpt" in models


def test_excerpt_model_exposes_quote_not_text() -> None:
    props = build_schema()["models"]["ChunkExcerpt"]["properties"]
    assert "quote" in props
    assert props["quote"]["maxLength"] == ports.MAX_QUOTE_CHARS
    assert "text" not in props
    assert set(build_schema()["models"]["ChunkExcerpt"]["required"]) >= {"chunk_id", "quote"}


def test_daily_digest_has_no_untyped_field() -> None:
    """`Sequence[dict]` is not a contract: it can neither be validated nor typed by a consumer."""
    props = build_schema()["models"]["DailyDigest"]["properties"]
    assert "episode_table" not in props
    for name, spec in props.items():
        if spec.get("type") == "array":
            assert "$ref" in json.dumps(spec["items"]), (
                f"DailyDigest.{name} is an array without an element type"
            )
