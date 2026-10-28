"""Contract consistency (invariant 5 of spec 02 / 05 §5): editing ports without re-exporting fails here.

This test is the CI contract gate itself - do not turn it into a skip.
"""

from __future__ import annotations

import json

from acgn_rag.contracts_export import SCHEMA_PATH, build_schema, check


def test_schema_matches_ports() -> None:
    ok, message = check()
    assert ok, f"{message} (fix: python scripts/export_schema.py)"


def test_schema_file_is_valid_json() -> None:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == build_schema()["schema_version"]
    assert "CorpusSnapshot" in payload["models"]


def test_every_port_model_is_exported() -> None:
    exported = set(build_schema()["models"])
    expected = {
        "CorpusSnapshot",
        "RetrievalFilters",
        "RetrievalQuery",
        "Citation",
        "RetrievedChunk",
        "GroundedAnswer",
        "DailyReportPort",
        "CategoryDigest",
        "ClusterCard",
        "FactUpdate",
        "DailyDigest",
    }
    missing = expected - exported
    assert not missing, f"models missing from the schema: {sorted(missing)}"
