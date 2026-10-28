"""契约一致性（02 §3 不变量 5 / 05 §5）：改 ports 不重新导出 schema 必须在这里红。

这条用例是 CI 的契约闸门本体 —— 别把它删成 skip。
"""

from __future__ import annotations

import json

from acgn_rag.contracts_export import SCHEMA_PATH, build_schema, check


def test_schema_matches_ports() -> None:
    ok, message = check()
    assert ok, f"{message}（修复：python scripts/export_schema.py）"


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
    assert not missing, f"schema 缺少模型：{sorted(missing)}"
