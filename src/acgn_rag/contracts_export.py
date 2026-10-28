"""KnowledgePort 契约的导出与一致性校验（05 §5 冻结纪律的工程实现）。

负责人：B　|　依据：05 §1 / §5、02 §3 不变量 5

`ports.py` 是唯一真相；本模块把它的 Pydantic 类型渲染成确定性 JSON Schema。
CI 会跑 `check()`，不一致即失败 —— 改 ports 不重新导出 schema 等于没改完。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from acgn_rag.core import ports


def _repo_root() -> Path:
    env = os.environ.get("ACGN_REPO_ROOT")
    if env:
        return Path(env)
    # src/acgn_rag/contracts_export.py -> src/acgn_rag -> src -> <repo>
    return Path(__file__).resolve().parents[2]


SCHEMA_PATH = _repo_root() / "contracts" / "knowledge_port.schema.json"


def build_schema() -> dict:
    models = {
        name: obj.model_json_schema()
        for name, obj in vars(ports).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    }
    return {
        "schema_version": ports.SCHEMA_VERSION,
        "source": "src/acgn_rag/core/ports.py",
        "models": {name: models[name] for name in sorted(models)},
    }


def dump(schema: dict | None = None) -> str:
    return json.dumps(schema or build_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def check() -> tuple[bool, str]:
    """返回 (是否一致, 说明)。不写文件。"""
    if not SCHEMA_PATH.exists():
        return False, f"MISSING: {SCHEMA_PATH} 尚未生成，先跑 `python scripts/export_schema.py`"
    if SCHEMA_PATH.read_text(encoding="utf-8") != dump():
        return False, "DRIFT: ports.py 与 contracts/knowledge_port.schema.json 不一致"
    return True, "OK: 契约 schema 与 ports.py 一致"


def write() -> Path:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(dump(), encoding="utf-8")
    return SCHEMA_PATH


__all__ = ["SCHEMA_PATH", "build_schema", "check", "dump", "write"]
