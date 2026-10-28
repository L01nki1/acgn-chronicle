"""Export and verify the KnowledgePort contract schema (the engineering side of 05 §5).

Owner: B | Spec: 05 §1 / §5, invariant 5 of spec 02

`ports.py` is the source of truth; this module renders the types named in `ports.WIRE_MODELS` into a
deterministic JSON Schema. CI calls `check()`: a change to `ports.py` without re-exporting is a failed
build, and an internal type leaking into the schema is a bug in `WIRE_MODELS`.
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


def _wire_models() -> dict[str, type[BaseModel]]:
    """Resolve `ports.WIRE_MODELS` to the classes it names, failing loudly on a typo.

    Only wire types are exported. An internal type in the schema is worse than useless: a
    consumer will find it, code against it, and then discover at runtime that it never appears
    in a response (this is exactly what happened with `RetrievedChunk.text`).
    """
    out: dict[str, type[BaseModel]] = {}
    for name in ports.WIRE_MODELS:
        obj = getattr(ports, name, None)
        if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
            raise AttributeError(
                f"ports.WIRE_MODELS names {name!r}, which is not a BaseModel in ports.py"
            )
        out[name] = obj
    return out


def build_schema() -> dict:
    models = {name: obj.model_json_schema() for name, obj in _wire_models().items()}
    return {
        "schema_version": ports.SCHEMA_VERSION,
        "source": "src/acgn_rag/core/ports.py",
        "models": {name: models[name] for name in sorted(models)},
    }


def dump(schema: dict | None = None) -> str:
    return json.dumps(schema or build_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def check() -> tuple[bool, str]:
    """Return (in_sync, message). Never writes."""
    if not SCHEMA_PATH.exists():
        return (
            False,
            f"MISSING: {SCHEMA_PATH} does not exist, run `python scripts/export_schema.py`",
        )
    if SCHEMA_PATH.read_text(encoding="utf-8") != dump():
        return False, "DRIFT: ports.py and contracts/knowledge_port.schema.json disagree"
    return True, "OK: contract schema matches ports.py"


def write() -> Path:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(dump(), encoding="utf-8")
    return SCHEMA_PATH


__all__ = ["SCHEMA_PATH", "build_schema", "check", "dump", "write"]
