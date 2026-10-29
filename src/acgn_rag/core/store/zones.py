"""Two-zone disk layout (invariant C2): data/facts/ publishable, data/raw/ internal-only.

Owner: F | Spec: 06 §3 C2, docs/RUNBOOK.md §3.2

C2 invariant: the raw zone (``data/raw/``) is internal retrieval input and is NEVER
serialized to a client. Only ``core/ground/serialize.py`` may build outbound payloads
(excerpts plus citations, see the compliance tests); the facts zone holds FactUpdate-shaped
rows (media / field / value / chunk_id / url / publisher), never article bodies. Rows are
validated against ``ports.FactUpdate`` at write time, so a body row cannot enter the
publishable zone by accident. ``assert_publishable`` enforces the zone boundary for any
outbound write: it raises when any component of the path is the raw-zone directory ``raw``
(conservative on purpose - a file *under* anything named ``raw`` is refused).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from acgn_rag.core.ports import FactUpdate

RAW_ZONE_DIR = "raw"


def assert_publishable(path: Path) -> Path:
    """Return the resolved path, or raise if it lies in the raw zone (never serialized).

    The check is on path *components*: ``data/raw/x.jsonl`` and ``data/raw/2026/x.jsonl``
    raise, while ``data/facts/raw-notes.jsonl`` passes (the file name is not a zone marker).
    A directory literally named ``raw`` anywhere under the data root marks the internal zone.
    """
    resolved = Path(path).resolve()
    if RAW_ZONE_DIR in resolved.parts:
        raise ValueError(
            f"{resolved} is inside the raw zone (data/{RAW_ZONE_DIR}/): the raw zone is "
            "internal-only and never serialized (spec 06 §3 C2). Publishable files live "
            "under data/facts/; outbound payloads are built by core/ground/serialize.py."
        )
    return resolved


def write_facts_zone(rows: list[dict], path: Path) -> Path:
    """Write FactUpdate-shaped rows as publishable JSONL, atomically (tmp file + os.replace).

    Rows are validated against ``ports.FactUpdate`` BEFORE any disk write: an invalid row
    leaves an existing file untouched rather than half-replaced. Keys are sorted per line so
    two builds of the same rows produce byte-identical files (manifest ordering, RUNBOOK §2).
    """
    target = assert_publishable(path)
    validated = [FactUpdate.model_validate(row).model_dump() for row in rows]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for row in validated:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, target)
    return target


__all__ = ["RAW_ZONE_DIR", "assert_publishable", "write_facts_zone"]
