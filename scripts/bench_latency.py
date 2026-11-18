#!/usr/bin/env python3
"""Measure the local-CPU ③ latency gate shape (issue #101 / #94).

Gate shape: the full E5 index for Config A, ``top_k=5``, the widest QA per-item
window as a ``published_ts`` range filter, one stored embedding as the query
vector, warm-up calls, then at least 30 timed ``ChromaStore.query`` calls.
Encoding time is excluded by construction.

The command writes a JSON artifact with the gate shape, environment, code and
full index identity (including the HNSW segment files), and the latency summary.
Point ``--chroma-root`` at a disposable copy of the index: Chroma opens the
store read-write.

Example (from the repository root):

    PYTHONPATH=src python scripts/bench_latency.py \
        --chroma-root /path/to/disposable/chroma \
        --chunk-config A \
        --qa-set eval/qa_set.jsonl \
        --out bench/latency-20261118-cpu.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

DEFAULT_SNAPSHOT = "20261105-d797b5f0ae8e"
DEFAULT_TOP_K = 5
DEFAULT_REPEATS = 30
DEFAULT_WARMUPS = 5
DEFAULT_SINCE = "2023-10-28"
DEFAULT_UNTIL = "2026-10-28"
ARTIFACT_VERSION = 1
ARTIFACT_SECTIONS = (
    "artifact_version",
    "gate_eligible",
    "gate_ineligible_reasons",
    "gate_shape",
    "environment",
    "code",
    "index",
    "latency",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the ceil(q/100 * n)-th smallest value."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sample")
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return float(ordered[rank - 1])


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "mean": round(statistics.fmean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def cpu_model() -> tuple[str, str]:
    """Return (model, source); never invent a model from platform.processor()."""
    if sys.platform == "darwin":
        try:
            model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown", "unavailable"
        return (model, "sysctl") if model else ("unknown", "unavailable")
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip(), "/proc/cpuinfo"
    except OSError:
        pass
    return "unknown", "unavailable"


def git_state(repo_root: Path) -> tuple[str, bool]:
    """Return (commit SHA, dirty); unknown SHA is treated as dirty."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return sha, bool(status.strip())


def hash_tree(root: Path) -> list[dict[str, object]]:
    """Hash every file under the Chroma root, including the HNSW segments."""
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def read_sidecars(root: Path, snapshot: str, chunk_config: str) -> dict[str, object]:
    label = "A-prime" if chunk_config == "A'" else chunk_config
    sidecar_dir = root.parent / snapshot / label
    result: dict[str, object] = {"directory": str(sidecar_dir)}
    for name in ("build.json", "identity.json"):
        path = sidecar_dir / name
        try:
            result[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result[name] = None
    return result


def build_window_where(since: str, until: str) -> dict:
    from acgn_rag.core.ports import RetrievalFilters
    from acgn_rag.core.retrieve.vector import build_where

    where = build_where(RetrievalFilters(since=since, until=until))
    if where is None:
        raise ValueError("the latency gate requires a date window")
    return where


def window_from_qa_set(path: Path) -> tuple[str, str, int]:
    """Return the widest per-item window in a QA set."""
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not items:
        raise ValueError(f"QA set {path} is empty")
    since = min(str(item["window"][0]) for item in items)
    until = max(str(item["window"][1]) for item in items)
    return since, until, len(items)


def measure(query, repeats: int, warmups: int) -> dict[str, object]:
    for _ in range(warmups):
        query()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        query()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"summary": latency_summary(samples), "raw_ms": samples}


def build_artifact(
    *,
    gate_eligible: bool,
    gate_ineligible_reasons: list[str],
    gate_shape: dict[str, object],
    environment: dict[str, object],
    code: dict[str, object],
    index: dict[str, object],
    latency: dict[str, object],
) -> dict[str, object]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "gate_eligible": gate_eligible,
        "gate_ineligible_reasons": gate_ineligible_reasons,
        "gate_shape": gate_shape,
        "environment": environment,
        "code": code,
        "index": index,
        "latency": latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-root", required=True, type=Path)
    parser.add_argument("--chunk-config", default="A", choices=("A", "A-prime"))
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--qa-set", type=Path, default=None)
    parser.add_argument("--since", default=DEFAULT_SINCE)
    parser.add_argument("--until", default=DEFAULT_UNTIL)
    parser.add_argument("--over-fetch", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.top_k < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("top_k and repeats must be >= 1; warmups must be >= 0")
    if args.over_fetch is not None and args.over_fetch < 1:
        parser.error("--over-fetch must be >= 1")

    import chromadb

    from acgn_rag.config import load_settings
    from acgn_rag.core.store.chroma_store import ChromaStore

    repo_root = Path(__file__).resolve().parents[1]
    store_config = load_settings().store
    if args.over_fetch is not None:
        store_config = replace(store_config, over_fetch_factor=args.over_fetch)
    store = ChromaStore(args.chroma_root, store_config)
    collection = store.bind_existing(args.snapshot, args.chunk_config)
    count = int(collection.count())
    if count == 0:
        raise RuntimeError(f"collection {collection.name!r} is empty")

    if args.qa_set is not None:
        since, until, qa_items = window_from_qa_set(args.qa_set)
        window_source = f"qa_set (widest per-item window): {args.qa_set}"
    else:
        since, until, qa_items = args.since, args.until, None
        window_source = "explicit --since/--until"
    where = build_window_where(since, until)

    first = collection.get(limit=1, include=["embeddings", "metadatas"])
    stored = first["embeddings"][0]
    vector = [float(value) for value in stored]
    query_chunk_id = (first["metadatas"][0] or {}).get("chunk_id")
    result = measure(lambda: store.query(vector, args.top_k, where), args.repeats, args.warmups)

    sha, dirty = git_state(repo_root)
    model, model_source = cpu_model()
    reasons: list[str] = []
    if args.chunk_config != "A":
        reasons.append("rule gates Config A only; A-prime is reported")
    if args.qa_set is None:
        reasons.append("rule requires the QA per-item window; pass --qa-set")
    if dirty:
        reasons.append("git worktree is dirty")
    if sha == "unknown":
        reasons.append("git SHA unavailable")
    if model == "unknown":
        reasons.append("CPU model unavailable")

    artifact = build_artifact(
        gate_eligible=not reasons,
        gate_ineligible_reasons=reasons,
        gate_shape={
            "lane": "local CPU",
            "index": f"full E5 Config {args.chunk_config}",
            "chunk_config": args.chunk_config,
            "snapshot": args.snapshot,
            "collection": collection.name,
            "collection_count": count,
            "top_k": args.top_k,
            "over_fetch_factor": store.settings.over_fetch_factor,
            "over_fetch_source": (
                "--over-fetch" if args.over_fetch is not None else "load_settings().store"
            ),
            "since": since,
            "until": until,
            "window_source": window_source,
            "qa_items": qa_items,
            "where": where,
            "query_vector": "one stored embedding (encode time excluded)",
            "query_vector_chunk_id": query_chunk_id,
            "warmup_calls": args.warmups,
            "measured_calls": args.repeats,
            "percentile_method": "nearest-rank: ceil(p/100 * n)-th smallest",
        },
        environment={
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_model": model,
            "cpu_model_source": model_source,
            "chromadb": chromadb.__version__,
            "sqlite": sqlite3.sqlite_version,
        },
        code={
            "git_sha": sha,
            "git_dirty": dirty,
            "chroma_store_sha256": sha256_file(
                repo_root / "src" / "acgn_rag" / "core" / "store" / "chroma_store.py"
            ),
        },
        index={
            "chroma_root": str(args.chroma_root),
            "files": hash_tree(args.chroma_root),
            "sidecars": read_sidecars(args.chroma_root, args.snapshot, args.chunk_config),
        },
        latency=result,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"gate_eligible={artifact['gate_eligible']} reasons={reasons}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
