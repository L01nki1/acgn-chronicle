"""Export one explicitly selected index collection for the fact-extraction CLI.

The output contains every indexed chunk, not a team-approved announcement selection. Existing
build records do not prove that raw input was a complete frozen snapshot. Serialize writers
while exporting; the checks here are not a collection lock or an atomic publication protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from stat import S_ISREG
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.config import Paths, StoreConfig
from acgn_rag.core.eval.runner import bind_existing_collection, load_build_records
from acgn_rag.core.store import ChromaStore, collection_name

_LABELS = {"A": "A", "A'": "A-prime", "A0": "A0"}


def _output_parts(raw: str) -> tuple[str, ...]:
    """Accept a relative JSONL path beneath the selected data root's chunks directory."""
    if not raw or "\\" in raw or "\0" in raw or ntpath.splitdrive(raw)[0]:
        raise ValueError("--output must be a relative JSONL path under data-root/chunks")
    path = PurePosixPath(raw)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or any(part in {".", ".."} or ntpath.isreserved(part) for part in parts)
        or path.suffix != ".jsonl"
    ):
        raise ValueError("--output must be a relative JSONL path under data-root/chunks")
    return parts


def _directory(path: Path) -> None:
    """Create one directory without accepting an existing link or non-directory."""
    path.mkdir(exist_ok=True)
    if path.is_symlink() or path.is_junction() or not path.is_dir():
        raise ValueError(f"output directory must not be a link: {path}")


def _existing_output_is_safe(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"output must be an unlinked regular file: {path}")


def _output_paths(data_root: Path, parts: tuple[str, ...]) -> tuple[Path, Path]:
    root = data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / "chunks"
    _directory(directory)
    for part in parts[:-1]:
        directory = directory / part
        _directory(directory)
    output = directory / parts[-1]
    receipt = output.with_name(output.name + ".receipt.json")
    _existing_output_is_safe(output)
    _existing_output_is_safe(receipt)
    return output, receipt


def _build_fingerprints(index_dir: Path, snapshot_id: str, label: str) -> tuple[str, str]:
    base = index_dir / snapshot_id / label
    for directory in (index_dir, index_dir / snapshot_id, base, index_dir / "chroma"):
        if directory.is_symlink() or directory.is_junction():
            raise ValueError(f"index path must not be a link: {directory}")
    marker = base / "build.incomplete"
    try:
        marker.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError(f"index build is incomplete: {marker}")
    hashes = []
    for name in ("identity.json", "build.json"):
        path = base / name
        info = path.lstat()
        if not S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"build record must be an unlinked regular file: {path}")
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashes[0], hashes[1]


def _chunk_row(
    chunk_id: Any, document: Any, metadata: Any, snapshot_id: str, config: str
) -> dict[str, str]:
    if not isinstance(chunk_id, str) or not chunk_id or not isinstance(metadata, dict):
        raise ValueError("chunk row is missing its ID or metadata")
    if not isinstance(document, str) or not document.strip():
        raise ValueError(f"chunk {chunk_id!r} has no document text")
    doc_id = metadata.get("doc_id")
    parts = chunk_id.rsplit("::", 2)
    if (
        len(parts) != 3
        or not isinstance(doc_id, str)
        or not doc_id
        or parts[0] != doc_id
        or parts[1] != config
        or not parts[2].isascii()
        or not parts[2].isdigit()
        or metadata.get("chunk_id") != chunk_id
        or metadata.get("snapshot_id") != snapshot_id
    ):
        raise ValueError(f"chunk {chunk_id!r} has inconsistent snapshot or config identity")
    url = metadata.get("url")
    publisher = metadata.get("publisher")
    if not isinstance(url, str) or not url or not isinstance(publisher, str):
        raise ValueError(f"chunk {chunk_id!r} has incomplete source metadata")
    return {"chunk_id": chunk_id, "text": document, "url": url, "publisher": publisher}


def _temporary_file(directory: Path, name: str) -> Any:
    return tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{name}.", suffix=".tmp", dir=directory, delete=False
    )


def _export(
    *, data_root: Path, snapshot_id: str, config: str, parts: tuple[str, ...], page_size: int
) -> tuple[Path, Path, int]:
    label = _LABELS[config]
    index_dir = data_root / "index"
    fingerprints = _build_fingerprints(index_dir, snapshot_id, label)
    identity, recorded_collection, counts, chunker = load_build_records(
        index_dir, snapshot_id=snapshot_id, config=config, label=label
    )
    if counts["chunks"] != counts["stored"]:
        raise ValueError("build record has stale or mismatched chunk and stored counts")
    store = ChromaStore(index_dir / "chroma", StoreConfig())
    collection = bind_existing_collection(
        store, snapshot_id=snapshot_id, chunk_config=label, collection=recorded_collection
    )
    total = collection.count()
    if total != counts["stored"] or total <= 0:
        raise ValueError("collection count does not match its completed build record")
    output, receipt = _output_paths(data_root, parts)
    tmp_output: Path | None = None
    tmp_receipt: Path | None = None
    try:
        digest = hashlib.sha256()
        seen: set[str] = set()
        offset = 0
        with _temporary_file(output.parent, output.name) as handle:
            tmp_output = Path(handle.name)
            while offset < total:
                page = collection.get(
                    limit=page_size, offset=offset, include=["documents", "metadatas"]
                )
                ids = page.get("ids") or []
                documents = page.get("documents") or []
                metadatas = page.get("metadatas") or []
                if not ids or len(ids) != len(documents) or len(ids) != len(metadatas):
                    raise ValueError("collection page is empty or has incomplete fields")
                for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True):
                    row = _chunk_row(chunk_id, document, metadata, snapshot_id, config)
                    if row["chunk_id"] in seen:
                        raise ValueError(f"duplicate chunk ID: {row['chunk_id']!r}")
                    seen.add(row["chunk_id"])
                    line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                    handle.write(line)
                    digest.update(line)
                offset += len(ids)
                if offset > total:
                    raise ValueError("collection returned more rows than its recorded count")
        if len(seen) != total or collection.count() != total:
            raise ValueError("collection changed or returned an incomplete set of rows")
        if _build_fingerprints(index_dir, snapshot_id, label) != fingerprints:
            raise ValueError("build records changed while exporting")
        record = {
            "snapshot_id": snapshot_id,
            "config": config,
            "label": label,
            "collection": recorded_collection,
            "rows": total,
            "jsonl_sha256": digest.hexdigest(),
            "identity_record_sha256": fingerprints[0],
            "build_record_sha256": fingerprints[1],
            "identity_slug": identity["identity_slug"],
            "chunker": chunker,
            "build_counts": counts,
        }
        with _temporary_file(receipt.parent, receipt.name) as handle:
            tmp_receipt = Path(handle.name)
            handle.write((json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        _existing_output_is_safe(output)
        _existing_output_is_safe(receipt)
        os.replace(tmp_receipt, receipt)
        tmp_receipt = None
        os.replace(tmp_output, output)
        tmp_output = None
        return output, receipt, total
    finally:
        if tmp_output is not None:
            tmp_output.unlink(missing_ok=True)
        if tmp_receipt is not None:
            tmp_receipt.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export one index configuration for fact extraction"
    )
    parser.add_argument("--snapshot", required=True, help="snapshot ID to export")
    parser.add_argument(
        "--config", required=True, choices=tuple(_LABELS), help="index configuration"
    )
    parser.add_argument(
        "--output", required=True, help="relative JSONL path under data-root/chunks"
    )
    parser.add_argument(
        "--data-root", default=None, help="data root; default ACGN_DATA_DIR or ./data"
    )
    parser.add_argument(
        "--page-size", type=int, default=500, help="rows per collection read (positive)"
    )
    args = parser.parse_args(argv)
    try:
        if args.page_size <= 0:
            raise ValueError("--page-size must be positive")
        parts = _output_parts(args.output)
        collection_name(args.snapshot, _LABELS[args.config])
        data_root = Path(args.data_root) if args.data_root is not None else Paths().root
        output, receipt, total = _export(
            data_root=data_root,
            snapshot_id=args.snapshot,
            config=args.config,
            parts=parts,
            page_size=args.page_size,
        )
    except (OSError, ValueError, RuntimeError, LookupError) as exc:
        print(f"EXPORT REJECTED: {exc}", file=sys.stderr)
        return 2
    print(f"exported {total} indexed chunk(s) to {output}")
    print(f"wrote receipt {receipt}; verify its JSONL hash before using it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
