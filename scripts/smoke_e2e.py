"""End-to-end offline smoke: frozen snapshot -> index -> facts -> retrieval -> grounded /ask.

Owner: F (WP2) | Spec: docs/ROADMAP.md G1b, docs/RUNBOOK.md §1-§3, docs/EVAL_PROTOCOL.md

G1b's four checks need a real corpus, a real index and a real LLM run. The corpus is frozen
and distributed out of band (RUNBOOK §2), so this script is the offline proof that every
*code* seam between them works on the frozen snapshot, before anyone spends the measured
hours of CPU encoding or an LLM budget:

    1. build_index.main      -- config A over the snapshot's representatives (RUNBOOK §2)
    2. export_fact_chunks    -- the indexed chunks the fact extractor consumes (§3.6)
    3. extract_facts.main    -- L0 rows with chunkId + url provenance (§3.6), stub vendor
    4. retrieval             -- the bound collection answering a query (EVAL_PROTOCOL)
    5. /ask                  -- the web app answering from the bound store, or abstaining

Every step runs with injected fakes: a deterministic hash-based encoder, a stub answer client
and a stub extraction client, so this proves plumbing, not quality. The report says so in one
line, and the real numbers stay the G1b evidence (ROADMAP §2, RUNBOOK §3.3/§3.6). No network,
no model download, no LLM call; the snapshot is read, never written.

    python scripts/smoke_e2e.py --snapshot 20261105-d797b5f0ae8e --limit 200 \
        --out eval/smoke/<run_id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from acgn_rag.config import Settings, load_settings
from acgn_rag.core.embed.encoder import Embedder
from acgn_rag.core.ground.generate import Completion
from acgn_rag.core.ingest.snapshot import iter_snapshot_items
from acgn_rag.core.ports import RetrievalQuery
from acgn_rag.core.store.chroma_store import ChromaStore

#: The sibling CLI modules live beside this file, not on a package path: a script run as
#: ``python scripts/smoke_e2e.py`` has ``scripts/`` as sys.path[0] but no ``scripts`` package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_index
import export_fact_chunks
import extract_facts

#: Fake vector width. Any positive width works: the store only needs a consistent dimension.
DIM = 32
#: Config this smoke builds. A is the default lane and the one the QA set is written against.
CONFIG = "A"
LABEL = "A"


class HashTokenizer:
    """Whitespace tokenizer with e5-shaped specials, so the chunk budget is the real one.

    Copied in spirit from tests/test_build_index.py: a build must never see a tokenizer that
    silently disagrees with the encoder's truncation limit (RUNBOOK §3.4), and the fake model
    below is paired with exactly this class.
    """

    BOS = 0
    EOS = 2

    def __init__(self) -> None:
        self._id_of: dict[str, int] = {}
        self._word_of: dict[int, str] = {}

    def _word_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            for part in _split_colon(word):
                if part not in self._id_of:
                    self._id_of[part] = 10 + len(self._id_of)
                    self._word_of[self._id_of[part]] = part
                ids.append(self._id_of[part])
        return ids

    def count(self, text: str) -> int:
        return len(self._word_ids(text))

    def ids(self, text: str) -> list[int]:
        return [self.BOS, *self._word_ids(text), self.EOS]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._word_of[i] for i in ids if i in self._word_of)


def _split_colon(word: str) -> list[str]:
    parts: list[str] = []
    for index, piece in enumerate(word.split(":")):
        if index:
            parts.append(":")
        if piece:
            parts.append(piece)
    return parts


class _Encoded:
    """The one attribute ``HFTokenCounter`` reads from a tokenizer call."""

    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class _CallableTokenizer:
    """``HashTokenizer`` behind the transformers calling convention (``tokenizer(text)``)."""

    def __init__(self, inner: HashTokenizer) -> None:
        self._inner = inner

    def __call__(self, text: str, *, add_special_tokens: bool = True, **_kwargs: Any) -> _Encoded:
        ids = self._inner.ids(text)
        return _Encoded(ids if add_special_tokens else ids[1:-1])

    def decode(self, ids: Sequence[int], **_kwargs: Any) -> str:
        return self._inner.decode(ids)

    def ids(self, text: str) -> list[int]:
        """``TokenCounter.ids`` as the chunker declares it (specials included)."""
        return self._inner.ids(text)

    def count(self, text: str) -> int:
        """``TokenCounter.count``: content tokens only, specials excluded."""
        return self._inner.count(text)


class HashModel:
    """Deterministic vectors from sha256(text): identical text, identical vector.

    It also stands in for the tokenizer object a real ``sentence-transformers`` model exposes
    (callable, ``add_special_tokens`` aware), because ``build_index`` derives its token counter
    from the *same* model that encodes (RUNBOOK §3.4).
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.max_seq_length = 512
        self.tokenizer = _CallableTokenizer(HashTokenizer())

    def __call__(self, text: str, *, add_special_tokens: bool = True, **_kwargs: Any) -> Any:
        """Transformers-shaped call, which is what ``build_index.HFTokenCounter`` expects."""
        return self.tokenizer(text, add_special_tokens=add_special_tokens)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        prompt_name: str | None = None,
        **_kwargs: Any,
    ) -> np.ndarray:
        return np.array([self.vector(text) for text in texts], dtype=np.float32)

    def vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)


class StubAnswerClient:
    """One fixed, well-formed answer so the grounding path runs without a vendor.

    The shape is the one ``SYSTEM_PROMPT`` asks for: a single ``claims`` array whose entries
    carry ``evidence_indices``. The stub cites evidence number 1, which the prompt numbers from
    the first retrieved chunk; the request-local mapping in ``core/ground`` turns that index
    back into a chunk id (RUNBOOK §3). A stub that echoed ids would prove nothing about the
    mapping.

    The claim is a verbatim 60-character window of that evidence, because the first retrieved
    chunk may be a ``media`` item and COMPLIANCE C4 only allows a media source to be quoted,
    never paraphrased. A paraphrase here would be rejected as ``all_claims_rejected`` - which
    is the guard working, not a plumbing failure.
    """

    #: Characters taken from the start of the first numbered evidence block.
    QUOTE_CHARS = 60

    def __init__(self) -> None:
        self.calls = 0
        self.quotes: list[str] = []

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        self.calls += 1
        quote = _first_evidence_text(user)[: self.QUOTE_CHARS].strip()
        self.quotes.append(quote)
        payload = json.dumps(
            {
                "claims": [
                    {
                        "text": f"“{quote}”",
                        "evidence_indices": [1],
                    }
                ]
            },
            ensure_ascii=False,
        )
        return Completion(text=payload, finish_reason="stop", usage=None)


def _first_evidence_text(user: str) -> str:
    """Read the ``text`` field of evidence block ``1.`` out of the numbered prompt."""
    for line in user.splitlines():
        if not line.startswith("1. "):
            continue
        try:
            block = json.loads(line[3:])
        except json.JSONDecodeError:
            return ""
        return str(block.get("text") or "")
    return ""


class StubExtractionClient:
    """A well-formed extraction response per batch, so the facts path runs without a vendor.

    ``extract_facts`` sends numbered announcements and expects one flat JSON object per slot,
    in order (RUNBOOK §3.6). The stub answers the same number of slots with a fixed schema; it
    proves the batch -> parse -> ``FactUpdate`` -> disk path, never extraction quality.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        self.calls += 1
        slots = sum(1 for line in user.splitlines() if line[:1].isdigit())
        return json.dumps(
            [
                {
                    "title": f"Synthetic Work {index}",
                    "release_date": "2026-10-05",
                    "platforms": ["Switch"],
                    "event_type": "release",
                    "relevance": 0.9,
                }
                for index in range(1, max(slots, 1) + 1)
            ],
            ensure_ascii=False,
        )


def _docs_path(snapshot: str, data_root: Path, limit: int | None) -> tuple[Path, int]:
    """Materialize the snapshot's representatives as the build's docs.jsonl input.

    ``build_index.py`` takes a docs.jsonl until A's reader is wired into it (its module TODO),
    so the smoke writes the same rows the snapshot holds; nothing is re-normalized.
    """
    out = data_root / "smoke" / snapshot / "docs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for item in iter_snapshot_items(snapshot, data_root=data_root, representatives_only=True):
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return out, count


def _build(snapshot: str, docs: Path, data_root: Path) -> dict[str, Any]:
    model = HashModel()
    started = time.monotonic()
    code = build_index.main(
        [
            "--snapshot",
            snapshot,
            "--input",
            str(docs),
            "--config",
            CONFIG,
            "--data-root",
            str(data_root),
            "--fresh",
        ],
        tokenizer=model.tokenizer,
        model=model,
    )
    return {"exit_code": code, "seconds": round(time.monotonic() - started, 3)}


def _export(snapshot: str, data_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    code = export_fact_chunks.main(
        [
            "--snapshot",
            snapshot,
            "--config",
            CONFIG,
            "--data-root",
            str(data_root),
            "--output",
            f"smoke/{snapshot}/chunks.jsonl",
        ]
    )
    return {"exit_code": code, "seconds": round(time.monotonic() - started, 3)}


def _facts(snapshot: str, data_root: Path, limit: int | None) -> dict[str, Any]:
    """Run the extractor over the exported chunks with a stub vendor, then audit the rows.

    ``extract_facts.py`` has no ``--data-root``: it reads ``ACGN_DATA_DIR`` through
    ``load_settings()``. The smoke points that variable at the same root for the call and
    restores it afterwards, so one run never leaves the process on another root.
    """
    chunks = data_root / "chunks" / "smoke" / snapshot / "chunks.jsonl"
    argv = ["--snapshot", snapshot, "--input", str(chunks)]
    if limit is not None:
        # A limited write needs a single-use preview id (RUNBOOK §1).
        argv += ["--limit", str(limit), "--preview", "--preview-id", f"smoke-{snapshot}"]
    client = StubExtractionClient()
    previous = os.environ.get("ACGN_DATA_DIR")
    os.environ["ACGN_DATA_DIR"] = str(data_root)
    started = time.monotonic()
    try:
        code = extract_facts.main(argv, client=client)
    finally:
        if previous is None:
            os.environ.pop("ACGN_DATA_DIR", None)
        else:
            os.environ["ACGN_DATA_DIR"] = previous
    seconds = round(time.monotonic() - started, 3)
    if code != 0:
        return {"exit_code": code, "seconds": seconds}

    if limit is not None:
        facts_path = (
            data_root / "previews" / f"smoke-{snapshot}" / "facts" / snapshot / "facts.jsonl"
        )
    else:
        facts_path = data_root / "facts" / snapshot / "facts.jsonl"
    rows = [
        json.loads(line)
        for line in facts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "exit_code": code,
        "seconds": seconds,
        "path": str(facts_path),
        "rows": len(rows),
        "model_calls": client.calls,
        # G1b: every fact row carries its chunkId and url (ROADMAP §2).
        "rows_with_provenance": sum(
            1 for row in rows if str(row.get("chunk_id") or "") and str(row.get("url") or "")
        ),
    }


def _retrieval(snapshot: str, data_root: Path, settings: Settings, top_k: int) -> dict[str, Any]:
    from acgn_rag.core.eval import runner
    from acgn_rag.core.retrieve.vector import search

    store = ChromaStore(data_root / "index" / "chroma", settings.store)
    build_identity, collection, counts, chunker = runner.load_build_records(
        data_root / "index", snapshot_id=snapshot, config=CONFIG, label=LABEL
    )
    runner.bind_existing_collection(
        store, snapshot_id=snapshot, chunk_config=LABEL, collection=collection
    )
    embedder = Embedder(settings.embed, model=HashModel())
    query = RetrievalQuery(text="anime announcement", top_k=top_k, snapshot_id=snapshot)

    started = time.monotonic()
    chunks = search(query, store=store, embedder=embedder)
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    return {
        "collection": collection,
        "counts": dict(counts),
        "chunker": dict(chunker),
        "encoder_model": build_identity.get("model"),
        "top_k": top_k,
        "returned": len(chunks),
        "latency_ms": elapsed_ms,
        "top_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "top_scores": [round(chunk.score, 6) for chunk in chunks],
    }


def _ask(
    snapshot: str,
    data_root: Path,
    settings: Settings,
    top_k: int,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    """Answer through the web app's own route, so the wire shape is what is proven.

    The backend here is the smallest real implementation of the four port operations: the
    store, the retrieval function and B's answerer, with no mock data in between. ``/ask``
    therefore exercises the request-local citation mapping and the final-payload guard on
    chunks that really came out of the built collection.
    """
    from fastapi.testclient import TestClient

    from acgn_rag.core.ground.generate import GroundedAnswerer
    from acgn_rag.core.ground.serialize import to_excerpt
    from acgn_rag.core.ports import CorpusSnapshot, SnapshotInventory
    from acgn_rag.core.retrieve.vector import search
    from acgn_rag.web.api import create_app

    store = ChromaStore(data_root / "index" / "chroma", settings.store)
    store.bind_existing(snapshot, LABEL)
    embedder = Embedder(settings.embed, model=HashModel())
    client_stub = StubAnswerClient()
    answerer = GroundedAnswerer(client_stub)
    snapshot_meta = _snapshot_meta(data_root, snapshot)

    class Backend:
        def inventory(self) -> SnapshotInventory:
            return SnapshotInventory(current=snapshot, retained=[snapshot])

        def snapshot(self, snapshot_id: str | None = None) -> CorpusSnapshot:
            return snapshot_meta

        def retrieve(self, query: RetrievalQuery) -> list[Any]:
            return [to_excerpt(chunk) for chunk in search(query, store=store, embedder=embedder)]

        def ask(self, query: RetrievalQuery) -> Any:
            return answerer.answer(query, search(query, store=store, embedder=embedder))

    client = TestClient(create_app(Backend()))
    body = {"text": "anime announcement", "top_k": top_k, "snapshot_id": snapshot}
    response = client.post("/ask", json=body)
    payload = response.json()
    cited = list(payload.get("cited_chunk_ids") or [])
    retrieved = retrieval.get("top_chunk_ids") or []
    return {
        "status_code": response.status_code,
        "abstained": payload.get("abstained"),
        "refusal_reason": payload.get("abstain_reason"),
        "citation_count": len(payload.get("citations") or []),
        "cited_chunk_ids": cited,
        # G2 hard check 4 / KnowledgePort §3.7: every cited chunk was actually retrieved here.
        "cited_subset_of_retrieved": bool(cited) and set(cited) <= set(retrieved),
        "answer_chars": len(payload.get("answer") or ""),
        "model_calls": client_stub.calls,
        "claim_quote": client_stub.quotes[0] if client_stub.quotes else None,
    }


def _snapshot_meta(data_root: Path, snapshot: str) -> Any:
    """The frozen manifest's counts, read back as the contract's CorpusSnapshot."""
    from acgn_rag.core.ports import CorpusSnapshot

    manifest = json.loads(
        (data_root / "snapshots" / snapshot / "manifest.json").read_text(encoding="utf-8")
    )
    return CorpusSnapshot(
        snapshot_id=snapshot,
        created_at=str(manifest.get("created_at") or manifest.get("published_at") or ""),
        item_count=int(manifest.get("item_count") or 0),
        deduped_count=int(manifest.get("deduped_count") or 0),
        source_count=int(manifest.get("source_count") or 0),
        window_start=str(manifest["window_start"]),
        window_end=str(manifest["window_end"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline end-to-end smoke on a frozen snapshot")
    parser.add_argument("--snapshot", required=True, help="frozen snapshot id under data/snapshots")
    parser.add_argument(
        "--data-root", default=None, help="data root; default ACGN_DATA_DIR or ./data"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="build from at most N representatives"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=None, help="report path (JSON); default stdout only")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        print("--limit must be a positive integer", file=sys.stderr)
        return 2

    settings = load_settings()
    data_root = Path(args.data_root) if args.data_root else settings.paths.root
    if not (data_root / "snapshots" / args.snapshot).is_dir():
        print(
            f"snapshot {args.snapshot!r} is not under {data_root / 'snapshots'} - extract the "
            "private archive first (RUNBOOK §2) and verify it against eval/snapshot_evidence.json",
            file=sys.stderr,
        )
        return 2

    report: dict[str, Any] = {
        "snapshot_id": args.snapshot,
        "config": CONFIG,
        "limit": args.limit,
        "fakes": "hash encoder + stub answer client: plumbing evidence only, not quality evidence",
        "steps": {},
    }
    started = time.monotonic()
    try:
        docs, doc_count = _docs_path(args.snapshot, data_root, args.limit)
        report["steps"]["docs"] = {"path": str(docs), "documents": doc_count}
        report["steps"]["build"] = _build(args.snapshot, docs, data_root)
        if report["steps"]["build"]["exit_code"] != 0:
            raise RuntimeError("build_index failed; see stderr above")
        report["steps"]["export"] = _export(args.snapshot, data_root)
        if report["steps"]["export"]["exit_code"] != 0:
            raise RuntimeError("export_fact_chunks failed; see stderr above")
        report["steps"]["facts"] = _facts(args.snapshot, data_root, args.limit)
        if report["steps"]["facts"]["exit_code"] != 0:
            raise RuntimeError("extract_facts failed; see stderr above")
        report["steps"]["retrieval"] = _retrieval(args.snapshot, data_root, settings, args.top_k)
        report["steps"]["ask"] = _ask(
            args.snapshot, data_root, settings, args.top_k, report["steps"]["retrieval"]
        )
    except Exception as exc:  # report the failing step rather than a bare traceback
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["total_seconds"] = round(time.monotonic() - started, 3)

    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
