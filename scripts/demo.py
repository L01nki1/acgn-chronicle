# ruff: noqa: RUF001 - the demo's default question is Chinese; the fullwidth question mark is correct.
"""One-command reproduce for the WP6 page: `uv run python scripts/demo.py`.

Owner: C | Spec: docs/ROADMAP.md G2, issue #5 (D's W3 assignment, item 1)

Two modes, both offline and both talking only to the contract routes:

    python scripts/demo.py              # transcript: /snapshots -> /retrieve -> /ask, then exit
    python scripts/demo.py --serve      # the same host over HTTP at http://127.0.0.1:8000

The transcript runs the real ASGI app (`create_app` with the mock backend) through an HTTP test
client, so the printed payloads are the wire shapes a browser receives — not a summary the script
invented. The backend is the fixed mock: this proves the page, the routes and the compliance
guards, never retrieval or answer quality. The real pipeline lives in `scripts/smoke_e2e.py` and
`docs/RUNBOOK.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from fastapi.testclient import TestClient

from acgn_rag.web.api import create_app

#: A question the mock corpus can answer with citations, and the G2 demo's default.
DEFAULT_QUESTION = "第二季什么时候开播？"
#: Mirrors `contract.mjs`: a rendered quote never exceeds this many Unicode code points.
MAX_QUOTE_CODEPOINTS = 200


def _post(client: TestClient, path: str, body: dict[str, Any]) -> tuple[int, Any]:
    response = client.post(path, json=body)
    return response.status_code, response.json()


def transcript(question: str = DEFAULT_QUESTION, top_k: int = 3) -> dict[str, Any]:
    """Drive the three contract reads and return what the page would render.

    The guards are checked here rather than asserted: a demo that printed "OK" without reading
    the payload would hide exactly the failures the page is meant to catch.
    """
    client = TestClient(create_app())

    inventory = client.get("/snapshots").json()
    snapshot_id = inventory["current"]
    query = {"text": question, "top_k": top_k, "snapshot_id": snapshot_id}

    retrieve_status, excerpts = _post(client, "/retrieve", query)
    ask_status, answer = _post(client, "/ask", query)

    rendered = []
    dropped_without_url = 0
    for excerpt in excerpts:
        citation = excerpt.get("citation") or {}
        if not citation.get("url"):
            dropped_without_url += 1
            continue
        rendered.append(
            {
                "chunk_id": excerpt["chunk_id"],
                "source_type": citation["source_type"],
                "publisher": citation["publisher"],
                "published_at": citation.get("published_at"),
                "url": citation["url"],
                "quote_codepoints": len(excerpt["quote"]),
            }
        )

    return {
        "mode": "mock",
        "snapshot_id": snapshot_id,
        "question": question,
        "retrieve": {
            "status": retrieve_status,
            "excerpts": len(excerpts),
            "rendered": rendered,
            "dropped_without_url": dropped_without_url,
            "quote_budget_ok": all(
                item["quote_codepoints"] <= MAX_QUOTE_CODEPOINTS for item in rendered
            ),
        },
        "ask": {
            "status": ask_status,
            "abstained": answer.get("abstained"),
            "abstain_reason": answer.get("abstain_reason"),
            "answer": answer.get("answer"),
            "citation_count": len(answer.get("citations") or []),
            "cited_chunk_ids": answer.get("cited_chunk_ids") or [],
        },
    }


def _print_transcript(report: dict[str, Any]) -> None:
    retrieve = report["retrieve"]
    ask = report["ask"]
    print(f"snapshot   {report['snapshot_id']} (mock)")
    print(f"question   {report['question']}")
    print(
        f"retrieve   HTTP {retrieve['status']}, {retrieve['excerpts']} excerpts, "
        f"{len(retrieve['rendered'])} renderable, {retrieve['dropped_without_url']} without a URL"
    )
    for item in retrieve["rendered"]:
        print(
            f"  [{item['source_type']}] {item['publisher']} · {item['published_at']}"
            f" · {item['quote_codepoints']} codepoints · {item['url']}"
        )
    state = "refused" if ask["abstained"] else "answered"
    print(f"ask        HTTP {ask['status']}, {state}, {ask['citation_count']} citations")
    if ask["abstained"]:
        print(f"  reason   {ask['abstain_reason']}")
    else:
        print(f"  answer   {ask['answer']}")
        print(f"  cited    {', '.join(ask['cited_chunk_ids'])}")
    print()
    print("mock backend: page, routes and compliance guards only — not retrieval or answer quality")
    print("open the page:  uv run python -m acgn_rag.web  ->  http://127.0.0.1:8000")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline G2 demo for the WP6 query page")
    parser.add_argument("--serve", action="store_true", help="serve the page over HTTP instead")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="print the transcript as JSON")
    args = parser.parse_args(argv)

    if args.serve:
        import uvicorn

        print(f"serving {args.host}:{args.port} (Ctrl-C to stop)")
        uvicorn.run("acgn_rag.web.api:app", host=args.host, port=args.port)
        return 0

    if args.top_k < 1:
        print("--top-k must be a positive integer", file=sys.stderr)
        return 2
    report = transcript(args.question, args.top_k)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_transcript(report)
    ok = (
        report["retrieve"]["status"] == 200
        and report["ask"]["status"] == 200
        and report["retrieve"]["quote_budget_ok"]
        and bool(report["retrieve"]["rendered"]) == (report["retrieve"]["excerpts"] > 0)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
