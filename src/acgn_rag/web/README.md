# WP6 · C — G1a query page

The browser-facing consumer of the KnowledgePort. G1a ships the page skeleton against a fixed mock;
W2 swaps the backend without changing any route or static file.

Owner: C · Gate: [issue #5](https://github.com/L01nki1/acgn-chronicle/issues/5) · Spec:
[`contracts/KnowledgePort.md`](../../../contracts/KnowledgePort.md), [`docs/COMPLIANCE.md`](../../../docs/COMPLIANCE.md)

## Run

```bash
uv sync --extra dev
uv run python -m acgn_rag.web          # http://127.0.0.1:8000
uv run python -m acgn_rag.web --reload # reload on source changes
```

The page loads `/snapshots` at startup and runs one default query, so the first screen is not an
empty form. `--host` and `--port` are also accepted.

## What G1a delivers

| Item | Where |
|---|---|
| query page skeleton (typing a question renders result cards) | `static/index.html`, `static/app.mjs` |
| data from a mock | `mock_port.py` — five fixed records, all URLs on `example.com` |
| snapshot picker + visible `snapshot_id` per response | `static/app.mjs` |
| citation jump left blank, marked "跳转将在 G2 启用" | `static/app.mjs` |
| empty vs refused as different panels and states | `contract.mjs` `viewState`, `app.mjs` |
| client-side second clipping (200 code points) and "no URL, no render" | `contract.mjs` |
| cache key `(snapshot_id, text, filters, top_k, config)` | `contract.mjs` `cacheKey` |

G2 adds clickable citation jumps, the evaluation view and the demo script. G3 adds the optional
`/daily` consumer; `/daily` currently answers `501`.

## Routes

Only the contract reads are exposed:

| Route | Behaviour |
|---|---|
| `GET /snapshots` | `current` + `retained`, most recent first |
| `GET /snapshot?id=…` | default = current; unknown = `404`; retired = `410` |
| `POST /retrieve` | `ChunkExcerpt[]`; no match = `[]` |
| `POST /ask` | `GroundedAnswer`; refusal keeps `abstained=true`, empty citations and a reason |
| `GET /daily` | `501` until G3 |

Errors follow `KnowledgePort` §5: `400` malformed body, `404` unknown snapshot, `410` retired,
`422` invalid filter, `503` not serving + `Retry-After`. API reads answer `Cache-Control: no-store`.
The page never falls back from `410` to the current snapshot; it offers an explicit re-run instead.

## Tests

```bash
uv run pytest -q tests/web
node --test tests/web/contract.test.mjs
```

The Node suite runs from pytest too; it is skipped only when Node is absent. Tests use no network:
`TestClient` drives the ASGI app in-process.

## Known G1a limits

- The mock does not rank by the question; selection is filter matching plus fixed scores. It exists
  to exercise display and compliance, not retrieval quality.
- Answers are a fixed sentence naming the top source. WP4 replaces both the data and the answer.
- All mock URLs are fictional. A citation with a missing or non-HTTP URL is dropped whole by
  `contract.mjs`, so an invalid response never reaches the DOM.
