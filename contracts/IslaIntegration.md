# Isla integration boundary

Owner: F. This note defines the future Chronicle-to-Isla consumer boundary. This change does not
modify the Isla repository and does not claim that Chronicle's W2 HTTP host is already available.
The normative wire shape remains
[`knowledge_port.schema.json`](knowledge_port.schema.json), with semantics in
[`KnowledgePort.md`](KnowledgePort.md).

## Consumer flow

1. Read `GET /snapshots` and select a full retained `snapshot_id`. Never invent, shorten, or silently
   replace the identifier.
2. Pin that full identifier and the trusted deployment/configuration identity for the whole request or
   run. A changed `CURRENT` pointer must not redirect an in-flight operation.
3. Send the user's question to `POST /retrieve` or `POST /ask`. Preserve the requested filters and
   positive `top_k`; an explicit filter with no matches remains an empty evidence set.
4. Accept only schema-valid excerpts or grounded answers. Every rendered citation retains its URL,
   publisher display name, source type, publication date, and chunk identifier.
5. Cache only inside the selected deployment/build namespace, keyed by
   `(snapshot_id, text, filters, top_k, config)`. Clear the cache when that trusted context changes.

The port returns excerpts of at most 200 Unicode code points, never full article bodies. Isla must not
fetch a citation URL to expand the returned quote. `publisher` is display text, not a canonical
organization identifier. `source_type` is the closed `official` / `media` / `metadata` value.

## Curl example

The identifier below is illustrative. A real client first obtains a retained full identifier from
`GET /snapshots` and substitutes it without shortening it.

```sh
curl --fail-with-body --silent --show-error \
  --request POST 'http://127.0.0.1:8000/retrieve' \
  --header 'Content-Type: application/json' \
  --data '{
    "text": "When does the second season start airing?",
    "top_k": 5,
    "snapshot_id": "20261028-abcdef012345",
    "filters": {
      "source_types": ["official"],
      "since": "2023-10-28"
    }
  }'
```

The client rejects a response that omits citation provenance or exceeds the quote budget. A valid
restrictive no-match response from `/retrieve` is an empty array; `/ask` must instead return a
schema-valid abstention with no citations and a non-empty reason.

## Failure handling

| Status | Isla action |
|---|---|
| `400` | Correct the request shape or enum; do not retry. |
| `404` | Refresh the inventory and ask for an explicit selection; never retarget automatically. |
| `410` | Report that the pinned snapshot was retired and offer an explicit re-run. |
| `422` | Correct the date or filter semantics; do not retry. |
| `503` | Honour integer `Retry-After` seconds and retry with bounded backoff. |

Every non-success response has `{"detail": string}`. An unavailable or inconsistent inventory is a
`503`, not evidence that a snapshot was never known. Isla records the failed pinned identifier rather
than attaching an answer to a different corpus.

## Deferred implementation

This repository carries only the contract and this example. A later Isla change may add Chronicle
output to its designated context/provenance adapters after the HTTP host has real W2 captures and
consumer tests. It must not copy Chronicle storage code, expose raw article bodies, create a second
snapshot definition, or treat the conditional `/daily` endpoint as available before G3.
