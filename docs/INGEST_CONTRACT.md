# A's ingestion and snapshot handoff v1

Owner: A. Consumer review: E (WP1), F (WP2), B (wire semantics).
Decision date: 2026-11-01. References: #23, #30, #31, PR #26 and PR #32.

This is A's producer commitment for implementation and review, not a claim that
the crawler, entity registry, clustering or snapshot publisher already runs.
No public wire field is changed here. A disagreement is resolved in the linked PR
and recorded in DECISIONS before a real evaluation snapshot is minted.

## 1. NormalizedItem consumed by F

One internal JSONL row per article; snake_case keys, not camelCase aliases.

| Field | Type | Producer rule |
|---|---|---|
| `doc_id` | string | `doc_` + full lowercase SHA-256 of canonical original URL encoded as UTF-8; does not change when the body changes |
| `title` | string | Source title; no navigation or advertising boilerplate |
| `body` | string | Clean article text; internal-only; never an outbound response |
| `url` | string | Canonical original HTTPS URL, including for archived fetches |
| `publisher` | string | Attributed publisher's display name; open text, not an identity key |
| `source_type` | `ports.SourceType` | Per-item `official` / `media` / `metadata`; assigned at ingestion |
| `published_at` | string | Aware ISO-8601, normalized to UTC `Z`; retain nonzero fractional seconds |
| `lang` | string | Lowercase language tag; initially `ja`, `en`, `zh`; unknown language is explicit `und` |
| `media_ids` | list[int] | Sorted unique positive local entity-registry IDs; `[]` means unresolved/no established link |
| `ingest_category` | string or null | Optional, default null; `corporate` is the #17 IR-prefilter tag, never a fourth SourceType |
| `cluster_id` | string or null | Default null before dedup; required and non-null on every frozen member, including singletons |
| `event_type` | string or null | Default null; internal clustering input, not an inferred answer fact |
| `source_key` | string | Stable adapter/feed key in the frozen source registry; used for source counts, not publisher identity |

The first nine fields are exactly F's requested input seam. Extra internal fields
do not require a Citation or FactUpdate change. F can select those nine fields for
IndexDoc/payload_from_fields. A will supply a snapshot iterator of member rows;
the index input is the representative subset described in section 3.

String normalization: Unicode NFC and CRLF/CR to LF; do not collapse internal body
whitespace after hashing. URL normalization removes fragments, lowercases scheme
and host and removes default ports. Preserve path case, query parameters and trailing
slash unless a verified redirect/canonical tag establishes equivalence. Conflicting
same-URL observations fail reconciliation rather than receiving an arbitrary winner.

### Entity linking and source/publisher identity

A produces media_ids in facts_link before dedup and freezes a versioned local
entity registry with external aliases `(namespace, external_id)` and title aliases.
Do not put bare AniList and Bangumi IDs in one list: their integer namespaces can
collide and games need coverage too. Existing local IDs are never reused or
renumbered. Ambiguous links remain empty and are counted. F/E consume that one
registry; they do not create independent mappings. The registry version and digest
are in the manifest. No populated registry or linking accuracy is claimed yet.

For B's 2026-11-01 publisher question on #23: no publisher_id is adopted. A feed key
identifies an acquisition channel, not the attributed organization (PR TIMES can
carry many organizations). Publisher remains display text. There is no actual
publisher identity/alias registry to publish yet, so deriving IDs from display
names would invent stability. A future proposal must include a real versioned
organization registry, alias/rename rules, a NormalizedItem -> F payload -> Citation
mapping and an absent/default rule for old rows; B arbitrates any Optional wire
addition and notifies C/D/E/F. This handoff adds no such field.

## 2. Dedup-cluster identity

A assigns clusters after linking, using entity/event compatibility and the 72-hour
event window. Same-language text similarity is auxiliary. Unresolved entities do
not justify merging unrelated records; absent other verified evidence they remain
singletons. Threshold calibration and cluster quality remain separate work.

`cluster_id = "cluster_" + sha256(canonical_json(sorted(unique_member_doc_ids)))[:16]`.
canonical_json is defined below. The ID is copied onto EVERY member, including a
singleton. Sort by doc_id; the lexicographically smallest doc_id is the deterministic
representative. Preserve all members and their original URLs for config B and
gold-source resolution. Representative choice is a reproducibility rule, not a
claim of source authority. Record the full cluster digest; reject short-ID
collisions before publication. Same members produce the same ID regardless of
input order. Changed membership can change the ID in a new snapshot: no promise
of immutable cluster IDs across different snapshots is made.

## 3. Frozen snapshot format and canonical manifest

Proposed internal files under `data/snapshots/<full_snapshot_id>/` (and `data/snapshots_daily/<full_snapshot_id>/` for unmixed daily increments):

- `items.jsonl`: all normalized member rows, sorted by doc_id; raw/internal only.
- `representatives.jsonl`: one representative row per cluster, sorted by doc_id; consumed by ordinary single-article chunk configs (`A`, `A'`, `A0`).
- `clusters.jsonl`: cluster_id, sorted member_doc_ids, representative_doc_id.
- `entities.json` and `sources.json`: frozen versioned registries, including alias
  mappings and source-key definitions actually used by that corpus. Seed entities use positive integer IDs `< 1,000`; dynamic external metadata refs (`anilist`, `bangumi`, `syosetu`) receive history-free SHA-256-derived positive IDs `>= 1,000,000`, verified against the append-only cross-build registry at `data/registry/entities.json`.
- `quarantine.jsonl`: quarantined raw rows with failure reasons (`missing_tz`, `garbled_text`, `bad_url`, `conflicting_observation`).
- `manifest.json`: canonical identity object below.
- `snapshot.json`: CorpusSnapshot statistics, `ingest_report` counters, `partial` flag, and operational creation time.

Manifest v1 contains `manifest_version`, `window_start`, `window_end`,
`normalization_version`, `entity_registry_sha256`, `source_registry_sha256`,
`dedup_policy` (version, algorithm, thresholds, window_hours), and sorted `items`.
Each item is the complete normalized row except body, plus `body_sha256` of its
normalized UTF-8 body. Explicit null/default values are emitted. media_ids are
sorted unique; doc_ids must be unique. All retrieval-relevant text and metadata,
cluster membership, registry mappings and the dedup policy participate in identity.

`canonical_json(value)` means Python
`json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)`
encoded as UTF-8, followed by exactly one LF. No BOM, absolute paths, fetch times,
created_at, build time, credentials or environment-dependent values enter identity.
Every hash below uses those exact bytes, except body/URL hashes which use their
specified text bytes. Versions are explicit and all strings have the normalization
above. Snapshot windows are inclusive UTC calendar-day bounds `YYYY-MM-DD`.
Adapter timestamps lacking an offset need a documented source timezone; otherwise
they fail/quarantine and are reported, never silently interpreted in host time.

`manifest_sha256 = sha256(canonical_json(manifest)).hexdigest()`.
Full ID = `window_end.replace("-", "") + "-" + manifest_sha256[:12]`.
Thus `YYYYMMDD-<12 lowercase hex>` uses the frozen window-end UTC day, NOT rebuild
time. Store the full digest and reject a same-ID/different-digest collision before
writing or overwriting anything. A body change, metadata change, added/removed
document or changed registry/policy changes identity (subject to collision rejection).

`item_count` is all unique normalized member rows; `deduped_count` is cluster count;
`source_count` counts distinct source_key values with accepted rows. Index ordinary
configs from one representative per cluster; B reads all cluster members. Preserve
member URL -> representative/cluster mappings for evaluation: do not silently treat
a removed syndicated URL as unavailable without the agreed E/F resolution rule.
window_start/end are the configured frozen bounds, not observed min/max timestamps;
record observed coverage separately and report gaps. Reject out-of-window rows.
created_at is operational UTC provenance outside the hash.

See `docs/examples/ingest-manifest-v1.json` and `ingest-identity-proof.json` for a
small SYNTHETIC example and two-build byte/hash comparison. They are format evidence,
not real corpus counts or evidence that the ingestion pipeline passes G1a.

Recheck the committed example without network, models or project dependencies:

```python
import copy
import hashlib
import json
from pathlib import Path

def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

root = Path("docs/examples")
original = (root / "ingest-manifest-v1.json").read_bytes()
manifest = json.loads(original)
proof = json.loads((root / "ingest-identity-proof.json").read_bytes())
def rebuild(value):
    result = copy.deepcopy(value)
    result["items"].sort(key=lambda row: row["doc_id"])
    return canonical(result)

assert rebuild(manifest) == original
manifest["items"].reverse()
assert rebuild(manifest) == original
assert hashlib.sha256(original).hexdigest() == proof["build_1"]["sha256"]
assert proof["build_1"] == proof["build_2_reversed_input"]
changed = copy.deepcopy(manifest)
changed["items"][0]["body_sha256"] = hashlib.sha256(b"different body").hexdigest()
assert hashlib.sha256(rebuild(changed)).hexdigest() != proof["build_1"]["sha256"]
```

## 4. Storage/publication agreement with F, E and C

- Keep the full snapshot ID end to end. Reject the old truncated-name convention;
  F owns the safe physical mapping in #31/PR #32. Read actual collection names from
  validated matching build.json/identity.json, never derive a shortened name.
- Labels: A, A-prime (variant A'), A0, B. A does not modify F's store implementation.
- Prepare corpus files in a unique staging directory; validate hashes, counts and
  required indexes before publishing. An atomic CURRENT replacement is the final
  serving switch, not evidence that prior steps completed. A/C/F must implement a
  catalog/build-generation completion protocol in W2: interrupted --fresh with old
  sidecars must remain unserved/503. Never overwrite a served generation in place.
- Retain current + previous by validated publication time, plus any actively pinned
  request/evaluation snapshots even if more than two are then retained. Record
  retired completed IDs; never infer retirement merely from missing storage.
- Reads bind existing verified storage only. Unknown 404, known retired 410,
  unserved/incomplete/inconsistent 503; no read-path 409, no automatic snapshot
  fallback. Runner pins full snapshot/config/build identity for the complete run.
- Leases, active-reader protection, catalog and HTTP behavior are target agreements,
  NOT implemented or validated by this documentation-only handoff.

## 5. Failure and evidence rules

Broken selector, unexpected empty historical extraction, 403, exhausted 429/timeout,
unresolved decoding failures, malformed timestamps and conflicting identities are
visible failures with source/URL/reason and counters. Never hide them as return [].
Honor robots, pacing, Retry-After and cooldown; never bypass access restrictions.
Expected no-new-items/304 on an incremental run is valid and distinguished from
parser failure. Partial crawls are logged and not published as complete snapshots.

Two independent acceptance checks remain: (1) rebuilding the SAME frozen normalized
input yields identical manifest bytes/full ID; (2) repeated acquisition with no new
content inserts zero records, while changed content creates a new version/snapshot.
The synthetic identity proof only addresses the canonicalization part of (1).
Real HTTP, adapter and production two-rebuild evidence remains pending.

The production evaluation window is NOT selected by this example. A supplies the
source-depth probe; A/D record the actual window and coordinate #28's out-of-window
gold evidence before freeze. Counts must come from real data. QA metric denominators
follow D's ruling on #30: eligible question sets, not deduped_count.
