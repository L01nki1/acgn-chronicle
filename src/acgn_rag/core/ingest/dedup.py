"""Deterministic deduplication clustering (`dedup.py`) for WP1 snapshots and Config B.

Implements `docs/INGEST_CONTRACT.md` §2, `docs/RUNBOOK.md` §3.3, and Issue #14 P2:
  - Primary channel: identical non-empty positive `media_ids` + matching non-null `event_type`
    within 72h (excluding `ingest_category == "corporate"`).
  - Auxiliary channel: same-language text similarity >= threshold (default 0.86) within 72h,
    merged via true agglomerative complete-linkage (highest-similarity pair first, never
    single-linkage or greedy first-fit) to prevent chain collapse.
  - Unresolved `media_ids == []` never merge via the entity channel.
  - Deterministic `cluster_id = "cluster_" + sha256(canonical_json(sorted(member_doc_ids)))[:16]`
    copied onto every member (including singletons), with full-digest collision checking.
  - Representative item is `min(member_doc_ids)`.

Owner: A (WP1) | Spec: docs/INGEST_CONTRACT.md §2, docs/RUNBOOK.md §3.3, Issue #14 P2, Issue #23
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from acgn_rag.core.ingest.normalize import NormalizedItem, canonical_json
from acgn_rag.core.store.payload import iso_to_epoch

DEDUP_POLICY_VERSION = "1"
DEFAULT_DEDUP_ALGORITHM = "entity-event-72h+agglomerative-complete:char3-tf-cosine-v1"
DEFAULT_SIMILARITY_THRESHOLD = 0.86
DEFAULT_WINDOW_HOURS = 72
SWEEP_THRESHOLDS: tuple[float, ...] = (0.82, 0.84, 0.86, 0.88)

_DIGIT_TOKENS_RE = re.compile(r"\d+")


class ClusterCollisionError(RuntimeError):
    """Raised if two distinct cluster member sets collide on their 16-hex short cluster_id."""


def _describe_similarity_fn(fn: Callable[[NormalizedItem, NormalizedItem], float]) -> str:
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__name__
    code = getattr(fn, "__code__", None)
    if code is not None:
        raw = f"{name}:{code.co_code.hex()}:{code.co_consts!r}".encode()
        digest = hashlib.sha256(raw).hexdigest()[:8]
        return f"custom:{name}:{digest}"
    return f"custom:{name}"


@dataclass(frozen=True)
class DedupPolicy:
    """Recorded deduplication policy included in `manifest.json` (`dedup_policy`)."""

    version: str = DEDUP_POLICY_VERSION
    algorithm: str = DEFAULT_DEDUP_ALGORITHM
    window_hours: int = DEFAULT_WINDOW_HOURS
    thresholds: dict[str, float] = field(
        default_factory=lambda: {"same_lang_cosine": DEFAULT_SIMILARITY_THRESHOLD}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "thresholds": dict(self.thresholds),
            "version": self.version,
            "window_hours": self.window_hours,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DedupPolicy:
        raw_thresh = data.get("thresholds") or {}
        return cls(
            version=str(data.get("version", DEDUP_POLICY_VERSION)),
            algorithm=str(data.get("algorithm", DEFAULT_DEDUP_ALGORITHM)),
            window_hours=int(data.get("window_hours", DEFAULT_WINDOW_HOURS)),
            thresholds={str(k): float(v) for k, v in raw_thresh.items()},
        )


@dataclass(frozen=True)
class DedupCluster:
    """One deduplication cluster written to `clusters.jsonl`."""

    cluster_id: str
    member_doc_ids: tuple[str, ...]
    representative_doc_id: str
    full_sha256: str

    def __post_init__(self) -> None:
        if not self.member_doc_ids:
            raise ValueError("DedupCluster must have at least one member_doc_id")
        sorted_members = tuple(sorted(set(self.member_doc_ids)))
        expected_cid, expected_sha = compute_cluster_id(sorted_members)
        if self.cluster_id != expected_cid:
            raise ValueError(
                f"Invalid cluster_id {self.cluster_id!r} for members {sorted_members}; "
                f"expected {expected_cid!r}"
            )
        if self.representative_doc_id != sorted_members[0]:
            raise ValueError(
                f"Invalid representative_doc_id {self.representative_doc_id!r}; "
                f"expected min(member_doc_ids)={sorted_members[0]!r}"
            )
        if self.full_sha256 and self.full_sha256 != expected_sha:
            raise ValueError(
                f"Invalid full_sha256 {self.full_sha256!r} for cluster {self.cluster_id!r}; "
                f"expected {expected_sha!r}"
            )
        object.__setattr__(self, "member_doc_ids", sorted_members)
        object.__setattr__(self, "representative_doc_id", sorted_members[0])
        object.__setattr__(self, "full_sha256", expected_sha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "full_sha256": self.full_sha256,
            "member_doc_ids": list(self.member_doc_ids),
            "representative_doc_id": self.representative_doc_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DedupCluster:
        members = tuple(str(m) for m in data["member_doc_ids"])
        if not members:
            raise ValueError("DedupCluster.from_dict requires non-empty member_doc_ids")
        cid, full_digest = compute_cluster_id(members)
        raw_cid = str(data.get("cluster_id") or cid)
        raw_rep = str(data.get("representative_doc_id") or min(members))
        raw_sha = str(data.get("full_sha256") or full_digest)
        return cls(
            cluster_id=raw_cid,
            member_doc_ids=members,
            representative_doc_id=raw_rep,
            full_sha256=raw_sha,
        )


@dataclass(frozen=True)
class DedupResult:
    """Complete output of `cluster_items`: member items with `cluster_id` + cluster records."""

    items: list[NormalizedItem]
    clusters: list[DedupCluster]
    policy: DedupPolicy

    @property
    def representatives(self) -> list[NormalizedItem]:
        """Return one deterministic representative `NormalizedItem` per cluster, sorted by `doc_id`."""
        rep_ids = {c.representative_doc_id for c in self.clusters}
        return [item for item in self.items if item.doc_id in rep_ids]


def compute_cluster_id(member_doc_ids: Iterable[str]) -> tuple[str, str]:
    """Compute `("cluster_" + sha256[:16], full_sha256)` over canonical JSON of sorted member IDs."""
    sorted_ids = sorted({str(doc_id) for doc_id in member_doc_ids if doc_id})
    if not sorted_ids:
        raise ValueError("Cannot compute cluster_id for empty member_doc_ids")
    digest = hashlib.sha256(canonical_json(sorted_ids)).hexdigest()
    return f"cluster_{digest[:16]}", digest


_NON_WORD_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def char_ngram_cosine(text_a: str, text_b: str, *, n: int = 3) -> float:
    """Deterministic character n-gram TF cosine similarity for same-language headline/lead text."""
    clean_a = _NON_WORD_RE.sub("", text_a.casefold())
    clean_b = _NON_WORD_RE.sub("", text_b.casefold())
    if not clean_a or not clean_b:
        return 0.0
    if clean_a == clean_b:
        return 1.0

    def _grams(s: str) -> Counter[str]:
        if len(s) <= n:
            return Counter({s: 1})
        return Counter(s[i : i + n] for i in range(len(s) - n + 1))

    counts_a = _grams(clean_a)
    counts_b = _grams(clean_b)
    shared_keys = set(counts_a) & set(counts_b)
    if not shared_keys:
        return 0.0
    dot = sum(counts_a[k] * counts_b[k] for k in shared_keys)
    norm_a = math.sqrt(sum(v * v for v in counts_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counts_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return min(1.0, max(0.0, dot / (norm_a * norm_b)))


def _has_conflicting_headline_numbers(title_a: str, title_b: str) -> bool:
    """Return True when two same-language headlines carry distinct numeric tokens (e.g.
    'December 8-14' vs 'December 15-21', or 'Monster Hunter Stories' vs 'Monster Hunter Stories 2').
    """
    nums_a = tuple(_DIGIT_TOKENS_RE.findall(title_a))
    nums_b = tuple(_DIGIT_TOKENS_RE.findall(title_b))
    if not nums_a and not nums_b:
        return False
    return nums_a != nums_b


_TEMPLATED_TITLE_RE = re.compile(r"プレスリリース\s*\d{4}年\d{1,2}月期")


def _strip_templated_title(text: str) -> str:
    return _TEMPLATED_TITLE_RE.sub("", text).strip()


def _default_item_similarity(a: NormalizedItem, b: NormalizedItem) -> float:
    if a.lang == b.lang and _has_conflicting_headline_numbers(a.title, b.title):
        return 0.0
    title_a = _strip_templated_title(a.title)
    title_b = _strip_templated_title(b.title)
    body_a = _strip_templated_title(a.body[:300])
    body_b = _strip_templated_title(b.body[:300])
    text_a = f"{title_a}\n{body_a}".strip()
    text_b = f"{title_b}\n{body_b}".strip()
    if not text_a or not text_b:
        return 0.0
    return char_ngram_cosine(text_a, text_b, n=3)


def _pair_merge_score(
    a: NormalizedItem,
    b: NormalizedItem,
    *,
    ts_a: int,
    ts_b: int,
    max_gap_s: int,
    sim_threshold: float,
    similarity_fn: Callable[[NormalizedItem, NormalizedItem], float],
) -> float | None:
    """Return a merge compatibility score `>= sim_threshold` if `a` and `b` can belong to the
    same event cluster, or `None` if they are incompatible under complete-linkage.
    """
    if abs(ts_a - ts_b) > max_gap_s:
        return None

    # Corporate / IR filings never merge via the primary entity channel
    either_corporate = a.ingest_category == "corporate" or b.ingest_category == "corporate"

    media_a = set(a.media_ids)
    media_b = set(b.media_ids)

    # If both items have resolved media_ids and they do not match the same work set, never merge
    if media_a and media_b and media_a != media_b:
        return None

    # If both items have explicit non-null event_types that conflict, do not merge
    if a.event_type is not None and b.event_type is not None and a.event_type != b.event_type:
        return None

    # 1. Primary channel: identical non-empty positive media_ids + matching non-null event_type within 72h
    if (
        not either_corporate
        and media_a
        and media_a == media_b
        and a.event_type is not None
        and a.event_type == b.event_type
    ):
        if a.lang == b.lang:
            # Same-language articles sharing (media_ids, event_type) must still share core headline topic
            if _has_conflicting_headline_numbers(a.title, b.title):
                return None
            title_sim = char_ngram_cosine(a.title, b.title, n=3)
            if title_sim < 0.30:
                return None
            return 1.0 + 0.01 * title_sim
        return 1.0

    # 2. Auxiliary channel: same language text/embedding similarity >= threshold within 72h
    if a.lang != "und" and a.lang == b.lang and sim_threshold <= 1.0:
        sim = float(similarity_fn(a, b))
        if sim >= sim_threshold:
            return sim

    return None


def _agglomerative_complete_linkage(
    sorted_items: list[NormalizedItem],
    *,
    timestamps: dict[str, int],
    max_gap_s: int,
    sim_threshold: float,
    similarity_fn: Callable[[NormalizedItem, NormalizedItem], float],
) -> list[list[NormalizedItem]]:
    """Deterministic agglomerative complete-linkage clustering with 72h time-window pruning.

    Merges the highest-similarity compatible pair first (breaking ties deterministically by
    `(min_doc_id_A, min_doc_id_B)`), and requires every pair `(u in A, v in B)` to be compatible.
    """
    n = len(sorted_items)
    if n <= 1:
        return [[it] for it in sorted_items]

    # Index items by time to prune pairwise comparisons outside `max_gap_s`
    time_ordered_indices = sorted(
        range(n),
        key=lambda idx: (timestamps[sorted_items[idx].doc_id], sorted_items[idx].doc_id),
    )

    pair_scores: dict[tuple[int, int], float] = {}
    neighbors: dict[int, set[int]] = {i: set() for i in range(n)}

    for pos_a, idx_a in enumerate(time_ordered_indices):
        item_a = sorted_items[idx_a]
        ts_a = timestamps[item_a.doc_id]
        for pos_b in range(pos_a + 1, n):
            idx_b = time_ordered_indices[pos_b]
            item_b = sorted_items[idx_b]
            ts_b = timestamps[item_b.doc_id]
            if ts_b - ts_a > max_gap_s:
                break
            lo, hi = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
            score = _pair_merge_score(
                sorted_items[lo],
                sorted_items[hi],
                ts_a=timestamps[sorted_items[lo].doc_id],
                ts_b=timestamps[sorted_items[hi].doc_id],
                max_gap_s=max_gap_s,
                sim_threshold=sim_threshold,
                similarity_fn=similarity_fn,
            )
            if score is not None:
                pair_scores[(lo, hi)] = score
                neighbors[lo].add(hi)
                neighbors[hi].add(lo)

    if not pair_scores:
        return [[it] for it in sorted_items]

    # Partition into connected components of the compatibility graph so agglomerative merging
    # only operates on small local components.
    visited: set[int] = set()
    groups: list[list[NormalizedItem]] = []

    for start_idx in range(n):
        if start_idx in visited:
            continue
        if not neighbors[start_idx]:
            visited.add(start_idx)
            groups.append([sorted_items[start_idx]])
            continue

        comp: list[int] = []
        queue = [start_idx]
        visited.add(start_idx)
        while queue:
            curr = queue.pop()
            comp.append(curr)
            for nb in neighbors[curr]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        comp.sort()

        # Run true agglomerative complete-linkage inside `comp`
        clusters_in_comp: dict[int, list[int]] = {cid: [cid] for cid in comp}

        def _cluster_pair_score(c1: list[int], c2: list[int]) -> float | None:
            min_s = math.inf
            for u in c1:
                for v in c2:
                    key = (u, v) if u < v else (v, u)
                    s = pair_scores.get(key)
                    if s is None:
                        return None
                    if s < min_s:
                        min_s = s
            return min_s

        while len(clusters_in_comp) > 1:
            active_ids = sorted(clusters_in_comp.keys())
            best_pair: tuple[int, int] | None = None
            best_key: tuple[float, str, str] | None = None

            for i_pos, cid_a in enumerate(active_ids):
                members_a = clusters_in_comp[cid_a]
                rep_a = sorted_items[members_a[0]].doc_id
                for cid_b in active_ids[i_pos + 1 :]:
                    members_b = clusters_in_comp[cid_b]
                    score = _cluster_pair_score(members_a, members_b)
                    if score is None:
                        continue
                    rep_b = sorted_items[members_b[0]].doc_id
                    # Maximize score; break ties by smallest (rep_a, rep_b)
                    if best_pair is None or best_key is None:
                        best_pair = (cid_a, cid_b)
                        best_key = (score, rep_a, rep_b)
                    else:
                        prev_score, prev_ra, prev_rb = best_key
                        if score > prev_score or (
                            score == prev_score and (rep_a, rep_b) < (prev_ra, prev_rb)
                        ):
                            best_pair = (cid_a, cid_b)
                            best_key = (score, rep_a, rep_b)

            if best_pair is None:
                break

            keep_cid, drop_cid = best_pair
            merged = sorted(clusters_in_comp[keep_cid] + clusters_in_comp.pop(drop_cid))
            clusters_in_comp[keep_cid] = merged

        for member_indices in clusters_in_comp.values():
            groups.append([sorted_items[idx] for idx in member_indices])

    groups.sort(key=lambda g: g[0].doc_id)
    return groups


def cluster_items(
    items: Sequence[NormalizedItem],
    *,
    policy: DedupPolicy | None = None,
    similarity_fn: Callable[[NormalizedItem, NormalizedItem], float] | None = None,
) -> DedupResult:
    """Cluster normalized items deterministically (order-independent) using agglomerative
    complete-linkage.

    Every returned `NormalizedItem` carries its assigned `cluster_id`.
    """
    base_policy = policy if policy is not None else DedupPolicy()
    sim_fn = similarity_fn if similarity_fn is not None else _default_item_similarity
    if similarity_fn is not None and similarity_fn is not _default_item_similarity:
        sim_tag = _describe_similarity_fn(similarity_fn)
        active_policy = DedupPolicy(
            version=base_policy.version,
            algorithm=f"{base_policy.algorithm}+{sim_tag}",
            window_hours=base_policy.window_hours,
            thresholds=dict(base_policy.thresholds),
        )
    else:
        active_policy = base_policy

    max_gap_s = int(active_policy.window_hours * 3600)
    sim_threshold = float(
        active_policy.thresholds.get("same_lang_cosine", DEFAULT_SIMILARITY_THRESHOLD)
    )

    # Sort deterministically by doc_id first so clustering is 100% invariant to input ordering
    sorted_items = sorted(items, key=lambda it: it.doc_id)
    if len(sorted_items) != len({it.doc_id for it in sorted_items}):
        raise ValueError("cluster_items requires unique doc_id values")

    if active_policy.algorithm.startswith("singleton-example"):
        # Exact singleton mode used by the synthetic format example
        raw_groups: list[list[NormalizedItem]] = [[it] for it in sorted_items]
    else:
        timestamps = {it.doc_id: iso_to_epoch(it.published_at) for it in sorted_items}
        raw_groups = _agglomerative_complete_linkage(
            sorted_items,
            timestamps=timestamps,
            max_gap_s=max_gap_s,
            sim_threshold=sim_threshold,
            similarity_fn=sim_fn,
        )

    clusters: list[DedupCluster] = []
    seen_short_ids: dict[str, str] = {}
    doc_to_cluster_id: dict[str, str] = {}

    for group in raw_groups:
        member_ids = tuple(sorted(m.doc_id for m in group))
        cid, full_digest = compute_cluster_id(member_ids)
        prior_digest = seen_short_ids.get(cid)
        if prior_digest is not None and prior_digest != full_digest:
            raise ClusterCollisionError(
                f"Short cluster_id collision on {cid!r}: {prior_digest} vs {full_digest}"
            )
        seen_short_ids[cid] = full_digest
        cluster = DedupCluster(
            cluster_id=cid,
            member_doc_ids=member_ids,
            representative_doc_id=member_ids[0],
            full_sha256=full_digest,
        )
        clusters.append(cluster)
        for doc_id in member_ids:
            doc_to_cluster_id[doc_id] = cid

    clusters.sort(key=lambda c: (c.representative_doc_id, c.cluster_id))
    clustered_items = [
        item.with_cluster_id(doc_to_cluster_id[item.doc_id]) for item in sorted_items
    ]
    return DedupResult(items=clustered_items, clusters=clusters, policy=active_policy)


def sweep_thresholds(
    items: Sequence[NormalizedItem],
    *,
    thresholds: Sequence[float] = SWEEP_THRESHOLDS,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    similarity_fn: Callable[[NormalizedItem, NormalizedItem], float] | None = None,
) -> list[dict[str, Any]]:
    """Run a threshold sweep (e.g. 0.82 / 0.84 / 0.86 / 0.88) and summarize cluster statistics."""
    results: list[dict[str, Any]] = []
    for thresh in thresholds:
        pol = DedupPolicy(
            window_hours=window_hours,
            thresholds={"same_lang_cosine": float(thresh)},
        )
        res = cluster_items(items, policy=pol, similarity_fn=similarity_fn)
        multi_clusters = [c for c in res.clusters if len(c.member_doc_ids) > 1]
        results.append(
            {
                "threshold": float(thresh),
                "item_count": len(res.items),
                "cluster_count": len(res.clusters),
                "multi_member_clusters": len(multi_clusters),
                "max_cluster_size": max((len(c.member_doc_ids) for c in res.clusters), default=0),
                "dedup_reduction_rate": round(
                    1.0 - (len(res.clusters) / max(len(res.items), 1)), 4
                ),
            }
        )
    return results


__all__ = [
    "DEDUP_POLICY_VERSION",
    "DEFAULT_DEDUP_ALGORITHM",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_WINDOW_HOURS",
    "SWEEP_THRESHOLDS",
    "ClusterCollisionError",
    "DedupCluster",
    "DedupPolicy",
    "DedupResult",
    "char_ngram_cosine",
    "cluster_items",
    "compute_cluster_id",
    "sweep_thresholds",
]
