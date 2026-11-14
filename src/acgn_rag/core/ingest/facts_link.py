"""Versioned local integer entity registry (`entities.json`), source feed registry (`sources.json`),
and deterministic pre-clustering entity linker (`facts_link.py`).

Enforces `docs/INGEST_CONTRACT.md` §1 & §3:
  - Local positive integer `media_id`s are distinct from bare AniList / Bangumi / Syosetu IDs.
  - External IDs are stored as namespaced pairs `(namespace, external_id)`.
  - Ambiguous title aliases that map to multiple entities are left unresolved (`[]`) and counted.
  - Both registries serialize via `canonical_json` so their SHA-256 digests participate in
    `manifest.json`.
  - Entity IDs are persistent and append-only across builds (never renumbered or reused).

Owner: A (WP1) | Spec: docs/INGEST_CONTRACT.md §1 & §3, Issue #9, Issue #23
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from acgn_rag.core.ingest.normalize import (
    NormalizedItem,
    RawIngestItem,
    canonical_json,
    classify_content_category,
    normalize_text,
)

ENTITY_REGISTRY_VERSION = "1"
SOURCE_REGISTRY_VERSION = "1"
DEFAULT_ENTITY_REGISTRY_PATH = Path("data/registry/entities.json")

# Extract work titles enclosed in JP/ZH brackets 『...』 or 《...》
_BRACKET_TITLE_RE = re.compile(r"[『《]([^『』《》\n]{2,60})[』》]")


def _norm_alias(alias: str) -> str:
    """Case-folded NFC alias key for exact title/alias lookup."""
    cleaned = normalize_text(alias).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.casefold()


def _deterministic_ext_media_id(
    namespace: str,
    external_id: str,
) -> int:
    """Derive a history-free positive integer `media_id` (>= 1_000_000) purely from
    `(namespace, external_id)` so ID assignment never depends on build history or crawl subset.
    """
    key_bytes = f"{namespace.lower().strip()}:{external_id.strip()}".encode()
    digest_int = int.from_bytes(hashlib.sha256(key_bytes).digest()[:6], "big")
    return 1_000_000 + (digest_int % 900_000_000_000)


@dataclass(frozen=True)
class EntityEntry:
    """One work/franchise entry in the local integer entity registry (`entities.json`)."""

    media_id: int
    canonical_title: str
    category: Literal["anime", "manga", "game", "light_novel"] = "anime"
    aliases: tuple[str, ...] = ()
    external_ids: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.media_id <= 0:
            raise ValueError(f"media_id must be a positive integer, got {self.media_id}")
        if not self.canonical_title.strip():
            raise ValueError(f"canonical_title must not be empty for media_id={self.media_id}")
        sorted_aliases = tuple(
            sorted(
                {
                    normalize_text(a).strip()
                    for a in (self.canonical_title, *self.aliases)
                    if a.strip()
                }
            )
        )
        sorted_ext = tuple(
            sorted(
                {
                    (str(ns).strip().lower(), str(eid).strip())
                    for ns, eid in self.external_ids
                    if ns and eid
                }
            )
        )
        object.__setattr__(self, "aliases", sorted_aliases)
        object.__setattr__(self, "external_ids", sorted_ext)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aliases": list(self.aliases),
            "canonical_title": self.canonical_title,
            "category": self.category,
            "external_ids": [
                {"external_id": ext_id, "namespace": ns} for ns, ext_id in self.external_ids
            ],
            "media_id": self.media_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EntityEntry:
        ext_raw = data.get("external_ids") or []
        ext_pairs = tuple(
            (str(item["namespace"]), str(item["external_id"]))
            for item in ext_raw
            if isinstance(item, Mapping) and item.get("namespace") and item.get("external_id")
        )
        return cls(
            media_id=int(data["media_id"]),
            canonical_title=str(data["canonical_title"]),
            category=data.get("category", "anime"),
            aliases=tuple(str(a) for a in (data.get("aliases") or [])),
            external_ids=ext_pairs,
        )


@dataclass(frozen=True)
class EntityRegistry:
    """Frozen versioned local entity registry (`entities.json`)."""

    version: str = ENTITY_REGISTRY_VERSION
    entries: tuple[EntityEntry, ...] = ()

    def __post_init__(self) -> None:
        seen_ids: set[int] = set()
        seen_ext: dict[tuple[str, str], int] = {}
        sorted_entries = tuple(sorted(self.entries, key=lambda e: e.media_id))
        for entry in sorted_entries:
            if entry.media_id in seen_ids:
                raise ValueError(f"Duplicate media_id in EntityRegistry: {entry.media_id}")
            seen_ids.add(entry.media_id)
            for ns, eid in entry.external_ids:
                key = (ns.lower().strip(), eid.strip())
                prior = seen_ext.get(key)
                if prior is not None and prior != entry.media_id:
                    raise ValueError(
                        f"Conflicting external_id {key} mapped to both media_id={prior} and {entry.media_id}"
                    )
                seen_ext[key] = entry.media_id
        object.__setattr__(self, "entries", sorted_entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [entry.to_dict() for entry in self.entries],
            "entity_registry_version": self.version,
        }

    def to_canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_canonical_bytes())

    @classmethod
    def load(cls, path: Path) -> EntityRegistry:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = tuple(EntityEntry.from_dict(row) for row in payload.get("entities", []))
        return cls(
            version=str(payload.get("entity_registry_version", ENTITY_REGISTRY_VERSION)),
            entries=entries,
        )


@dataclass(frozen=True)
class SourceRegistryEntry:
    """One acquisition channel / feed definition in `sources.json` (not a publisher identity)."""

    source_key: str
    adapter_family: str
    default_publisher: str
    default_source_type: Literal["official", "media", "metadata"]
    lang: str
    source_tz: str
    endpoint: str
    canonical_host: str | None = None
    categories: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "adapter_family": self.adapter_family,
            "default_publisher": self.default_publisher,
            "default_source_type": self.default_source_type,
            "endpoint": self.endpoint,
            "lang": self.lang,
            "source_key": self.source_key,
            "source_tz": self.source_tz,
        }
        if self.canonical_host:
            out["canonical_host"] = self.canonical_host
        if self.categories:
            out["categories"] = [int(c) for c in self.categories]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceRegistryEntry:
        raw_cats = data.get("categories") or ()
        return cls(
            source_key=str(data["source_key"]),
            adapter_family=str(data["adapter_family"]),
            default_publisher=str(data["default_publisher"]),
            default_source_type=data["default_source_type"],
            lang=str(data["lang"]),
            source_tz=str(data["source_tz"]),
            endpoint=str(data["endpoint"]),
            canonical_host=str(data["canonical_host"]) if data.get("canonical_host") else None,
            categories=tuple(int(c) for c in raw_cats),
        )


@dataclass(frozen=True)
class SourceRegistry:
    """Frozen versioned source/feed registry (`sources.json`)."""

    version: str = SOURCE_REGISTRY_VERSION
    sources: tuple[SourceRegistryEntry, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        sorted_sources = tuple(sorted(self.sources, key=lambda s: s.source_key))
        for src in sorted_sources:
            if src.source_key in seen:
                raise ValueError(f"Duplicate source_key in SourceRegistry: {src.source_key!r}")
            seen.add(src.source_key)
        object.__setattr__(self, "sources", sorted_sources)

    def filter_used(self, used_source_keys: Iterable[str]) -> SourceRegistry:
        """Return a `SourceRegistry` containing only the source definitions actually used
        by the corpus (`docs/INGEST_CONTRACT.md` §3 L84-85).
        """
        key_set = {str(k).strip() for k in used_source_keys if str(k).strip()}
        if not key_set:
            return self
        filtered = tuple(s for s in self.sources if s.source_key in key_set)
        if not filtered:
            return self
        return SourceRegistry(version=self.version, sources=filtered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_registry_version": self.version,
            "sources": [s.to_dict() for s in self.sources],
        }

    def to_canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_canonical_bytes())

    @classmethod
    def load(cls, path: Path) -> SourceRegistry:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = tuple(SourceRegistryEntry.from_dict(row) for row in payload.get("sources", []))
        return cls(
            version=str(payload.get("source_registry_version", SOURCE_REGISTRY_VERSION)),
            sources=entries,
        )


@dataclass(frozen=True)
class LinkingStats:
    """Counters reported after running `EntityLinker` over a batch."""

    total_items: int
    linked_items: int
    unresolved_items: int
    ambiguous_hits: int

    def to_dict(self) -> dict[str, int]:
        return {
            "ambiguous_hits": self.ambiguous_hits,
            "linked_items": self.linked_items,
            "total_items": self.total_items,
            "unresolved_items": self.unresolved_items,
        }


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _find_alias_spans(haystack: str, norm_alias: str) -> list[tuple[int, int]]:
    """Find non-overlapping occurrences of `norm_alias` in `haystack`, enforcing word boundaries
    for ASCII aliases so short words/prefixes do not match inside longer words.
    """
    spans: list[tuple[int, int]] = []
    is_ascii = norm_alias.isascii()
    start = 0
    alias_len = len(norm_alias)
    while True:
        idx = haystack.find(norm_alias, start)
        if idx == -1:
            break
        end = idx + alias_len
        if is_ascii:
            left_ok = idx == 0 or not haystack[idx - 1].isalnum()
            right_ok = end == len(haystack) or not haystack[end].isalnum()
            if not (left_ok and right_ok):
                start = idx + 1
                continue
        spans.append((idx, end))
        start = end
    return spans


@dataclass
class EntityLinker:
    """Deterministic entity linker mapping external refs and work titles/aliases to `media_ids`."""

    registry: EntityRegistry
    _by_ext: dict[tuple[str, str], int] = field(init=False, default_factory=dict, repr=False)
    _by_alias: dict[str, set[int]] = field(init=False, default_factory=dict, repr=False)
    _unambiguous_aliases: list[tuple[str, str, int]] = field(
        init=False, default_factory=list, repr=False
    )

    def __post_init__(self) -> None:
        for entry in self.registry.entries:
            for ns, ext_id in entry.external_ids:
                key = (ns.lower().strip(), ext_id.strip())
                existing = self._by_ext.get(key)
                if existing is not None and existing != entry.media_id:
                    raise ValueError(
                        f"Conflicting external ID {key} mapped to both {existing} and {entry.media_id}"
                    )
                self._by_ext[key] = entry.media_id

            for alias in entry.aliases:
                norm = _norm_alias(alias)
                if len(norm) < 2:
                    continue
                self._by_alias.setdefault(norm, set()).add(entry.media_id)

        # Longer aliases matched first so specific titles (e.g. "Monster Hunter Wilds")
        # consume their character span before shorter prefix aliases ("Monster Hunter").
        unambiguous: list[tuple[str, str, int]] = []
        for entry in self.registry.entries:
            for alias in entry.aliases:
                norm = _norm_alias(alias)
                ids = self._by_alias.get(norm, set())
                if len(ids) == 1 and len(norm) >= 2:
                    unambiguous.append((alias, norm, entry.media_id))
        unambiguous.sort(key=lambda t: (-len(t[1]), t[1], t[2]))
        self._unambiguous_aliases = unambiguous

    def resolve_item(
        self,
        *,
        title: str,
        body: str = "",
        existing_media_ids: Iterable[int] = (),
        external_refs: Iterable[tuple[str, str]] = (),
    ) -> tuple[list[int], int]:
        """Return `(sorted_media_ids, ambiguous_hit_count)` for one article."""
        matched: set[int] = {int(m) for m in existing_media_ids if int(m) > 0}
        ambiguous_hits = 0

        for ns, ext_id in external_refs:
            found = self._by_ext.get((ns.lower().strip(), ext_id.strip()))
            if found is not None:
                matched.add(found)

        # 1. Check bracketed titles 『...』 / 《...》 in title and lead paragraph
        lead_text = f"{title}\n{body[:500]}"
        for bracket_match in _BRACKET_TITLE_RE.finditer(lead_text):
            candidate = _norm_alias(bracket_match.group(1))
            ids = self._by_alias.get(candidate)
            if not ids:
                continue
            if len(ids) == 1:
                matched.update(ids)
            else:
                ambiguous_hits += 1

        # 2. Longest-match-first span scan over title and lead text for unambiguous aliases
        haystack = _norm_alias(lead_text)
        occupied_spans: list[tuple[int, int]] = []
        for _raw_alias, norm_alias, media_id in self._unambiguous_aliases:
            if len(norm_alias) < 3 and norm_alias.isascii():
                continue
            candidate_spans = _find_alias_spans(haystack, norm_alias)
            for span in candidate_spans:
                if not any(_spans_overlap(span, occ) for occ in occupied_spans):
                    occupied_spans.append(span)
                    matched.add(media_id)

        # Check if any explicitly ambiguous alias appears in the title
        title_norm = _norm_alias(title)
        for norm_alias, ids in self._by_alias.items():
            if len(ids) > 1 and len(norm_alias) >= 3 and norm_alias in title_norm:
                ambiguous_hits += 1

        return sorted(matched), ambiguous_hits

    def link_normalized_items(
        self,
        items: Sequence[NormalizedItem],
        *,
        external_refs_by_doc_id: Mapping[str, Sequence[tuple[str, str]]] | None = None,
    ) -> tuple[list[NormalizedItem], LinkingStats]:
        """Attach resolved `media_ids` to each `NormalizedItem` in `items`."""
        linked_list: list[NormalizedItem] = []
        linked_count = 0
        unresolved_count = 0
        total_ambiguous = 0

        for item in items:
            ext_refs = (
                external_refs_by_doc_id.get(item.doc_id, ()) if external_refs_by_doc_id else ()
            )
            media_ids, amb = self.resolve_item(
                title=item.title,
                body=item.body,
                existing_media_ids=item.media_ids,
                external_refs=ext_refs,
            )
            total_ambiguous += amb
            updated = item.with_media_ids(media_ids)
            if updated.media_ids:
                linked_count += 1
            else:
                unresolved_count += 1
            linked_list.append(updated)

        stats = LinkingStats(
            total_items=len(items),
            linked_items=linked_count,
            unresolved_items=unresolved_count,
            ambiguous_hits=total_ambiguous,
        )
        return linked_list, stats


SEED_ENTITIES: tuple[EntityEntry, ...] = (
    EntityEntry(
        media_id=101,
        canonical_title="Monster Hunter Wilds",
        category="game",
        aliases=(
            "Monster Hunter Wilds",
            "モンスターハンターワイルズ",
            "モンハンワイルズ",
            "怪物猎人\uff1a荒野",
            "怪物猎人荒野",
        ),
        external_ids=(("capcom", "mhwilds"),),
    ),
    EntityEntry(
        media_id=102,
        canonical_title="Monster Hunter",
        category="game",
        aliases=("Monster Hunter", "モンスターハンター", "怪物猎人", "Monster Hunter Now"),
    ),
    EntityEntry(
        media_id=103,
        canonical_title="Street Fighter 6",
        category="game",
        aliases=("Street Fighter 6", "ストリートファイター6", "スト6", "街头霸王6", "街霸6"),
    ),
    EntityEntry(
        media_id=104,
        canonical_title="Resident Evil",
        category="game",
        aliases=("Resident Evil", "バイオハザード", "生化危机", "BIOHAZARD"),
    ),
    EntityEntry(
        media_id=105,
        canonical_title="Dragon's Dogma 2",
        category="game",
        aliases=("Dragon's Dogma 2", "ドラゴンズドグマ 2", "ドラゴンズドグマ2", "龙之信条2"),
    ),
    EntityEntry(
        media_id=106,
        canonical_title="Ace Attorney",
        category="game",
        aliases=("Ace Attorney", "逆転裁判", "逆转裁判", "逆転検事"),
    ),
    EntityEntry(
        media_id=201,
        canonical_title="One Piece",
        category="anime",
        aliases=("One Piece", "ワンピース", "ONE PIECE", "航海王", "海贼王"),
        external_ids=(("anilist", "21"), ("bangumi", "975")),
    ),
    EntityEntry(
        media_id=202,
        canonical_title="Dragon Ball",
        category="anime",
        aliases=(
            "Dragon Ball",
            "ドラゴンボール",
            "ドラゴンボールDAIMA",
            "Dragon Ball Daima",
            "龙珠",
        ),
    ),
    EntityEntry(
        media_id=203,
        canonical_title="Precure",
        category="anime",
        aliases=(
            "Precure",
            "プリキュア",
            "わんだふるぷりきゅあ",
            "キミとアイドルプリキュア",
            "光之美少女",
        ),
    ),
    EntityEntry(
        media_id=204,
        canonical_title="Digimon",
        category="anime",
        aliases=("Digimon", "デジモン", "数码宝贝"),
    ),
    EntityEntry(
        media_id=205,
        canonical_title="Girls Band Cry",
        category="anime",
        aliases=("Girls Band Cry", "ガールズバンドクライ", "ガルクラ", "GIRLS BAND CRY"),
    ),
    EntityEntry(
        media_id=206,
        canonical_title="Jujutsu Kaisen",
        category="anime",
        aliases=("Jujutsu Kaisen", "呪術廻戦", "咒术回战"),
        external_ids=(("anilist", "113415"),),
    ),
    EntityEntry(
        media_id=207,
        canonical_title="Chainsaw Man",
        category="anime",
        aliases=("Chainsaw Man", "チェンソーマン", "链锯人", "电锯人"),
        external_ids=(("anilist", "127230"),),
    ),
    EntityEntry(
        media_id=208,
        canonical_title="Attack on Titan",
        category="anime",
        aliases=("Attack on Titan", "進撃の巨人", "进击的巨人"),
    ),
    EntityEntry(
        media_id=209,
        canonical_title="My Hero Academia",
        category="anime",
        aliases=("My Hero Academia", "僕のヒーローアカデミア", "ヒロアカ", "我的英雄学院"),
    ),
    EntityEntry(
        media_id=210,
        canonical_title="Frieren: Beyond Journey's End",
        category="anime",
        aliases=(
            "Frieren: Beyond Journey's End",
            "Frieren",
            "葬送のフリーレン",
            "葬送的芙莉莲",
        ),
        external_ids=(("anilist", "154587"), ("bangumi", "400602")),
    ),
    EntityEntry(
        media_id=301,
        canonical_title="That Time I Got Reincarnated as a Slime",
        category="light_novel",
        aliases=(
            "That Time I Got Reincarnated as a Slime",
            "転生したらスライムだった件",
            "転スラ",
            "关于我转生变成史莱姆这档事",
        ),
        external_ids=(("syosetu", "n6316bn"),),
    ),
    EntityEntry(
        media_id=302,
        canonical_title="The Apothecary Diaries",
        category="light_novel",
        aliases=("The Apothecary Diaries", "薬屋のひとりごと", "药屋少女的呢喃"),
        external_ids=(("syosetu", "n9636x"),),
    ),
    EntityEntry(
        media_id=303,
        canonical_title="Re:Zero -Starting Life in Another World-",
        category="light_novel",
        aliases=("Re:Zero", "Re:ゼロから始める異世界生活", "リゼロ", "从零开始的异世界生活"),
        external_ids=(("syosetu", "n2267be"),),
    ),
    EntityEntry(
        media_id=304,
        canonical_title="Mushoku Tensei: Jobless Reincarnation",
        category="light_novel",
        aliases=("Mushoku Tensei", "無職転生", "无职转生"),
        external_ids=(("syosetu", "n9669bk"),),
    ),
)


def build_entity_registry_from_raw(
    raw_items: Sequence[RawIngestItem] = (),
    *,
    seed_entries: Sequence[EntityEntry] = SEED_ENTITIES,
    base_registry: EntityRegistry | None = None,
    persist_path: Path | None = None,
) -> EntityRegistry:
    """Build a history-free, deterministic `EntityRegistry` combining `seed_entries`, any
    explicitly passed `base_registry`, and new structured external metadata items (`anilist`,
    `bangumi`, `syosetu`) present in `raw_items`.

    Enforces `docs/INGEST_CONTRACT.md` §3 & §5(1):
      - Existing `media_id`s are never renumbered or reused for a different entity.
      - New `(namespace, external_id)` pairs receive a history-free hash-derived `media_id`
        so ID assignment is invariant to crawl subset, ordering, or prior builds.
      - When `persist_path` is provided, cross-checks the built registry against the on-disk
        accumulated registry to assert zero ID or external_id conflicts, then updates the
        on-disk registry with any newly seen entries without polluting this snapshot's
        frozen `entities.json` with entities from other snapshots.
    """
    entries_by_id: dict[int, EntityEntry] = {}
    ext_to_id: dict[tuple[str, str], int] = {}
    seed_alias_to_id: dict[str, int] = {}

    def _register_into(
        target_by_id: dict[int, EntityEntry],
        target_ext_to_id: dict[tuple[str, str], int],
        entry: EntityEntry,
    ) -> None:
        for ns, eid in entry.external_ids:
            ext_key = (ns.lower().strip(), eid.strip())
            existing_id = target_ext_to_id.get(ext_key)
            if existing_id is not None and existing_id != entry.media_id:
                raise ValueError(
                    f"External ID {ext_key} cannot be reassigned from media_id={existing_id} "
                    f"to media_id={entry.media_id}"
                )
        prior = target_by_id.get(entry.media_id)
        if prior is not None:
            if prior.canonical_title != entry.canonical_title:
                raise ValueError(
                    f"Conflicting entity definition for media_id={entry.media_id}: "
                    f"{prior.canonical_title!r} ({prior.external_ids}) vs "
                    f"{entry.canonical_title!r} ({entry.external_ids})"
                )
            merged_ext = tuple(sorted(set(prior.external_ids) | set(entry.external_ids)))
            merged_aliases = tuple(sorted(set(prior.aliases) | set(entry.aliases)))
            target_by_id[entry.media_id] = EntityEntry(
                media_id=prior.media_id,
                canonical_title=prior.canonical_title,
                category=prior.category,
                aliases=merged_aliases,
                external_ids=merged_ext,
            )
            for ns, eid in merged_ext:
                target_ext_to_id[(ns.lower().strip(), eid.strip())] = entry.media_id
            return
        target_by_id[entry.media_id] = entry
        for ns, eid in entry.external_ids:
            target_ext_to_id[(ns.lower().strip(), eid.strip())] = entry.media_id

    for entry in seed_entries:
        _register_into(entries_by_id, ext_to_id, entry)
        for alias in entry.aliases:
            seed_alias_to_id.setdefault(_norm_alias(alias), entry.media_id)

    if base_registry is not None:
        for entry in base_registry.entries:
            _register_into(entries_by_id, ext_to_id, entry)
            for alias in entry.aliases:
                seed_alias_to_id.setdefault(_norm_alias(alias), entry.media_id)

    # Sort raw_items deterministically by (sorted external_media_refs, url, title)
    def _sort_raw_key(r: RawIngestItem) -> tuple[tuple[tuple[str, str], ...], str, str]:
        norm_refs = tuple(
            sorted(
                {
                    (ns.lower().strip(), eid.strip())
                    for ns, eid in r.external_media_refs
                    if ns and eid
                }
            )
        )
        return (norm_refs, r.url, r.title)

    for raw in sorted(raw_items, key=_sort_raw_key):
        if raw.external_media_refs:
            sorted_refs = tuple(
                sorted(
                    {
                        (ns.lower().strip(), eid.strip())
                        for ns, eid in raw.external_media_refs
                        if ns and eid
                    }
                )
            )
            if not sorted_refs:
                continue
            title_clean = normalize_text(raw.title).strip()
            norm_t = _norm_alias(title_clean)
            matched_existing_ids = {ext_to_id[rk] for rk in sorted_refs if rk in ext_to_id}
            if norm_t in seed_alias_to_id:
                matched_existing_ids.add(seed_alias_to_id[norm_t])
            if len(matched_existing_ids) > 1:
                raise ValueError(
                    f"Conflicting external_media_refs {sorted_refs} on {raw.url!r} resolve to "
                    f"multiple existing media_ids: {sorted(matched_existing_ids)}"
                )
            if len(matched_existing_ids) == 1:
                existing_mid = next(iter(matched_existing_ids))
                prior_entry = entries_by_id[existing_mid]
                merged_ext_ids = tuple(sorted(set(prior_entry.external_ids) | set(sorted_refs)))
                if merged_ext_ids != prior_entry.external_ids:
                    updated_entry = EntityEntry(
                        media_id=prior_entry.media_id,
                        canonical_title=prior_entry.canonical_title,
                        category=prior_entry.category,
                        aliases=prior_entry.aliases,
                        external_ids=merged_ext_ids,
                    )
                    entries_by_id[existing_mid] = updated_entry
                    for rk in sorted_refs:
                        ext_to_id[rk] = existing_mid
                continue
            if not title_clean:
                continue
            cat = classify_content_category(raw.title, raw.body, source_key=raw.source_key)
            primary_ref = sorted_refs[0]
            assigned_id = _deterministic_ext_media_id(primary_ref[0], primary_ref[1])
            new_entry = EntityEntry(
                media_id=assigned_id,
                canonical_title=title_clean,
                category=cat,
                aliases=(title_clean,),
                external_ids=sorted_refs,
            )
            _register_into(entries_by_id, ext_to_id, new_entry)

    registry = EntityRegistry(
        version=ENTITY_REGISTRY_VERSION,
        entries=tuple(entries_by_id.values()),
    )
    if persist_path is not None:
        combined_by_id: dict[int, EntityEntry] = {}
        combined_ext_to_id: dict[tuple[str, str], int] = {}
        if persist_path.exists():
            loaded_reg = EntityRegistry.load(persist_path)
            for entry in loaded_reg.entries:
                _register_into(combined_by_id, combined_ext_to_id, entry)
        for entry in registry.entries:
            _register_into(combined_by_id, combined_ext_to_id, entry)
        EntityRegistry(
            version=ENTITY_REGISTRY_VERSION,
            entries=tuple(combined_by_id.values()),
        ).save(persist_path)
    return registry


def build_default_source_registry() -> SourceRegistry:
    """Return the canonical W1 `SourceRegistry` (`sources.json`)."""
    sources = (
        SourceRegistryEntry(
            source_key="anilist-graphql",
            adapter_family="StructuredApiAdapter",
            default_publisher="AniList",
            default_source_type="metadata",
            lang="en",
            source_tz="UTC",
            endpoint="https://graphql.anilist.co",
        ),
        SourceRegistryEntry(
            source_key="ann-news",
            adapter_family="AnnArchiveAdapter",
            default_publisher="Anime News Network",
            default_source_type="media",
            lang="en",
            source_tz="UTC",
            endpoint="https://www.animenewsnetwork.com/news/",
        ),
        SourceRegistryEntry(
            source_key="bangumi-api",
            adapter_family="StructuredApiAdapter",
            default_publisher="Bangumi",
            default_source_type="metadata",
            lang="zh",
            source_tz="Asia/Shanghai",
            endpoint="https://api.bgm.tv/v0/subjects",
        ),
        SourceRegistryEntry(
            source_key="bilibili-timeline",
            adapter_family="StructuredApiAdapter",
            default_publisher="Bilibili",
            default_source_type="metadata",
            lang="zh",
            source_tz="Asia/Shanghai",
            endpoint="https://api.bilibili.com/pgc/web/timeline",
        ),
        SourceRegistryEntry(
            source_key="bones-news",
            adapter_family="WpPagedAdapter",
            default_publisher="BONES",
            default_source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            endpoint="https://www.bones.co.jp",
        ),
        SourceRegistryEntry(
            source_key="capcom-news",
            adapter_family="RssAdapter",
            default_publisher="Capcom",
            default_source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            endpoint="https://www.capcom.co.jp/rss/news.xml",
        ),
        SourceRegistryEntry(
            source_key="chuapp-news",
            adapter_family="RssAdapter",
            default_publisher="Chuapp",
            default_source_type="media",
            lang="zh",
            source_tz="Asia/Shanghai",
            endpoint="https://www.chuapp.com/feed",
        ),
        SourceRegistryEntry(
            source_key="yystv-news",
            adapter_family="RssAdapter",
            default_publisher="Yystv",
            default_source_type="media",
            lang="zh",
            source_tz="Asia/Shanghai",
            endpoint="https://www.yystv.cn/rss/feed",
        ),
        SourceRegistryEntry(
            source_key="gcores-news",
            adapter_family="RssAdapter",
            default_publisher="Gcores",
            default_source_type="media",
            lang="zh",
            source_tz="Asia/Shanghai",
            endpoint="https://www.gcores.com/rss",
        ),
        SourceRegistryEntry(
            source_key="mappa-news",
            adapter_family="WpPagedAdapter",
            default_publisher="MAPPA",
            default_source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            endpoint="https://www.mappa.co.jp",
        ),
        SourceRegistryEntry(
            source_key="nijimen-news",
            adapter_family="WpPagedAdapter",
            default_publisher="にじめん",
            default_source_type="media",
            lang="ja",
            source_tz="Asia/Tokyo",
            endpoint="https://nijimen.kusuguru.co.jp/wp-json/wp/v2/posts",
            canonical_host="nijimen.kusuguru.co.jp",
        ),
        SourceRegistryEntry(
            source_key="paworks-news",
            adapter_family="WpPagedAdapter",
            default_publisher="P.A.WORKS",
            default_source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            endpoint="https://www.pa-works.jp",
        ),
        SourceRegistryEntry(
            source_key="prtimes-acgn",
            adapter_family="RssAdapter",
            default_publisher="PR TIMES",
            default_source_type="media",
            lang="ja",
            source_tz="Asia/Tokyo",
            endpoint="https://prtimes.jp/index.rdf",
        ),
        SourceRegistryEntry(
            source_key="realsound-acgn",
            adapter_family="WpPagedAdapter",
            default_publisher="Real Sound",
            default_source_type="media",
            lang="ja",
            source_tz="Asia/Tokyo",
            endpoint="https://realsound.jp/wp-json/wp/v2/posts",
            categories=(116583, 116584),
        ),
        SourceRegistryEntry(
            source_key="syosetu-novelapi",
            adapter_family="StructuredApiAdapter",
            default_publisher="Syosetu",
            default_source_type="metadata",
            lang="ja",
            source_tz="Asia/Tokyo",
            endpoint="https://api.syosetu.com/novelapi/api/",
        ),
        SourceRegistryEntry(
            source_key="toei-news",
            adapter_family="HtmlListAdapter",
            default_publisher="Toei Animation",
            default_source_type="official",
            lang="ja",
            source_tz="Asia/Tokyo",
            endpoint="https://www.toei-anim.co.jp/news/",
        ),
    )
    return SourceRegistry(version=SOURCE_REGISTRY_VERSION, sources=sources)


__all__ = [
    "DEFAULT_ENTITY_REGISTRY_PATH",
    "ENTITY_REGISTRY_VERSION",
    "SEED_ENTITIES",
    "SOURCE_REGISTRY_VERSION",
    "EntityEntry",
    "EntityLinker",
    "EntityRegistry",
    "LinkingStats",
    "SourceRegistry",
    "SourceRegistryEntry",
    "build_default_source_registry",
    "build_entity_registry_from_raw",
]
