"""Offline deterministic unit tests for WP1 normalization, entity linking, deduplication, and snapshots."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from acgn_rag.core.ingest import (
    ConflictingObservationError,
    DedupPolicy,
    EntityEntry,
    EntityLinker,
    EntityRegistry,
    NormalizedItem,
    RawIngestItem,
    SnapshotCollisionError,
    SnapshotError,
    TimestampParseError,
    build_default_source_registry,
    build_entity_registry_from_raw,
    build_manifest_dict,
    build_snapshot_from_raw,
    clean_title,
    cluster_items,
    compute_cluster_id,
    compute_doc_id,
    compute_snapshot_id_from_manifest,
    iter_snapshot_items,
    load_corpus_snapshot,
    load_snapshot_clusters,
    load_snapshot_items,
    normalize_batch,
    normalize_item,
    normalize_text,
    normalize_url,
    parse_utc_timestamp,
    publish_snapshot,
    resolve_url_in_snapshot,
    sweep_thresholds,
)
from acgn_rag.core.store.payload import iso_to_epoch

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_normalize_url_and_doc_id_golden_fixture() -> None:
    """Canonical URL preserves query string byte-for-byte per INGEST_CONTRACT §1 L37-39."""
    golden_manifest = json.loads(
        (REPO_ROOT / "docs" / "examples" / "ingest-manifest-v1.json").read_text(encoding="utf-8")
    )
    assert (
        normalize_url("HTTPS://Natalie.MU:443/comic/news/598001#section")
        == "https://natalie.mu/comic/news/598001"
    )
    # Query strings must be preserved byte-for-byte without re-encoding or param stripping (Probe 2)
    assert normalize_url("https://a.com/p?q=a%20b&flag") == "https://a.com/p?q=a%20b&flag"
    assert (
        normalize_url("https://A.com:443/P/?utm_source=x&id=1#frag")
        == "https://a.com/P/?utm_source=x&id=1"
    )
    assert normalize_url("https://a.com/p?u=%7Euser") == "https://a.com/p?u=%7Euser"

    for item in golden_manifest["items"]:
        assert compute_doc_id(item["url"]) == item["doc_id"]
        cid, _ = compute_cluster_id([item["doc_id"]])
        assert cid == item["cluster_id"]


def test_clean_title_strips_tags_before_unescaping_entities() -> None:
    """Escaped angle-bracket titles like &lt;Oshi no Ko&gt; must not be stripped as HTML tags (Probe 3)."""
    assert clean_title("&lt;Oshi no Ko&gt; Season 3 Announced") == "<Oshi no Ko> Season 3 Announced"
    assert clean_title("Score &lt;3 &gt; 2 tie") == "Score <3 > 2 tie"
    assert clean_title("<h3><b>Real Tag</b> &amp; &lt;Title&gt;</h3>") == "Real Tag & <Title>"


def test_parse_utc_timestamp_fractions_and_tz() -> None:
    """Sub-second fractions are retained in ISO strings and truncated toward zero by iso_to_epoch (Probe 4)."""
    # INGEST_CONTRACT §1 L23 & INGEST_REVIEW #3: preserve sub-seconds, never round up across UTC day
    assert parse_utc_timestamp("2026-10-28T23:59:59.9Z") == "2026-10-28T23:59:59.9Z"
    assert iso_to_epoch("2026-10-28T23:59:59.9Z") == iso_to_epoch("2026-10-28T23:59:59Z")
    # Offset conversion + fraction preservation
    assert parse_utc_timestamp("2026-10-28T18:00:00.8+09:00") == "2026-10-28T09:00:00.8Z"
    assert parse_utc_timestamp("2026-11-05T06:00:00+0900") == "2026-11-04T21:00:00Z"
    # RFC-2822 -0000 is UTC and must not be shifted by source_tz (Probe 4c)
    assert (
        parse_utc_timestamp("Wed, 28 Oct 2026 10:00:00 -0000", source_tz="Asia/Tokyo")
        == "2026-10-28T10:00:00Z"
    )
    # Naive datetime uses explicit source_tz
    assert (
        parse_utc_timestamp("2026-10-28 18:00:00", source_tz="Asia/Tokyo") == "2026-10-28T09:00:00Z"
    )
    # Naive datetime without source_tz is rejected; RawIngestItem.source_tz defaults to None (Probe 4b, 4e)
    with pytest.raises(TimestampParseError, match="source_tz"):
        parse_utc_timestamp("2026-10-28 18:00:00")
    raw_no_tz = RawIngestItem(
        url="https://x.jp/a",
        title="t",
        body="b",
        published_at="2026-10-28T10:00:00",
        publisher="p",
        source_type="official",
        source_key="k",
    )
    assert raw_no_tz.source_tz is None
    with pytest.raises(TimestampParseError, match="source_tz"):
        normalize_item(raw_no_tz)


def test_normalize_text_nfc_and_lf() -> None:
    """Text normalization converts CRLF/CR to LF and applies Unicode NFC."""
    raw = "Line 1\r\nLine 2\rLine 3"
    assert normalize_text(raw) == "Line 1\nLine 2\nLine 3"


def test_normalize_batch_rejects_conflicting_same_url_payload_and_accepts_datetime_bounds() -> None:
    """Same canonical_url with different normalized content or publisher/source_key raises ConflictingObservationError."""
    obs1 = RawIngestItem(
        url="https://natalie.mu/comic/news/598001#lead",
        title="葬送のフリーレン 第2期 2026年1月放送",
        body="TVアニメ『葬送のフリーレン』第2期が2026年1月より放送される。",
        published_at="2026-10-28T18:00:00+09:00",
        publisher="Comic Natalie",
        source_key="natalie",
        source_type="media",
        lang="ja",
    )
    obs1_dup = RawIngestItem(
        url="https://natalie.mu/comic/news/598001",
        title="葬送のフリーレン 第2期 2026年1月放送",
        body="TVアニメ『葬送のフリーレン』第2期が2026年1月より放送される。\r\n",
        published_at="2026-10-28T09:00:00Z",
        publisher="Comic Natalie",
        source_key="natalie",
        source_type="media",
        lang="ja",
    )
    # Probe 8: datetime window bounds work identically to date/ISO strings
    report = normalize_batch(
        [obs1, obs1_dup],
        window_start=datetime(2023, 10, 28, 12, 0),
        window_end=datetime(2026, 10, 28, 23, 59),
    )
    assert len(report.items) == 1
    assert report.exact_duplicates_collapsed == 1

    obs_conflict_body = RawIngestItem(
        url="https://natalie.mu/comic/news/598001",
        title="葬送のフリーレン 第2期 2026年1月放送",
        body="DIFFERENT BODY PAYLOAD",
        published_at="2026-10-28T09:00:00Z",
        publisher="Comic Natalie",
        source_key="natalie",
        source_type="media",
        lang="ja",
    )
    with pytest.raises(ConflictingObservationError):
        normalize_batch(
            [obs1, obs_conflict_body],
            window_start="2023-10-28",
            window_end="2026-10-28",
        )

    # Probe 7: differing publisher or source_key on same URL must raise ConflictingObservationError
    obs_conflict_pub = RawIngestItem(
        url="https://natalie.mu/comic/news/598001",
        title="葬送のフリーレン 第2期 2026年1月放送",
        body="TVアニメ『葬送のフリーレン』第2期が2026年1月より放送される。",
        published_at="2026-10-28T09:00:00Z",
        publisher="Different Publisher",
        source_key="other-feed",
        source_type="media",
        lang="ja",
    )
    with pytest.raises(ConflictingObservationError):
        normalize_batch([obs1, obs_conflict_pub])
    with pytest.raises(ConflictingObservationError):
        normalize_batch([obs_conflict_pub, obs1])


def test_entity_and_source_registry_linking_ambiguity_and_persistence(tmp_path: Path) -> None:
    """EntityLinker links longest spans without prefix co-linking, rejects ambiguous aliases,
    and persists append-only entity IDs across builds (P0-3, Probe 10).
    """
    registry = build_entity_registry_from_raw(())
    sources = build_default_source_registry()
    assert any(s.source_key == "ann-news" for s in sources.sources)
    assert any(s.source_key == "bangumi-api" for s in sources.sources)
    filtered_sources = sources.filter_used(["ann-news", "capcom-news"])
    assert [s.source_key for s in filtered_sources.sources] == ["ann-news", "capcom-news"]

    linker = EntityLinker(registry)
    # Unambiguous bracketed title in Japanese (Frieren is media_id=210 in SEED_ENTITIES)
    frieren_ids, amb_hits = linker.resolve_item(
        title="『葬送のフリーレン』第2期が2026年1月放送決定",
        body="アニメ続編の放送時期が発表された。",
    )
    assert frieren_ids == [210]
    assert amb_hits == 0

    # Probe 10b: "Monster Hunter Wilds" (101) must consume its span and NOT co-link prefix "Monster Hunter" (102)
    mhw_ids, _ = linker.resolve_item(
        title="Monster Hunter Wilds update 2 detailed",
        body="Capcom revealed new quest details.",
    )
    assert mhw_ids == [101]

    # External ID resolution takes highest priority
    ext_ids, _ = linker.resolve_item(
        title="新作情報",
        body="詳細はこちら",
        external_refs=[("anilist", "154587")],
    )
    assert ext_ids == [210]

    # Ambiguous alias ("命运石之门" registered under two entries) must return []
    ambiguous_registry = EntityRegistry(
        entries=(
            EntityEntry(
                media_id=1,
                canonical_title="STEINS;GATE",
                aliases=("命运石之门",),
            ),
            EntityEntry(
                media_id=2,
                canonical_title="STEINS;GATE 0",
                aliases=("命运石之门",),
            ),
        )
    )
    amb_linker = EntityLinker(ambiguous_registry)
    resolved_amb, amb_count = amb_linker.resolve_item(
        title="《命运石之门》新作消息",
        body="《命运石之门》纪念活动公开。",
    )
    assert resolved_amb == []
    assert amb_count >= 1

    # P0-3: Entity IDs must never renumber or be reused across builds with different raw item subsets
    reg_file = tmp_path / "registry" / "entities.json"
    raw_novel_a = RawIngestItem(
        url="https://ncode.syosetu.com/n6755gk/",
        title="Novel Alpha",
        body="Synopsis A",
        published_at="2025-01-10T00:00:00Z",
        publisher="Syosetu",
        source_type="metadata",
        source_key="syosetu-novelapi",
        lang="ja",
        external_media_refs=(("syosetu", "n6755gk"),),
    )
    raw_novel_b = RawIngestItem(
        url="https://ncode.syosetu.com/n4218cw/",
        title="Novel Beta",
        body="Synopsis B",
        published_at="2025-02-10T00:00:00Z",
        publisher="Syosetu",
        source_type="metadata",
        source_key="syosetu-novelapi",
        lang="ja",
        external_media_refs=(("syosetu", "n4218cw"),),
    )
    reg_build_1 = build_entity_registry_from_raw([raw_novel_a], persist_path=reg_file)
    reg_build_2 = build_entity_registry_from_raw([raw_novel_b], persist_path=reg_file)
    reg_build_1_re = build_entity_registry_from_raw([raw_novel_a], persist_path=reg_file)
    by_ext_1 = {e.external_ids[0]: e.media_id for e in reg_build_1.entries if e.external_ids}
    by_ext_2 = {e.external_ids[0]: e.media_id for e in reg_build_2.entries if e.external_ids}
    # B1: snapshot registry only freezes seed + entities derived from that snapshot's raw input
    assert ("syosetu", "n6755gk") not in by_ext_2
    assert reg_build_1_re.sha256 == reg_build_1.sha256
    # P0-3: persisted registry union retains both with disjoint, deterministic IDs
    persisted = EntityRegistry.load(reg_file)
    by_ext_persisted = {e.external_ids[0]: e.media_id for e in persisted.entries if e.external_ids}
    assert by_ext_persisted[("syosetu", "n6755gk")] == by_ext_1[("syosetu", "n6755gk")]
    assert by_ext_persisted[("syosetu", "n4218cw")] == by_ext_2[("syosetu", "n4218cw")]
    assert by_ext_2[("syosetu", "n4218cw")] != by_ext_1[("syosetu", "n6755gk")]


def test_golden_manifest_and_identity_proof_exact_bytes() -> None:
    """Reproduce docs/examples/ingest-manifest-v1.json byte-for-byte (1,348 bytes, sha256, snapshot_id)."""
    proof = json.loads(
        (REPO_ROOT / "docs" / "examples" / "ingest-identity-proof.json").read_text(encoding="utf-8")
    )
    golden_manifest_bytes = (
        REPO_ROOT / "docs" / "examples" / "ingest-manifest-v1.json"
    ).read_bytes()
    golden_manifest = json.loads(golden_manifest_bytes.decode("utf-8"))

    synthetic_bodies = {
        "https://example.org/synthetic/alpha": "Synthetic body alpha",
        "https://example.org/synthetic/beta": "Synthetic body beta",
    }
    items = [
        NormalizedItem.from_dict({**row, "body": synthetic_bodies[row["url"]]})
        for row in golden_manifest["items"]
    ]
    policy = DedupPolicy.from_dict(golden_manifest["dedup_policy"])
    dedup_res = cluster_items(items, policy=policy)

    manifest_dict = build_manifest_dict(
        dedup_res.items,
        window_start=golden_manifest["window_start"],
        window_end=golden_manifest["window_end"],
        entity_registry_sha256=golden_manifest["entity_registry_sha256"],
        source_registry_sha256=golden_manifest["source_registry_sha256"],
        dedup_policy=policy,
    )
    snap_id, manifest_sha, rendered_bytes = compute_snapshot_id_from_manifest(manifest_dict)
    assert len(rendered_bytes) == proof["build_1"]["manifest_bytes"] == 1348
    assert rendered_bytes == golden_manifest_bytes
    assert manifest_sha == proof["build_1"]["sha256"]
    assert snap_id == proof["build_1"]["snapshot_id"] == "20261028-7576782837e7"

    # Reversed input order must yield byte-identical manifest and snapshot_id
    dedup_rev = cluster_items(list(reversed(items)), policy=policy)
    manifest_rev = build_manifest_dict(
        dedup_rev.items,
        window_start=golden_manifest["window_start"],
        window_end=golden_manifest["window_end"],
        entity_registry_sha256=golden_manifest["entity_registry_sha256"],
        source_registry_sha256=golden_manifest["source_registry_sha256"],
        dedup_policy=policy,
    )
    snap_id_2, manifest_sha_2, rendered_bytes_2 = compute_snapshot_id_from_manifest(manifest_rev)
    assert rendered_bytes_2 == rendered_bytes
    assert manifest_sha_2 == manifest_sha
    assert snap_id_2 == snap_id


def test_dedup_agglomerative_complete_linkage_and_policy_identity() -> None:
    """Unresolved media_ids=[] never merge on primary channel; agglomerative complete-linkage
    merges highest-similarity pairs first (Probe 14) and records custom similarity_fn in policy (Probe 9).
    """
    u1 = normalize_item(
        RawIngestItem(
            url="https://example.com/news/u1",
            title="Unknown Anime A Announces Cast",
            body="Cast details for completely unknown independent project Alpha.",
            published_at="2026-10-28T10:00:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
            event_type="CastStaff",
            media_ids=(),
        )
    )
    u2 = normalize_item(
        RawIngestItem(
            url="https://example.com/news/u2",
            title="Unknown Anime B Announces Cast",
            body="Totally unrelated cast announcement for project Beta in winter.",
            published_at="2026-10-28T11:00:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
            event_type="CastStaff",
            media_ids=(),
        )
    )
    res_unresolved = cluster_items([u1, u2])
    assert len(res_unresolved.clusters) == 2

    # Probe 9: custom similarity_fn must be reflected in policy.to_dict()
    res_always = cluster_items([u1, u2], similarity_fn=lambda _x, _y: 1.0)
    assert len(res_always.clusters) == 1
    assert res_unresolved.policy.to_dict() != res_always.policy.to_dict()

    # Probe 14: True agglomerative complete-linkage merges highest-similarity pair (0.99) first
    doc_a = normalize_item(
        RawIngestItem(
            url="https://example.com/news/chain-a",
            title="Headline Alpha",
            body="Body Alpha",
            published_at="2026-10-28T10:00:00Z",
            publisher="ANN",
            source_key="ann-news",
            source_type="media",
            lang="en",
            event_type=None,
            media_ids=(),
        )
    )
    doc_b = normalize_item(
        RawIngestItem(
            url="https://example.com/news/chain-b",
            title="Headline Beta",
            body="Body Beta",
            published_at="2026-10-28T11:00:00Z",
            publisher="Crunchyroll",
            source_key="crunchyroll-news",
            source_type="media",
            lang="en",
            event_type=None,
            media_ids=(),
        )
    )
    doc_c = normalize_item(
        RawIngestItem(
            url="https://example.com/news/chain-c",
            title="Headline Gamma",
            body="Body Gamma",
            published_at="2026-10-28T12:00:00Z",
            publisher="Natalie",
            source_key="natalie-comic",
            source_type="media",
            lang="en",
            event_type=None,
            media_ids=(),
        )
    )
    ordered = sorted([doc_a, doc_b, doc_c], key=lambda d: d.doc_id)
    d0, d1, d2 = ordered[0], ordered[1], ordered[2]
    pair_scores = {
        frozenset({d0.doc_id, d1.doc_id}): 0.90,
        frozenset({d1.doc_id, d2.doc_id}): 0.99,
        frozenset({d0.doc_id, d2.doc_id}): 0.10,
    }
    chain_res = cluster_items(
        ordered,
        similarity_fn=lambda x, y: pair_scores.get(frozenset({x.doc_id, y.doc_id}), 0.0),
    )
    assert all(len(c.member_doc_ids) < 3 for c in chain_res.clusters)
    cluster_member_sets = [set(c.member_doc_ids) for c in chain_res.clusters]
    assert {d1.doc_id, d2.doc_id} in cluster_member_sets
    assert {d0.doc_id} in cluster_member_sets

    # Weekly ranking titles with different date ranges must not merge on default similarity (P1-7d)
    rank_w1 = normalize_item(
        RawIngestItem(
            url="https://example.com/news/rank-1",
            title="Japanese Animation TV Ranking, December 8-14",
            body="Japanese Animation TV Ranking, December 8-14",
            published_at="2026-10-28T10:00:00Z",
            publisher="ANN",
            source_key="ann-news",
            source_type="media",
            lang="en",
        )
    )
    rank_w2 = normalize_item(
        RawIngestItem(
            url="https://example.com/news/rank-2",
            title="Japanese Animation TV Ranking, December 15-21",
            body="Japanese Animation TV Ranking, December 15-21",
            published_at="2026-10-28T11:00:00Z",
            publisher="ANN",
            source_key="ann-news",
            source_type="media",
            lang="en",
        )
    )
    assert len(cluster_items([rank_w1, rank_w2]).clusters) == 2

    # Templated Capcom press release titles ("プレスリリース 2027年3月期") must not false-merge (P2-7)
    pr_1 = normalize_item(
        RawIngestItem(
            url="https://www.capcom.co.jp/ir/news/html/261022.html",
            title="プレスリリース 2027年3月期",
            body="プレスリリース 2027年3月期",
            published_at="2026-10-22T06:00:00Z",
            publisher="Capcom",
            source_key="capcom-news",
            source_type="official",
            lang="ja",
        )
    )
    pr_2 = normalize_item(
        RawIngestItem(
            url="https://www.capcom.co.jp/ir/news/html/261023.html",
            title="プレスリリース 2027年3月期",
            body="プレスリリース 2027年3月期",
            published_at="2026-10-23T06:00:00Z",
            publisher="Capcom",
            source_key="capcom-news",
            source_type="official",
            lang="ja",
        )
    )
    # N2 + P2-7: "プレスリリース 2027年3月期" alone is not tagged corporate, and dedup similarity
    # still excludes templated fiscal-period titles from false-merging
    assert pr_1.ingest_category is None
    assert len(cluster_items([pr_1, pr_2]).clusters) == 2

    sweep = sweep_thresholds([doc_a, doc_b, doc_c], thresholds=(0.82, 0.86, 0.90))
    assert len(sweep) == 3


def test_snapshot_publish_idempotency_catalog_retirement_and_readers(tmp_path: Path) -> None:
    """Two consecutive rebuilds in reversed order produce identical manifest bytes and snapshot_id;
    catalog.json keeps retained and retired strictly disjoint (P1-10, P1-11).
    """
    raw_forward = [
        RawIngestItem(
            url="https://natalie.mu/comic/news/598001",
            title="『葬送のフリーレン』第2期が2026年1月放送決定",
            body="TVアニメ『葬送のフリーレン』第2期が2026年1月より放送されることが決定した。",
            published_at="2026-10-28T09:00:00Z",
            publisher="Comic Natalie",
            source_key="natalie-comic",
            source_type="media",
            lang="ja",
            event_type="ReleaseDate",
        ),
        RawIngestItem(
            url="https://www.animenewsnetwork.com/news/2026-10-28/frieren-season-2-january-2026/.217200",
            title="Frieren: Beyond Journey's End Season 2 Debuts in January 2026",
            body="The official website announced that Frieren: Beyond Journey's End Season 2 premieres in January 2026.",
            published_at="2026-10-28T10:30:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
            event_type="ReleaseDate",
            external_media_refs=(("anilist", "154587"),),
        ),
        RawIngestItem(
            url="https://www.gcores.com/articles/190001",
            title="《怪物猎人荒野》最新更新将于本周五配信",
            body="Capcom宣布《怪物猎人荒野》新活动任务上线日期定于下月。",
            published_at="2026-10-27T04:00:00Z",
            publisher="机核 GCORES",
            source_key="gcores-news",
            source_type="media",
            lang="zh",
            event_type="ReleaseDate",
        ),
    ]
    raw_reverse = list(reversed(raw_forward))

    b1 = build_snapshot_from_raw(
        raw_forward,
        window_start=date(2023, 10, 28),
        window_end=date(2026, 10, 28),
    )
    b2 = build_snapshot_from_raw(
        raw_reverse,
        window_start=date(2023, 10, 28),
        window_end=date(2026, 10, 28),
    )
    assert b1.snapshot_id == b2.snapshot_id
    assert b1.manifest_sha256 == b2.manifest_sha256
    assert b1.manifest_bytes == b2.manifest_bytes
    assert b1.corpus_snapshot.item_count == 3
    assert b1.corpus_snapshot.deduped_count == 2

    # Default publish_snapshot(set_current=False) writes catalog.json and quarantine.jsonl without flipping CURRENT (P1-11)
    snap_dir_1 = publish_snapshot(b1, snapshots_root=tmp_path)
    assert (snap_dir_1 / "quarantine.jsonl").exists()
    assert not (tmp_path / "CURRENT").exists()
    snap_dir_2 = publish_snapshot(b2, snapshots_root=tmp_path, set_current=True)
    assert snap_dir_1 == snap_dir_2
    assert (tmp_path / "CURRENT").read_text(encoding="utf-8").strip() == b1.snapshot_id

    # P1-10: Publish S1, S2, S3 with keep=2, then re-publish S1 -> S1 must NOT remain in retired
    from acgn_rag.core.ingest.snapshot import _update_current_and_catalog

    cat_root = tmp_path / "cat_test"
    cat_root.mkdir()
    _update_current_and_catalog(
        cat_root, "S1", published_at="2026-10-28T01:00:00Z", keep_snapshots=2
    )
    _update_current_and_catalog(
        cat_root, "S2", published_at="2026-10-28T02:00:00Z", keep_snapshots=2
    )
    _update_current_and_catalog(
        cat_root, "S3", published_at="2026-10-28T03:00:00Z", keep_snapshots=2
    )
    _update_current_and_catalog(
        cat_root, "S1", published_at="2026-10-28T04:00:00Z", keep_snapshots=2
    )
    cat_data = json.loads((cat_root / "catalog.json").read_text(encoding="utf-8"))
    assert cat_data["current"] == "S1"
    assert cat_data["retained"] == ["S1", "S3"]
    assert "S1" not in cat_data["retired"]
    assert set(cat_data["retained"]).isdisjoint(set(cat_data["retired"]))

    # Corrupt catalog.json must raise SnapshotError, never silently reset
    (cat_root / "catalog.json").write_text("{not-valid-json", encoding="utf-8")
    with pytest.raises(SnapshotError, match=r"Corrupt catalog\.json"):
        _update_current_and_catalog(
            cat_root, "S4", published_at="2026-10-28T05:00:00Z", keep_snapshots=2
        )

    # Verify snapshot reader functions for WP2 / WP4
    loaded_meta = load_corpus_snapshot(snap_dir_1)
    assert loaded_meta.snapshot_id == b1.snapshot_id
    assert loaded_meta.item_count == 3
    assert loaded_meta.deduped_count == 2

    loaded_clusters = load_snapshot_clusters(snap_dir_1)
    assert len(loaded_clusters) == 2

    loaded_items = load_snapshot_items(snap_dir_1)
    assert [it.doc_id for it in loaded_items] == sorted(it.doc_id for it in loaded_items)
    assert len(list(iter_snapshot_items(snap_dir_1))) == 3
    assert len(load_snapshot_items(snap_dir_1, representatives_only=True)) == 2

    # Verify raw URL -> canonical_url -> representative_doc_id resolution
    resolved_rep_id = resolve_url_in_snapshot(
        snap_dir_1,
        "HTTPS://Natalie.MU:443/comic/news/598001#top",
    )
    assert resolved_rep_id is not None
    assert resolved_rep_id.startswith("doc_")

    # P2-1: Editing body in items.jsonl must fail verified reader check against manifest.json
    items_path = snap_dir_1 / "items.jsonl"
    orig_items_text = items_path.read_text(encoding="utf-8")
    tampered_rows = []
    for line in orig_items_text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["body"] = row["body"] + " TAMPERED"
        tampered_rows.append(json.dumps(row, ensure_ascii=False))
    items_path.write_text("\n".join(tampered_rows) + "\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="Tampered"):
        load_snapshot_items(snap_dir_1)
    items_path.write_text(orig_items_text, encoding="utf-8")

    # B1: Build window X, build window Y (adding new entities to persist_path),
    # then rebuild window X -> must produce the exact same snapshot_id and manifest bytes!
    reg_persist = tmp_path / "registry" / "entities.json"
    raw_window_y = [
        *raw_forward,
        RawIngestItem(
            url="https://ncode.syosetu.com/n9999zz",
            title="New Light Novel Added In Window Y",
            body="Synopsis of new novel",
            published_at="2026-11-04T00:00:00Z",
            publisher="Syosetu",
            source_key="syosetu-novelapi",
            source_type="metadata",
            lang="ja",
            external_media_refs=(("syosetu", "n9999zz"),),
        ),
    ]
    snap_x_1 = build_snapshot_from_raw(
        raw_forward,
        window_start=date(2023, 10, 28),
        window_end=date(2026, 10, 28),
        persist_entity_registry_path=reg_persist,
    )
    snap_y = build_snapshot_from_raw(
        raw_window_y,
        window_start=date(2023, 11, 5),
        window_end=date(2026, 11, 5),
        persist_entity_registry_path=reg_persist,
    )
    snap_x_2 = build_snapshot_from_raw(
        raw_forward,
        window_start=date(2023, 10, 28),
        window_end=date(2026, 10, 28),
        persist_entity_registry_path=reg_persist,
    )
    snap_x_fresh = build_snapshot_from_raw(
        raw_forward,
        window_start=date(2023, 10, 28),
        window_end=date(2026, 10, 28),
        persist_entity_registry_path=None,
    )
    assert snap_x_1.snapshot_id == snap_x_2.snapshot_id == snap_x_fresh.snapshot_id
    assert snap_x_1.manifest_bytes == snap_x_2.manifest_bytes == snap_x_fresh.manifest_bytes
    assert snap_y.snapshot_id != snap_x_1.snapshot_id

    # Round-3 P2-4: Idempotent re-publish of an already-retained older snapshot preserves
    # catalog order, latest_corpus, and original published_at timestamp
    cat_root = tmp_path / "cat_order"
    publish_snapshot(snap_x_1, snapshots_root=cat_root, set_current=False, keep_snapshots=2)
    cat_after_x1 = json.loads((cat_root / "catalog.json").read_text(encoding="utf-8"))
    orig_x_pub_at = cat_after_x1["publications"][0]["published_at"]
    publish_snapshot(snap_y, snapshots_root=cat_root, set_current=False, keep_snapshots=2)
    publish_snapshot(snap_x_2, snapshots_root=cat_root, set_current=False, keep_snapshots=2)
    cat_after_x2 = json.loads((cat_root / "catalog.json").read_text(encoding="utf-8"))
    assert cat_after_x2["latest_corpus"] == snap_y.snapshot_id
    assert [p["snapshot_id"] for p in cat_after_x2["publications"]] == [
        snap_y.snapshot_id,
        snap_x_1.snapshot_id,
    ]
    assert cat_after_x2["publications"][1]["published_at"] == orig_x_pub_at

    # Verify short-ID collision detection raises SnapshotCollisionError
    manifest_file = snap_dir_1 / "manifest.json"
    manifest_file.write_bytes(b'{"tampered":true}\n')
    with pytest.raises(SnapshotCollisionError):
        publish_snapshot(b1, snapshots_root=tmp_path)


def test_backfill_and_discover_daily_scripts_offline(tmp_path: Path) -> None:
    """scripts/backfill.py enforces C1 window, refuses partial crawls, and supports --since/--until;
    discover_daily avoids ghost snapshots even on cross-mode RSS vs archive copies.
    """
    import importlib.util
    import sys

    def _load_script(name: str):
        script_path = REPO_ROOT / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, script_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    backfill_mod = _load_script("backfill")
    discover_mod = _load_script("discover_daily")

    # Missing --start/--since or --end/--until must return non-zero exit code 2 (C1)
    assert backfill_mod.main([]) == 2
    assert backfill_mod.main(["--start", "2023-10-28"]) == 2

    raw_items = [
        RawIngestItem(
            url="https://www.animenewsnetwork.com/news/2026-10-28/frieren-season-2/.217200",
            title="Frieren Season 2 Debuts in January 2026",
            body="Frieren Season 2 Debuts in January 2026 (ANN Archive 2026-10-28)",
            published_at="2026-10-28T00:00:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
            event_type="ReleaseDate",
        )
    ]
    raw_file = tmp_path / "raw_items.jsonl"
    backfill_mod.save_raw_jsonl(raw_items, raw_file)

    rc = backfill_mod.main(
        [
            "--since",
            "2023-10-28",
            "--until",
            "2026-10-28",
            "--from-raw",
            str(raw_file),
            "--data-root",
            str(tmp_path),
            "--verify-idempotency",
            "--set-current",
        ]
    )
    assert rc == 0
    current_snap_id = (tmp_path / "snapshots" / "CURRENT").read_text(encoding="utf-8").strip()
    assert current_snap_id.startswith("20261028-")

    # Incremental run returning an RSS version of the same ANN URL (with real time + blurb)
    # must recognize the existing doc_id as unchanged and mint no ghost snapshot (P1-13)
    rss_version_of_same_url = [
        RawIngestItem(
            url="https://www.animenewsnetwork.com/news/2026-10-28/frieren-season-2/.217200",
            title="Frieren Season 2 Debuts in January 2026",
            body="Frieren Season 2 Debuts in January 2026\n\nFull RSS summary blurb text.",
            published_at="2026-10-28T14:22:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
            event_type="ReleaseDate",
        )
    ]

    class _RssDiscoveryAdapter:
        def discover(self, since=None):
            return list(rss_version_of_same_url)

        def backfill(self, window):
            return list(raw_items)

    inc_res, inc_built = discover_mod.run_incremental_discovery(
        registry={"ann-news": _RssDiscoveryAdapter()},
        base_snapshot_dir_or_id=current_snap_id,
        data_root=tmp_path,
        publish_if_changed=True,
    )
    assert inc_res.inserted_count == 0
    assert inc_res.updated_count == 0
    assert inc_res.unchanged_count == 1
    assert inc_res.new_snapshot_id is None
    assert inc_built is None

    # Incremental run with a genuinely new article builds and publishes to snapshots_daily
    class _NewItemDiscoveryAdapter:
        def discover(self, since=None, *, use_conditional=True):
            _ = (since, use_conditional)
            return [
                *rss_version_of_same_url,
                RawIngestItem(
                    url="https://www.animenewsnetwork.com/news/2026-10-28/new-story/.217201",
                    title="Brand New Story Published Later on 2026-10-28",
                    body="Brand New Story Published Later on 2026-10-28",
                    published_at="2026-10-28T18:00:00Z",
                    publisher="Anime News Network",
                    source_key="ann-news",
                    source_type="media",
                    lang="en",
                    event_type="ReleaseDate",
                ),
            ]

        def backfill(self, window):
            return list(raw_items)

    inc_new_res, inc_new_built = discover_mod.run_incremental_discovery(
        registry={"ann-news": _NewItemDiscoveryAdapter()},
        base_snapshot_dir_or_id=current_snap_id,
        data_root=tmp_path,
        publish_if_changed=True,
    )
    assert inc_new_res.inserted_count == 1
    assert inc_new_res.unchanged_count == 1
    assert inc_new_res.new_snapshot_id is not None
    assert inc_new_built is not None
    assert (tmp_path / "snapshots_daily" / inc_new_res.new_snapshot_id / "manifest.json").exists()

    # P2-4: discover_daily refuses to publish when a source errors unless allow_partial=True
    class _ErroredDiscoveryAdapter:
        def discover(self, since=None):
            raise RuntimeError("simulated daily discovery source failure")

        def backfill(self, window):
            return []

    with pytest.raises(RuntimeError, match="Incremental discovery encountered source errors"):
        discover_mod.run_incremental_discovery(
            registry={"ann-news": _RssDiscoveryAdapter(), "broken-src": _ErroredDiscoveryAdapter()},
            base_snapshot_dir_or_id=current_snap_id,
            data_root=tmp_path,
            publish_if_changed=True,
        )

    # P2-6 & B2: --from-raw loads <raw_path>.summary.json and preserves partial flag
    partial_raw_file = tmp_path / "partial_raw.jsonl"
    backfill_mod.save_raw_jsonl(raw_items, partial_raw_file)
    partial_summary_file = partial_raw_file.with_suffix(partial_raw_file.suffix + ".summary.json")
    partial_summary_file.write_text(
        json.dumps(
            {
                "partial": True,
                "has_errors": False,
                "sampled_sources": ["ann-news(15/1096)"],
                "sources": {
                    "ann-news": {
                        "count": 1,
                        "mode": "backfill",
                        "status": "sampled_partial",
                        "sampled_days": 15,
                        "requested_days": 1096,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        backfill_mod.main(
            [
                "--since",
                "2023-10-28",
                "--until",
                "2026-10-28",
                "--from-raw",
                str(partial_raw_file),
                "--data-root",
                str(tmp_path / "partial_out"),
            ]
        )
        == 1
    )
    assert (
        backfill_mod.main(
            [
                "--since",
                "2023-10-28",
                "--until",
                "2026-10-28",
                "--from-raw",
                str(partial_raw_file),
                "--data-root",
                str(tmp_path / "partial_out"),
                "--allow-partial",
                "--set-current",
            ]
        )
        == 0
    )
    assert not (tmp_path / "partial_out" / "snapshots" / "CURRENT").exists()
    partial_snaps = list((tmp_path / "partial_out" / "snapshots").glob("20261028-*"))
    assert len(partial_snaps) == 1
    partial_meta = json.loads((partial_snaps[0] / "snapshot.json").read_text(encoding="utf-8"))
    assert partial_meta["partial"] is True
