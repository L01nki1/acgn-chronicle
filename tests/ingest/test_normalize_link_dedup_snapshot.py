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


def test_corporate_ir_never_merges_via_auxiliary_text_similarity_with_positive_control() -> None:
    """E P0-2, D P1-3, Claude A5: corporate/IR items must never merge via either the primary
    entity-event channel OR the auxiliary same-language text similarity channel, even when
    char-3-gram cosine similarity exceeds 0.86 (or similarity_fn returns 1.0); a non-corporate
    control pair with identical text DOES merge.
    """
    from dataclasses import replace

    from acgn_rag.core.ingest.dedup import char_ngram_cosine

    corp_a = normalize_item(
        RawIngestItem(
            url="https://www.toei-anim.co.jp/ir/20261028_q2.html",
            title="2027年3月期 第2四半期決算短信[日本基準](連結)の開示に関するお知らせ",
            body="当社は本日開催の取締役会において、2027年3月期第2四半期決算短信(連結)を開示いたしました。",
            published_at="2026-10-28T06:00:00Z",
            publisher="Toei Animation",
            source_key="toei-news",
            source_type="official",
            lang="ja",
            ingest_category="corporate",
            event_type="corporate",
            media_ids=(10,),
        )
    )
    corp_b = normalize_item(
        RawIngestItem(
            url="https://www.toei-anim.co.jp/ir/20261029_q2_rev.html",
            title="2027年3月期 第2四半期決算短信[日本基準](連結)の開示に関するお知らせ(追報)",
            body="当社は本日開催の取締役会において、2027年3月期第2四半期決算短信(連結)を開示いたしました。",
            published_at="2026-10-29T06:00:00Z",
            publisher="Toei Animation",
            source_key="toei-news",
            source_type="official",
            lang="ja",
            ingest_category="corporate",
            event_type="corporate",
            media_ids=(10,),
        )
    )
    # Prove the test pair genuinely exceeds the 0.86 threshold on the default similarity function
    assert (
        char_ngram_cosine(
            f"{corp_a.title}\n{corp_a.body[:240]}",
            f"{corp_b.title}\n{corp_b.body[:240]}",
        )
        >= 0.90
    )

    res_corp = cluster_items([corp_a, corp_b])
    assert len(res_corp.clusters) == 2

    res_corp_forced = cluster_items([corp_a, corp_b], similarity_fn=lambda _a, _b: 1.0)
    assert len(res_corp_forced.clusters) == 2

    # Positive control: identical non-corporate pair merges into 1 cluster
    ctrl_a = replace(corp_a, ingest_category=None, event_type="release")
    ctrl_b = replace(corp_b, ingest_category=None, event_type="release")
    res_ctrl = cluster_items([ctrl_a, ctrl_b])
    assert len(res_ctrl.clusters) == 1
    assert len(res_ctrl.clusters[0].member_doc_ids) == 2


def test_entity_registry_multi_ref_order_invariance_and_conflict_detection() -> None:
    """E P2-4 & Claude A8: build_entity_registry_from_raw registers all external_media_refs
    deterministically (independent of tuple ordering) and raises ValueError if two refs on
    one item resolve to conflicting existing entities.
    """
    from acgn_rag.core.ingest.facts_link import build_entity_registry_from_raw

    item_ab = RawIngestItem(
        url="https://anilist.co/anime/154587",
        title="Frieren: Beyond Journey's End",
        body="Aliases: 葬送のフリーレン",
        published_at="2023-09-29T00:00:00Z",
        publisher="AniList",
        source_key="anilist-graphql",
        source_type="metadata",
        lang="en",
        external_media_refs=(("anilist", "154587"), ("bangumi", "400602")),
    )
    item_ba = RawIngestItem(
        url="https://anilist.co/anime/154587",
        title="Frieren: Beyond Journey's End",
        body="Aliases: 葬送のフリーレン",
        published_at="2023-09-29T00:00:00Z",
        publisher="AniList",
        source_key="anilist-graphql",
        source_type="metadata",
        lang="en",
        external_media_refs=(("bangumi", "400602"), ("anilist", "154587")),
    )
    reg_ab = build_entity_registry_from_raw([item_ab], seed_entries=())
    reg_ba = build_entity_registry_from_raw([item_ba], seed_entries=())
    assert reg_ab.sha256 == reg_ba.sha256
    assert len(reg_ab.entries) == 1
    assert set(reg_ab.entries[0].external_ids) == {
        ("anilist", "154587"),
        ("bangumi", "400602"),
    }

    # Conflict: anilist:1 and bangumi:2 already belong to two distinct entities
    separate_1 = RawIngestItem(
        url="https://anilist.co/anime/1",
        title="Cowboy Bebop",
        body="Aliases: カウボーイビバップ",
        published_at="1998-04-03T00:00:00Z",
        publisher="AniList",
        source_key="anilist-graphql",
        source_type="metadata",
        lang="en",
        external_media_refs=(("anilist", "1"),),
    )
    separate_2 = RawIngestItem(
        url="https://bgm.tv/subject/2",
        title="全く別の作品",
        body="Summary: unrelated",
        published_at="2020-01-01T00:00:00Z",
        publisher="Bangumi",
        source_key="bangumi-api",
        source_type="metadata",
        lang="zh",
        external_media_refs=(("bangumi", "2"),),
    )
    base_reg = build_entity_registry_from_raw([separate_1, separate_2])
    conflicting_item = RawIngestItem(
        url="https://anilist.co/anime/999999",
        title="Conflicting Cross-Ref Item",
        body="Body",
        published_at="2025-01-01T00:00:00Z",
        publisher="AniList",
        source_key="anilist-graphql",
        source_type="metadata",
        lang="en",
        external_media_refs=(("anilist", "1"), ("bangumi", "2")),
    )
    with pytest.raises(ValueError, match="Conflicting external_media_refs"):
        build_entity_registry_from_raw([conflicting_item], base_registry=base_reg)


def test_snapshot_reader_eager_completeness_and_tamper_detection(tmp_path: Path) -> None:
    """E P1-3, F #6659994806, D P2-5, Claude A3:
    iter_snapshot_items / load_snapshot_items / load_corpus_snapshot / load_snapshot_clusters /
    resolve_url_in_snapshot / publish_snapshot must eagerly reject truncated files, duplicated rows,
    extra rows, byte-divergent representatives, tampered entities/sources/snapshot/clusters, and
    missing sidecars BEFORE yielding row 0.
    """
    raw_items = [
        RawIngestItem(
            url="https://natalie.mu/comic/news/600001",
            title="『葬送のフリーレン』第2期が2026年1月放送決定",
            body="TVアニメ『葬送のフリーレン』第2期が2026年1月より放送されることが決定した。",
            published_at="2026-10-28T09:00:00Z",
            publisher="Comic Natalie",
            source_key="toei-news",
            source_type="official",
            lang="ja",
            event_type="broadcast",
        ),
        RawIngestItem(
            url="https://natalie.mu/comic/news/600002",
            title="『葬送のフリーレン』第2期が2026年1月放送決定",
            body="TVアニメ『葬送のフリーレン』第2期が2026年1月より放送されることが決定した。",
            published_at="2026-10-28T10:00:00Z",
            publisher="Comic Natalie",
            source_key="toei-news",
            source_type="official",
            lang="ja",
            event_type="broadcast",
        ),
        RawIngestItem(
            url="https://www.gcores.com/articles/199999",
            title="《怪物猎人荒野》最新更新配信",
            body="Capcom宣布《怪物猎人荒野》新活动任务上线。",
            published_at="2026-10-27T04:00:00Z",
            publisher="Gcores",
            source_key="gcores-news",
            source_type="media",
            lang="zh",
            event_type="update",
        ),
    ]
    built = build_snapshot_from_raw(
        raw_items,
        window_start=date(2023, 10, 28),
        window_end=date(2026, 10, 28),
    )
    snap_dir = publish_snapshot(built, snapshots_root=tmp_path)

    # Save pristine bytes of all files
    pristine = {
        fname: (snap_dir / fname).read_bytes()
        for fname in (
            "manifest.json",
            "snapshot.json",
            "items.jsonl",
            "representatives.jsonl",
            "clusters.jsonl",
            "entities.json",
            "sources.json",
            "quarantine.jsonl",
        )
    }

    def _restore() -> None:
        for fname, data in pristine.items():
            (snap_dir / fname).write_bytes(data)

    # (a) Truncated representatives.jsonl: must raise BEFORE yielding row 0 on BOTH paths!
    rep_lines = pristine["representatives.jsonl"].decode("utf-8").splitlines()
    assert len(rep_lines) == 2
    (snap_dir / "representatives.jsonl").write_text(rep_lines[0] + "\n", encoding="utf-8")
    for rep_mode in (True, False):
        yielded_before_err: list[str] = []
        with pytest.raises(SnapshotError, match="Incomplete, duplicate, or out-of-order"):
            for it in iter_snapshot_items(
                snap_dir, data_root=tmp_path, representatives_only=rep_mode
            ):
                yielded_before_err.append(it.doc_id)
        assert yielded_before_err == []
    _restore()

    # (a2) Removing a non-representative row from items.jsonl: must raise on BOTH paths! (E Blocker 1 / F)
    item_lines = pristine["items.jsonl"].decode("utf-8").splitlines()
    assert len(item_lines) == 3
    rep_doc_ids = {json.loads(line)["doc_id"] for line in rep_lines}
    only_rep_item_lines = [line for line in item_lines if json.loads(line)["doc_id"] in rep_doc_ids]
    assert len(only_rep_item_lines) == 2
    (snap_dir / "items.jsonl").write_text("\n".join(only_rep_item_lines) + "\n", encoding="utf-8")
    for rep_mode in (True, False):
        with pytest.raises(SnapshotError, match="Incomplete, duplicate, or out-of-order"):
            next(iter_snapshot_items(snap_dir, data_root=tmp_path, representatives_only=rep_mode))
    _restore()

    # (b) Duplicated row in items.jsonl or representatives.jsonl: must raise on BOTH paths!
    (snap_dir / "items.jsonl").write_text(
        "\n".join([item_lines[0], item_lines[0], *item_lines[1:]]) + "\n",
        encoding="utf-8",
    )
    for rep_mode in (True, False):
        with pytest.raises(SnapshotError, match="Incomplete, duplicate, or out-of-order"):
            next(iter_snapshot_items(snap_dir, data_root=tmp_path, representatives_only=rep_mode))
    _restore()

    (snap_dir / "representatives.jsonl").write_text(
        "\n".join([rep_lines[0], rep_lines[0], rep_lines[1]]) + "\n",
        encoding="utf-8",
    )
    for rep_mode in (True, False):
        with pytest.raises(SnapshotError, match="Incomplete, duplicate, or out-of-order"):
            next(iter_snapshot_items(snap_dir, data_root=tmp_path, representatives_only=rep_mode))
    _restore()

    # (c) Extra non-representative row added to representatives.jsonl: must raise on BOTH paths!
    (snap_dir / "representatives.jsonl").write_bytes(pristine["items.jsonl"])
    for rep_mode in (True, False):
        with pytest.raises(SnapshotError, match="Incomplete, duplicate, or out-of-order"):
            next(iter_snapshot_items(snap_dir, data_root=tmp_path, representatives_only=rep_mode))
    _restore()

    # (d) representatives.jsonl row formatting differs from items.jsonl byte-for-byte
    rep_obj = json.loads(rep_lines[0])
    non_canonical_line = json.dumps(
        rep_obj, ensure_ascii=False, indent=None, separators=(", ", ": ")
    )
    (snap_dir / "representatives.jsonl").write_text(
        "\n".join([non_canonical_line, rep_lines[1]]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SnapshotError, match=r"not byte-identical to items\.jsonl"):
        next(iter_snapshot_items(snap_dir, data_root=tmp_path, representatives_only=True))
    _restore()

    # (e) Tampered entities.json and sources.json must fail verify_snapshot_integrity (A3)
    (snap_dir / "entities.json").write_bytes(pristine["entities.json"] + b" ")
    with pytest.raises(SnapshotError, match=r"Tampered entities\.json"):
        load_snapshot_items(snap_dir, data_root=tmp_path)
    _restore()

    (snap_dir / "sources.json").write_bytes(pristine["sources.json"] + b" ")
    with pytest.raises(SnapshotError, match=r"Tampered sources\.json"):
        load_snapshot_items(snap_dir, data_root=tmp_path)
    _restore()

    # (f) Tampered snapshot.json (item_count or url_to_representative) must fail (E P1-3)
    meta_obj = json.loads(pristine["snapshot.json"].decode("utf-8"))
    meta_obj["item_count"] = 999999
    (snap_dir / "snapshot.json").write_text(json.dumps(meta_obj), encoding="utf-8")
    with pytest.raises(SnapshotError, match=r"Tampered snapshot\.json"):
        load_corpus_snapshot(snap_dir, data_root=tmp_path)
    _restore()

    meta_obj2 = json.loads(pristine["snapshot.json"].decode("utf-8"))
    meta_obj2["url_to_representative"] = {"https://natalie.mu/comic/news/600001": "doc_forged"}
    (snap_dir / "snapshot.json").write_text(json.dumps(meta_obj2), encoding="utf-8")
    with pytest.raises(SnapshotError, match="Tampered url_to_representative"):
        resolve_url_in_snapshot(
            snap_dir, "https://natalie.mu/comic/news/600001", data_root=tmp_path
        )
    _restore()

    # (g) Tampered clusters.jsonl (missing cluster) must fail load_snapshot_clusters
    cluster_lines = pristine["clusters.jsonl"].decode("utf-8").splitlines()
    assert len(cluster_lines) == 2
    (snap_dir / "clusters.jsonl").write_text(cluster_lines[0] + "\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="Incomplete cluster partition"):
        load_snapshot_clusters(snap_dir, data_root=tmp_path)
    _restore()

    # (h) Missing sidecar file (representatives.jsonl deleted) must raise SnapshotError in
    # both iter_snapshot_items and idempotent publish_snapshot (F #6659994806, D P2-5)
    (snap_dir / "representatives.jsonl").unlink()
    with pytest.raises(SnapshotError, match="missing required files"):
        next(iter_snapshot_items(snap_dir, data_root=tmp_path, representatives_only=True))
    with pytest.raises(SnapshotError, match="is incomplete"):
        publish_snapshot(built, snapshots_root=tmp_path)
    _restore()


def test_catalog_chronological_sort_tie_rejection_current_retention_and_partial_isolation(
    tmp_path: Path,
) -> None:
    """E P1-4, D P1-4, Claude A4:
    _update_current_and_catalog sorts publications by published_at descending, rejects exact
    timestamp ties between distinct snapshot_ids, keeps active CURRENT in retained even when older
    than top-K, and isolates partial=True snapshots in partial_publications.
    """
    from acgn_rag.core.ingest.snapshot import _update_current_and_catalog

    cat_root = tmp_path / "catalog_checks"
    cat_root.mkdir()

    # S1 is set as active CURRENT at 01:00
    _update_current_and_catalog(
        cat_root,
        "20261028-aaaaaaaaaaaa",
        published_at="2026-10-28T01:00:00Z",
        keep_snapshots=2,
        set_current=True,
    )
    # S3 is published at 03:00 (set_current=False)
    _update_current_and_catalog(
        cat_root,
        "20261105-cccccccccccc",
        published_at="2026-10-28T03:00:00Z",
        keep_snapshots=2,
        set_current=False,
    )
    # S2 is published out-of-order with an earlier timestamp 02:00 (set_current=False)
    _update_current_and_catalog(
        cat_root,
        "20261101-bbbbbbbbbbbb",
        published_at="2026-10-28T02:00:00Z",
        keep_snapshots=2,
        set_current=False,
    )

    cat = json.loads((cat_root / "catalog.json").read_text(encoding="utf-8"))
    # Sorted by published_at descending: S3 (03:00), S2 (02:00), S1 (01:00)
    assert [p["snapshot_id"] for p in cat["publications"]] == [
        "20261105-cccccccccccc",
        "20261101-bbbbbbbbbbbb",
        "20261028-aaaaaaaaaaaa",
    ]
    assert cat["latest_corpus"] == "20261105-cccccccccccc"
    # S1 is still CURRENT, so it MUST remain in retained and never enter retired!
    assert cat["current"] == "20261028-aaaaaaaaaaaa"
    assert "20261028-aaaaaaaaaaaa" in cat["retained"]
    assert "20261028-aaaaaaaaaaaa" not in cat["retired"]

    # Exact timestamp tie between distinct snapshot_ids must raise SnapshotError
    with pytest.raises(SnapshotError, match="Ambiguous tied published_at"):
        _update_current_and_catalog(
            cat_root,
            "20261106-dddddddddddd",
            published_at="2026-10-28T03:00:00Z",
            keep_snapshots=2,
            set_current=False,
        )

    # Partial snapshot is isolated in partial_publications and never enters publications/retained/retired
    _update_current_and_catalog(
        cat_root,
        "20261106-partial00000",
        published_at="2026-10-28T05:00:00Z",
        keep_snapshots=2,
        set_current=False,
        partial=True,
    )
    cat_after_partial = json.loads((cat_root / "catalog.json").read_text(encoding="utf-8"))
    assert cat_after_partial["latest_corpus"] == "20261105-cccccccccccc"
    assert [p["snapshot_id"] for p in cat_after_partial["partial_publications"]] == [
        "20261106-partial00000"
    ]
    assert "20261106-partial00000" not in cat_after_partial["retained"]
    assert "20261106-partial00000" not in cat_after_partial["retired"]


def test_discover_daily_next_day_20261106_rolling_14d_window_future_clamp_and_drift(
    tmp_path: Path,
) -> None:
    """E P0-1, D P0-2, E P2-1, Claude A1:
    discover_daily against a base snapshot ending 2026-11-05 accepts a newly discovered item
    published on 2026-11-06, mints a 14-day rolling daily snapshot (2026-10-24..2026-11-06,
    snapshot_id prefix 20261106-...) without mixing in 2024 historical items from the base
    snapshot, clamps future 2027 metadata items into out_of_window_count, and reports
    metadata_drift_count on modified known doc_ids.
    """
    import importlib.util
    import sys

    script_path = REPO_ROOT / "scripts" / "discover_daily.py"
    spec = importlib.util.spec_from_file_location("discover_daily_test", script_path)
    assert spec is not None and spec.loader is not None
    discover_mod = importlib.util.module_from_spec(spec)
    sys.modules["discover_daily_test"] = discover_mod
    spec.loader.exec_module(discover_mod)

    base_raw = [
        RawIngestItem(
            url="https://www.animenewsnetwork.com/news/2024-05-01/old-2024-item/.100001",
            title="Historical 2024 Anime News",
            body="Historical 2024 body",
            published_at="2024-05-01T10:00:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
        ),
        RawIngestItem(
            url="https://www.animenewsnetwork.com/news/2026-11-05/recent-nov5-item/.200001",
            title="Recent Nov 5 Anime News",
            body="Recent Nov 5 Archive Headline",
            published_at="2026-11-05T10:00:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
        ),
    ]
    base_built = build_snapshot_from_raw(
        base_raw,
        window_start="2023-11-05",
        window_end="2026-11-05",
    )
    publish_snapshot(base_built, snapshots_root=tmp_path / "snapshots", set_current=True)

    class _Nov6Adapter:
        source_key = "ann-news"
        publisher = "Anime News Network"

        def discover(self, since=None, *, use_conditional=True):
            _ = (since, use_conditional)
            return [
                # 1. Existing 2026-11-05 URL with updated RSS blurb -> unchanged + metadata_drift_count=1
                RawIngestItem(
                    url="https://www.animenewsnetwork.com/news/2026-11-05/recent-nov5-item/.200001",
                    title="Recent Nov 5 Anime News",
                    body="Recent Nov 5 Archive Headline\n\nExpanded RSS summary paragraph.",
                    published_at="2026-11-05T12:30:00Z",
                    publisher="Anime News Network",
                    source_key="ann-news",
                    source_type="media",
                    lang="en",
                ),
                # 2. Brand new item published on 2026-11-06 -> inserted_count=1
                RawIngestItem(
                    url="https://www.animenewsnetwork.com/news/2026-11-06/brand-new-nov6/.200002",
                    title="Brand New Nov 6 Anime Announcement",
                    body="Brand new announcement body on Nov 6.",
                    published_at="2026-11-06T08:00:00Z",
                    publisher="Anime News Network",
                    source_key="ann-news",
                    source_type="media",
                    lang="en",
                ),
                # 3. Future 2027 air date item -> clamped out of daily window (out_of_window_count=1)
                RawIngestItem(
                    url="https://anilist.co/anime/999888",
                    title="Future 2027 Anime Entry",
                    body="Release Date: 2027-04-01",
                    published_at="2027-04-01T00:00:00Z",
                    publisher="AniList",
                    source_key="anilist-graphql",
                    source_type="metadata",
                    lang="en",
                ),
            ]

        def backfill(self, date_range=None):
            return []

    res, daily_built = discover_mod.run_incremental_discovery(
        reference_date="2026-11-06",
        registry={"ann-news": _Nov6Adapter()},
        base_snapshot_dir_or_id=base_built.snapshot_id,
        data_root=tmp_path,
        publish_if_changed=True,
    )
    assert res.inserted_count == 1
    assert res.unchanged_count == 1
    assert res.metadata_drift_count == 1
    assert len(res.metadata_drift_doc_ids) == 1
    assert res.out_of_window_count == 1
    assert res.window_start == "2026-10-24"
    assert res.window_end == "2026-11-06"
    assert res.new_snapshot_id is not None
    assert res.new_snapshot_id.startswith("20261106-")
    assert daily_built is not None
    # Daily snapshot contains ONLY the 14-day window items (Nov 5 + Nov 6 = 2 items), NOT the 2024 item!
    assert daily_built.corpus_snapshot.item_count == 2


def test_from_raw_missing_or_mismatched_sidecar_and_page_cap_partial(tmp_path: Path) -> None:
    """E P2-2, Claude A2 & A9:
    --from-raw without a .summary.json sidecar or with a mismatched raw_sha256 marks partial=True
    and exits non-zero unless --allow-partial is passed; crawl_backfill_window marks partial=True
    when an adapter hits its page cap (last_page_cap_hit=True).
    """
    import importlib.util
    import sys

    script_path = REPO_ROOT / "scripts" / "backfill.py"
    spec = importlib.util.spec_from_file_location("backfill_sidecar_test", script_path)
    assert spec is not None and spec.loader is not None
    backfill_mod = importlib.util.module_from_spec(spec)
    sys.modules["backfill_sidecar_test"] = backfill_mod
    spec.loader.exec_module(backfill_mod)

    raw_items = [
        RawIngestItem(
            url="https://www.animenewsnetwork.com/news/2026-10-28/item-1/.217200",
            title="Item 1",
            body="Body 1",
            published_at="2026-10-28T00:00:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
        )
    ]
    raw_file = tmp_path / "unverified.jsonl"
    backfill_mod.save_raw_jsonl(raw_items, raw_file)
    sidecar = raw_file.with_suffix(".jsonl.summary.json")
    assert sidecar.exists()

    # 1. Missing sidecar -> exit code 1 without --allow-partial
    sidecar.unlink()
    assert (
        backfill_mod.main(
            [
                "--since",
                "2023-10-28",
                "--until",
                "2026-10-28",
                "--from-raw",
                str(raw_file),
                "--data-root",
                str(tmp_path / "missing_sidecar_out"),
            ]
        )
        == 1
    )

    # 2. Stale/mismatched raw_sha256 in sidecar -> exit code 1 without --allow-partial
    sidecar.write_text(
        json.dumps(
            {
                "partial": False,
                "raw_bytes": 999999,
                "raw_sha256": "0" * 64,
                "sources": {},
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
                str(raw_file),
                "--data-root",
                str(tmp_path / "stale_sidecar_out"),
            ]
        )
        == 1
    )

    # 3. Adapter hitting max_pages cap sets partial=True and coverage_mode="page_cap_partial"
    class _CappedWpAdapter:
        source_key = "capped-wp"
        publisher = "Capped WP"
        last_page_cap_hit = True
        last_pages_fetched = 300
        last_earliest_observed_date = "2025-06-01"
        last_reached_window_start = False

        def discover(self, since=None, *, use_conditional=True):
            return list(raw_items)

        def backfill(self, date_range=None):
            return list(raw_items)

    _items, summary = backfill_mod.crawl_backfill_window(
        "2023-10-28",
        "2026-10-28",
        registry={"capped-wp": _CappedWpAdapter()},
    )
    assert summary["partial"] is True
    assert summary["coverage_complete"] is False
    assert summary["capped_sources"] == ["capped-wp"]
    assert summary["sources"]["capped-wp"]["status"] == "page_cap_partial"
    assert summary["sources"]["capped-wp"]["coverage_mode"] == "page_cap_partial"


def test_coverage_complete_false_when_coverage_gaps_present_and_strict_body_level(
    tmp_path: Path,
) -> None:
    """E Blocker 2 & D body_level constraint:
    - coverage_complete must be False whenever coverage_gaps is non-empty (even if partial=False).
    - _classify_item_body_level requires >= 300 chars AND >= 2 non-title paragraphs for 'full',
      and classifies single-paragraph 350-char blurbs and metadata synopses as 'blurb'.
    """
    from acgn_rag.core.ingest.snapshot import (
        _classify_item_body_level,
        build_snapshot_from_items,
    )

    single_para_blurb = normalize_item(
        RawIngestItem(
            url="https://www.animenewsnetwork.com/news/2026-10-20/blurb-card/.100010",
            title="Short Listing Card Title",
            body="Short Listing Card Title\n" + ("A single paragraph card blurb text. " * 12),
            published_at="2026-10-20T10:00:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
        )
    )
    assert len(single_para_blurb.body) >= 300
    assert _classify_item_body_level(single_para_blurb) == "blurb"

    multi_para_article = normalize_item(
        RawIngestItem(
            url="https://www.animenewsnetwork.com/news/2026-10-20/full-article/.100011",
            title="Full Article Title",
            body=(
                "Full Article Title\n"
                + ("First paragraph of full news article text with detailed reporting. " * 4)
                + "\n"
                + ("Second paragraph of full news article text with staff and cast details. " * 4)
            ),
            published_at="2026-10-20T11:00:00Z",
            publisher="Anime News Network",
            source_key="ann-news",
            source_type="media",
            lang="en",
        )
    )
    assert _classify_item_body_level(multi_para_article) == "full"

    metadata_multi_para = normalize_item(
        RawIngestItem(
            url="https://anilist.co/anime/99999",
            title="Metadata Entry",
            body=("Paragraph one of synopsis. " * 8) + "\n" + ("Paragraph two of synopsis. " * 8),
            published_at="2026-10-20T00:00:00Z",
            publisher="AniList",
            source_key="anilist-graphql",
            source_type="metadata",
            lang="en",
        )
    )
    assert _classify_item_body_level(metadata_multi_para) == "blurb"

    built = build_snapshot_from_items(
        [single_para_blurb, multi_para_article, metadata_multi_para],
        window_start=date(2023, 10, 28),
        window_end=date(2026, 10, 28),
        ingest_report={
            "partial": False,
            "coverage_gaps": ["anilist-graphql:catalog_top_n"],
        },
    )
    snap_dir = publish_snapshot(built, snapshots_root=tmp_path)
    meta = json.loads((snap_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert meta["partial"] is False
    assert meta["coverage_complete"] is False
    assert meta["body_level_counts"] == {"full": 1, "blurb": 2, "headline": 0}
    assert meta["full_body_item_count"] == 1
    assert meta["body_level_by_source"]["ann-news"] == {"full": 1, "blurb": 1, "headline": 0}
    assert meta["body_level_by_source"]["anilist-graphql"] == {"full": 0, "blurb": 1, "headline": 0}
