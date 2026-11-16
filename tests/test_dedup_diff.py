"""dedup_diff: the strict reproduction comparison (E's #75 review, P2 + P7).

Two layers:

* `_compare` is tested directly — including E's P2 scenario (identical aggregate counts,
  disjoint cluster ids) that a counts-only comparison would have called "reproduces".
* An end-to-end run over a published synthetic snapshot pins the consistent path. A
  *tampered* end-to-end variant is impossible by design: `load_snapshot_clusters` and the
  integrity check cross-check `clusters.jsonl` against the content-addressed manifest, so a
  disagreeing cluster file is unloadable rather than comparable — the strict comparison is
  exactly what guards the already-loadable case.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dedup_diff  # noqa: E402
import make_smoke_snapshot  # noqa: E402

from acgn_rag.core.ingest.dedup import (  # noqa: E402
    DedupPolicy,
    cluster_items,
)

EVIDENCE = {
    "threshold_sweep_20261105": {
        "sweep_table": [{"threshold": 0.86, "cluster_count": 2, "total_items": 3}],
        "sample_merged_clusters_at_0_86": [],
    }
}


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("ACGN_DATA_DIR", str(root))
    return root


def _publish_pair(root: Path) -> str:
    assert make_smoke_snapshot.main(["--data-root", str(root), "--count", "3", "--pair"]) == 0
    snapshots = sorted(path for path in (root / "snapshots").iterdir() if path.is_dir())
    assert len(snapshots) == 1
    return snapshots[0].name


def _evidence_file(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(EVIDENCE), encoding="utf-8")
    return path


def _items(count: int, *, media: int | None = None) -> list:
    """Synthetic items; with ``media`` set, docs 0 and 1 are primary-mergeable within 72h."""
    seeds = make_smoke_snapshot.SEEDS
    items = [make_smoke_snapshot._doc(index, seeds[index % len(seeds)]) for index in range(count)]
    if media is not None and count >= 2:
        published = make_smoke_snapshot.date(2026, 1, 5).isoformat()
        items[0] = replace(items[0], media_ids=[media], event_type="event")
        items[1] = replace(
            items[1],
            media_ids=[media],
            event_type="event",
            title=items[0].title,
            published_at=f"{published}T00:00:00Z",
        )
    return items


def test_reproduction_of_a_consistent_snapshot_matches(data_root: Path, tmp_path: Path) -> None:
    snapshot = _publish_pair(data_root)
    out = tmp_path / "findings.json"
    code = dedup_diff.main(
        [
            "--snapshot",
            snapshot,
            "--data-root",
            str(data_root),
            "--threshold",
            "0.86",
            "--evidence",
            str(_evidence_file(tmp_path)),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["matches_snapshot"] is True
    assert report["cluster_id_set_identical"] is True
    assert report["member_sets_identical"] is True
    assert report["item_counts_match"] is True
    assert report["snapshot"]["non_singleton"] == 1
    assert report["reproduced"]["non_singleton"] == 1


def test_compare_passes_an_identical_reproduction() -> None:
    items = _items(3, media=7)  # {doc0, doc1} merge via primary; doc2 alone
    frozen = cluster_items(items, policy=DedupPolicy(thresholds={"same_lang_cosine": 0.86}))
    reproduced = cluster_items(items, policy=DedupPolicy(thresholds={"same_lang_cosine": 0.86}))
    report = dedup_diff._compare(
        frozen=frozen.clusters,
        reproduced=reproduced.clusters,
        frozen_items=len(items),
        reproduced_items=len(reproduced.items),
    )
    assert report["matches_snapshot"] is True
    assert report["cluster_id_set_identical"] is True
    assert report["member_sets_identical"] is True
    assert report["item_counts_match"] is True


def test_compare_rejects_same_counts_with_a_different_partition() -> None:
    """E's P2 repro: same items/clusters/non-singleton counts, disjoint ids and members."""
    base = _items(3)  # A, B, C — nothing merges (distinct seeds, no media_ids)
    frozen = cluster_items(
        _items(3, media=7), policy=DedupPolicy(thresholds={"same_lang_cosine": 0.86})
    )
    # The reproduction pairs (doc0, doc2) instead of (doc0, doc1): same aggregate counts,
    # different partition.
    reproduced_items = [
        replace(base[0], media_ids=[7], event_type="event"),
        base[1],
        replace(
            base[2],
            media_ids=[7],
            event_type="event",
            title=base[0].title,
            published_at="2026-01-06T00:00:00Z",
        ),
    ]
    reproduced = cluster_items(
        reproduced_items, policy=DedupPolicy(thresholds={"same_lang_cosine": 0.86})
    )

    report = dedup_diff._compare(
        frozen=frozen.clusters,
        reproduced=reproduced.clusters,
        frozen_items=len(base),
        reproduced_items=len(reproduced_items),
    )
    assert report["item_counts_match"] is True
    assert report["cluster_id_set_identical"] is False
    assert report["member_sets_identical"] is False
    assert report["matches_snapshot"] is False


def test_reason_labels_use_the_current_policy_verdict(data_root: Path) -> None:
    """E's P6: the sub-reason must agree with _pair_merge_score's own verdict."""
    snapshot = _publish_pair(data_root)
    items = {
        item.doc_id: item
        for item in dedup_diff.iter_snapshot_items(snapshot, data_root=data_root, verify=True)
    }
    frozen = dedup_diff.load_snapshot_clusters(snapshot, data_root=data_root)
    multi = next(cluster for cluster in frozen if len(cluster.member_doc_ids) > 1)
    first, second = (items[doc_id] for doc_id in multi.member_doc_ids)

    reason = dedup_diff._reason(first, second, 0.86)
    # The pair is primary-mergeable in the current policy, so the reason must say the
    # recorded split was wrong — not invent an unrelated rejection cause.
    assert reason == "primary_channel_should_merge"
