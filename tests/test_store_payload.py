"""Payload shape: flat scalars, int-epoch time, native int[] media_ids, adapter seam."""

from __future__ import annotations

import dataclasses

import pytest

from acgn_rag.core.store.payload import (
    ChunkPayload,
    iso_to_epoch,
    payload_from_fields,
    payload_from_metadata,
    to_payload,
)

# Verified against datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp().
EPOCH_20261028_UTC = 1793145600


def _fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "chunk_id": "doc-1::A::0",
        "doc_id": "doc-1",
        "snapshot_id": "20261028abcdef1234",
        "title": "Title",
        "url": "https://example.com/1",
        "publisher": "Capcom",
        "source_type": "official",
        "published_at": "2026-10-28T00:00:00+00:00",
        "lang": "ja",
        "body": "Body text",
        "media_ids": [1, 2],
    }
    fields.update(overrides)
    return fields


def test_iso_to_epoch_is_utc_explicit() -> None:
    # "+00:00", "Z" and a naive stamp (read AS UTC, not as local time) agree to the second.
    assert iso_to_epoch("2026-10-28T00:00:00+00:00") == EPOCH_20261028_UTC
    assert iso_to_epoch("2026-10-28T00:00:00Z") == EPOCH_20261028_UTC
    assert iso_to_epoch("2026-10-28T00:00:00") == EPOCH_20261028_UTC


def test_iso_to_epoch_honours_offsets() -> None:
    # 09:30 at +09:00 is 00:30 UTC - 1800 seconds after midnight.
    assert iso_to_epoch("2026-10-28T09:30:00+09:00") == EPOCH_20261028_UTC + 1800


def test_payload_is_frozen() -> None:
    payload = payload_from_fields(**_fields())
    with pytest.raises(dataclasses.FrozenInstanceError):
        payload.title = "mutated"  # type: ignore[misc]


def test_payload_from_fields_freezes_media_ids_into_a_tuple() -> None:
    payload = payload_from_fields(**_fields(media_ids=[3, 1, 3]))
    assert payload.media_ids == (3, 1, 3)
    assert isinstance(payload.media_ids, tuple)


def test_payload_from_fields_accepts_published_ts_directly() -> None:
    payload = payload_from_fields(**_fields(published_ts=42, published_at=None))
    assert payload.published_ts == 42


def test_payload_from_fields_requires_some_published_form() -> None:
    fields = _fields()
    del fields["published_at"]
    with pytest.raises(TypeError, match="published"):
        payload_from_fields(**fields)


def test_to_payload_is_scalars_plus_int_array_only() -> None:
    meta = to_payload(payload_from_fields(**_fields()))
    assert meta["published_ts"] == EPOCH_20261028_UTC
    assert isinstance(meta["published_ts"], int)
    assert meta["media_ids"] == [1, 2]
    assert all(isinstance(m, int) for m in meta["media_ids"])
    for key, value in meta.items():
        assert isinstance(value, (str, int, float, bool, list)), f"{key} is not scalar/array"
        if isinstance(value, list):
            assert all(isinstance(item, int) for item in value)  # flat, one type: int[]


def test_to_payload_stores_no_body_and_round_trips() -> None:
    payload = payload_from_fields(**_fields())
    meta = to_payload(payload)
    assert "body" not in meta  # the body is the Chroma document, never the metadata
    assert payload_from_metadata(meta, document=payload.body) == payload


def test_to_payload_omits_none_ingest_category() -> None:
    without = to_payload(payload_from_fields(**_fields()))
    assert "ingest_category" not in without
    tagged = payload_from_fields(**_fields(ingest_category="corporate"))
    assert to_payload(tagged)["ingest_category"] == "corporate"
    assert payload_from_metadata(to_payload(tagged), document="Body text") == tagged


def test_chunk_payload_field_list_is_explicit() -> None:
    names = [field.name for field in dataclasses.fields(ChunkPayload)]
    assert names == [
        "chunk_id",
        "doc_id",
        "snapshot_id",
        "title",
        "url",
        "publisher",
        "source_type",
        "published_ts",
        "lang",
        "body",
        "media_ids",
        "ingest_category",
    ]
