"""C6: invalid result counts must fail request validation, never look like an empty result."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from acgn_rag.contracts_export import build_schema
from acgn_rag.core.ports import RetrievalQuery


def _request(**updates: object) -> str:
    return json.dumps(
        {"text": "When does it release?", "snapshot_id": "20261028-abcdef012345", **updates}
    )


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_top_k_fails_wire_request_validation(top_k: int) -> None:
    with pytest.raises(ValidationError, match="top_k"):
        RetrievalQuery.model_validate_json(_request(top_k=top_k))


def test_minimum_top_k_and_omitted_default_remain_usable() -> None:
    assert RetrievalQuery.model_validate_json(_request(top_k=1)).top_k == 1
    assert RetrievalQuery.model_validate_json(_request()).top_k == 5


def test_query_schema_carries_the_same_minimum_and_default() -> None:
    top_k = build_schema()["models"]["RetrievalQuery"]["properties"]["top_k"]
    assert top_k["minimum"] == 1
    assert top_k["default"] == 5
