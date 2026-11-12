"""Retrieval-score abstention threshold (`docs/EVAL_PROTOCOL.md` §6 and §8).

Without a threshold, abstention can only happen after generation, so the calibration set has
nothing to sweep. These tests pin the observable contract: the decision is made from the best
request-local retrieval score, before any prompt or provider call, and `score == threshold`
stays answerable.
"""

from __future__ import annotations

import json

import pytest

from acgn_rag.config import LLMConfig
from acgn_rag.core.ground import (
    ABSTAIN_NO_EVIDENCE,
    ABSTAIN_RETRIEVAL_BELOW_THRESHOLD,
    RETRIEVAL_BELOW_THRESHOLD_REASON,
    Completion,
    GroundedAnswerer,
    InvalidRetrievedEvidenceError,
    PromptBudgetError,
    generate_grounded_answer,
    generate_grounded_answer_run,
)
from acgn_rag.core.ground.provider import make_grounded_answerer
from acgn_rag.core.ports import Citation, RetrievalQuery, RetrievedChunk

SNAPSHOT_ID = "20261104-abcdef012345"
ENV_KEY = "ACGN_ANSWER_MIN_RETRIEVAL_SCORE"

ZH_REASON = "检索到的最佳证据得分低于配置的检索阈值。未尝试作答。"
JA_REASON = "取得した根拠の最高スコアが設定された検索しきい値を下回ったため、回答は試みていません。"


class FakeAnswerClient:
    """Deterministic client that records whether the provider was called."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        self.calls.append((system, user, max_tokens))
        return Completion(text=self.response, finish_reason="stop")


def _citation(index: int) -> Citation:
    return Citation(
        source_type="official",
        publisher=f"Publisher {index}",
        title=f"Announcement {index}",
        url=f"https://example.com/news/{index}",
        published_at="2026-11-04",
        chunk_id=f"chunk-{index}",
    )


def _chunk(index: int, *, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        doc_id=f"doc-{index}",
        snapshot_id=SNAPSHOT_ID,
        text=f"Evidence body {index} with a concrete fact.",
        score=score,
        citation=_citation(index),
    )


def _query(text: str = "What concrete facts were announced?") -> RetrievalQuery:
    return RetrievalQuery(text=text, snapshot_id=SNAPSHOT_ID)


def _response() -> str:
    return json.dumps({"claims": [{"text": "Supported fact.", "evidence_indices": [1]}]})


def test_default_answerer_still_calls_the_provider() -> None:
    client = FakeAnswerClient(_response())
    answerer = GroundedAnswerer(client)

    answer = answerer.answer(_query(), [_chunk(1, score=0.05)])

    assert answerer.min_retrieval_score is None
    assert not answer.abstained
    assert answer.answer == "Supported fact. [1]"
    assert len(client.calls) == 1


def test_below_threshold_refuses_before_the_provider_is_called() -> None:
    client = FakeAnswerClient(_response())

    run = generate_grounded_answer_run(
        _query(),
        [_chunk(1, score=0.42), _chunk(2, score=0.31)],
        client,
        min_retrieval_score=0.5,
    )

    assert client.calls == []
    assert run.answer.abstained
    assert run.answer.answer == RETRIEVAL_BELOW_THRESHOLD_REASON
    assert run.answer.abstain_reason == RETRIEVAL_BELOW_THRESHOLD_REASON
    assert list(run.answer.citations) == []
    assert list(run.answer.cited_chunk_ids) == []
    assert run.diagnostics.abstain_code == ABSTAIN_RETRIEVAL_BELOW_THRESHOLD
    assert run.diagnostics.provider_completion is False
    assert run.diagnostics.min_retrieval_score == 0.5
    assert run.diagnostics.retrieval_top_score == pytest.approx(0.42)


def test_plain_generate_entry_point_applies_the_same_threshold() -> None:
    client = FakeAnswerClient(_response())

    answer = generate_grounded_answer(
        _query(), [_chunk(1, score=0.2)], client, min_retrieval_score=0.3
    )

    assert answer.abstained
    assert client.calls == []


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("这部作品什么时候播出?", ZH_REASON),
        ("この作品はいつ放送されますか?", JA_REASON),
    ],
)
def test_threshold_refusal_uses_the_question_language(question: str, expected: str) -> None:
    client = FakeAnswerClient(_response())

    answer = generate_grounded_answer(
        _query(question), [_chunk(1, score=0.2)], client, min_retrieval_score=0.9
    )

    assert answer.abstained
    assert answer.answer == expected
    assert answer.abstain_reason == expected
    assert client.calls == []


@pytest.mark.parametrize(("score", "threshold"), [(0.7, 0.7), (0.71, 0.7), (0.9, 0.7)])
def test_score_at_or_above_the_threshold_answers_normally(score: float, threshold: float) -> None:
    client = FakeAnswerClient(_response())

    run = generate_grounded_answer_run(
        _query(), [_chunk(1, score=score)], client, min_retrieval_score=threshold
    )

    assert not run.answer.abstained
    assert run.answer.answer == "Supported fact. [1]"
    assert run.diagnostics.abstain_code is None
    assert len(client.calls) == 1


def test_from_config_reads_the_env_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_KEY, "0.75")
    config = LLMConfig(answer_base_url="http://example.invalid", answer_model="m")
    client = FakeAnswerClient(_response())

    answerer = GroundedAnswerer.from_config(config, client)

    assert answerer.min_retrieval_score == 0.75
    answer = answerer.answer(_query(), [_chunk(1, score=0.5)])
    assert answer.abstained
    assert answer.answer == RETRIEVAL_BELOW_THRESHOLD_REASON
    assert client.calls == []


@pytest.mark.parametrize("raw", [None, ""])
def test_unset_or_empty_env_keeps_the_threshold_disabled(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    if raw is None:
        monkeypatch.delenv(ENV_KEY, raising=False)
    else:
        monkeypatch.setenv(ENV_KEY, raw)

    config = LLMConfig(answer_base_url="http://example.invalid", answer_model="m")

    assert config.answer_min_retrieval_score is None


def test_zero_env_threshold_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_KEY, "0")
    config = LLMConfig(answer_base_url="http://example.invalid", answer_model="m")

    assert config.answer_min_retrieval_score == 0.0


def test_provider_factory_respects_the_configured_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_KEY, "0.6")
    config = LLMConfig(answer_base_url="http://example.invalid", answer_model="m")
    fake_client = FakeAnswerClient(_response())
    monkeypatch.setattr(
        "acgn_rag.core.ground.provider.make_openai_answer_client",
        lambda config, http_client=None: fake_client,
    )

    answerer = make_grounded_answerer(config)

    assert answerer.min_retrieval_score == 0.6
    answer = answerer.answer(_query(), [_chunk(1, score=0.2)])
    assert answer.abstained
    assert fake_client.calls == []


@pytest.mark.parametrize(("threshold", "recorded"), [(None, None), (0.5, 0.5)])
def test_diagnostics_record_the_threshold_and_top_score(
    threshold: float | None, recorded: float | None
) -> None:
    client = FakeAnswerClient(_response())

    run = generate_grounded_answer_run(
        _query(),
        [_chunk(1, score=0.55), _chunk(2, score=0.91)],
        client,
        min_retrieval_score=threshold,
    )

    assert not run.answer.abstained
    assert run.diagnostics.min_retrieval_score == recorded
    assert run.diagnostics.retrieval_top_score == pytest.approx(0.91)


def test_no_evidence_abstention_records_the_configured_threshold() -> None:
    run = generate_grounded_answer_run(
        _query(), [], FakeAnswerClient(_response()), min_retrieval_score=0.5
    )

    assert run.answer.abstained
    assert run.diagnostics.abstain_code == ABSTAIN_NO_EVIDENCE
    assert run.diagnostics.min_retrieval_score == 0.5
    assert run.diagnostics.retrieval_top_score is None


def test_request_validation_still_precedes_the_threshold() -> None:
    client = FakeAnswerClient(_response())
    duplicate = [_chunk(1, score=0.1), _chunk(1, score=0.1)]

    with pytest.raises(InvalidRetrievedEvidenceError):
        generate_grounded_answer_run(_query(), duplicate, client, min_retrieval_score=0.9)

    assert client.calls == []


def test_threshold_refusal_is_deterministic() -> None:
    first = generate_grounded_answer_run(
        _query(), [_chunk(1, score=0.3)], FakeAnswerClient(_response()), min_retrieval_score=0.8
    )
    second = generate_grounded_answer_run(
        _query(), [_chunk(1, score=0.3)], FakeAnswerClient(_response()), min_retrieval_score=0.8
    )

    assert first.answer.model_dump() == second.answer.model_dump()
    assert first.diagnostics == second.diagnostics


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
def test_non_finite_env_threshold_fails_loudly(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(ENV_KEY, raw)

    with pytest.raises(ValueError, match=ENV_KEY):
        LLMConfig(answer_base_url="http://example.invalid", answer_model="m")


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_threshold_argument_is_rejected(threshold: float) -> None:
    client = FakeAnswerClient(_response())

    with pytest.raises(ValueError, match="min_retrieval_score"):
        generate_grounded_answer_run(
            _query(), [_chunk(1, score=0.5)], client, min_retrieval_score=threshold
        )

    assert client.calls == []


@pytest.mark.parametrize(
    "scores",
    [
        (float("nan"),),
        (float("nan"), 0.9),
        (0.9, float("nan")),
        (-float("inf"), 0.9),
        (0.9, float("inf")),
    ],
)
def test_non_finite_retrieval_score_is_rejected(scores: tuple[float, ...]) -> None:
    client = FakeAnswerClient(_response())
    chunks = [_chunk(index, score=score) for index, score in enumerate(scores, 1)]

    with pytest.raises(ValueError, match="finite"):
        generate_grounded_answer_run(_query(), chunks, client)

    assert client.calls == []


def test_straddling_scores_use_the_best_request_local_score() -> None:
    client = FakeAnswerClient(_response())
    answered = generate_grounded_answer_run(
        _query(),
        [_chunk(1, score=0.1), _chunk(2, score=0.9)],
        client,
        min_retrieval_score=0.5,
    )

    assert not answered.answer.abstained
    assert answered.diagnostics.retrieval_top_score == pytest.approx(0.9)
    assert len(client.calls) == 1

    refused = generate_grounded_answer_run(
        _query(),
        [_chunk(1, score=0.2), _chunk(2, score=0.4)],
        FakeAnswerClient(_response()),
        min_retrieval_score=0.5,
    )

    assert refused.answer.abstained
    assert refused.diagnostics.retrieval_top_score == pytest.approx(0.4)


def test_zero_threshold_is_not_treated_as_disabled() -> None:
    answered = generate_grounded_answer(
        _query(), [_chunk(1, score=0.0)], FakeAnswerClient(_response()), min_retrieval_score=0.0
    )
    assert not answered.abstained

    refused = generate_grounded_answer(
        _query(), [_chunk(1, score=-0.1)], FakeAnswerClient(_response()), min_retrieval_score=0.0
    )
    assert refused.abstained


def test_gate_runs_before_the_prompt_budget() -> None:
    def large_chunk(*, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            snapshot_id=SNAPSHOT_ID,
            text="x" * 15_000,
            score=score,
            citation=_citation(1),
        )

    client = FakeAnswerClient(_response())
    answer = generate_grounded_answer(
        _query(), [large_chunk(score=0.1)], client, min_retrieval_score=0.5
    )
    assert answer.abstained
    assert client.calls == []

    with pytest.raises(PromptBudgetError):
        generate_grounded_answer(
            _query(),
            [large_chunk(score=0.9)],
            FakeAnswerClient(_response()),
            min_retrieval_score=0.5,
        )


def test_threshold_gate_does_not_build_a_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from acgn_rag.core.ground import generate as generate_module

    calls: list[tuple[object, object]] = []
    real_builder = generate_module.build_grounding_prompt

    def spy(query: object, chunks: object) -> tuple[str, str]:
        calls.append((query, chunks))
        return real_builder(query, chunks)  # type: ignore[arg-type]

    monkeypatch.setattr(generate_module, "build_grounding_prompt", spy)

    refused = generate_module.generate_grounded_answer(
        _query(), [_chunk(1, score=0.1)], FakeAnswerClient(_response()), min_retrieval_score=0.5
    )
    assert refused.abstained
    assert calls == []

    generate_module.generate_grounded_answer(
        _query(), [_chunk(1, score=0.9)], FakeAnswerClient(_response()), min_retrieval_score=0.5
    )
    assert len(calls) == 1
