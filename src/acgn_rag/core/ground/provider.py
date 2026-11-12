"""OpenAI-compatible vendor adapter for grounded answer generation."""

from __future__ import annotations

import httpx

from acgn_rag.config import LLMConfig
from acgn_rag.core.ground.generate import AnswerClient, Completion, GroundedAnswerer
from acgn_rag.core.ground.validation import ProviderResponseError


def make_openai_answer_client(
    config: LLMConfig,
    *,
    http_client: httpx.Client | None = None,
) -> AnswerClient:
    """Build an answer client from the existing answer-specific vendor settings."""
    try:
        from openai import OpenAI, OpenAIError
    except ImportError as exc:  # pragma: no cover - CI installs the optional llm extra
        raise ImportError(
            "the answer client needs the [llm] extra: pip install -e '.[llm]'"
        ) from exc

    if http_client is None:
        vendor = OpenAI(
            base_url=config.answer_base_url or None,
            api_key=config.answer_api_key or None,
        )
    else:
        vendor = OpenAI(
            base_url=config.answer_base_url or None,
            api_key=config.answer_api_key or None,
            http_client=http_client,
        )

    class _Adapter:
        def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
            try:
                response = vendor.chat.completions.create(
                    model=config.answer_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=config.temperature,
                )
            except (OpenAIError, httpx.HTTPError) as exc:
                raise ProviderResponseError(
                    f"the answer provider transport failed: {type(exc).__name__}: {exc}"
                ) from exc
            if not response.choices:
                raise ProviderResponseError("the answer provider returned no completion choices")
            choice = response.choices[0]
            content = choice.message.content
            if content is None or not content.strip():
                raise ProviderResponseError(
                    "the answer provider returned no completion content "
                    f"(finish_reason={choice.finish_reason})"
                )
            usage = response.usage
            usage_counts = None
            if usage is not None:
                usage_counts = {
                    "prompt_tokens": int(usage.prompt_tokens),
                    "completion_tokens": int(usage.completion_tokens),
                    "total_tokens": int(usage.total_tokens),
                }
            return Completion(
                text=content,
                finish_reason=choice.finish_reason,
                usage=usage_counts,
            )

    return _Adapter()


def make_grounded_answerer(
    config: LLMConfig,
    *,
    http_client: httpx.Client | None = None,
) -> GroundedAnswerer:
    """Build the production answerer from the answer-line config, threshold included."""
    return GroundedAnswerer.from_config(
        config,
        make_openai_answer_client(config, http_client=http_client),
    )
