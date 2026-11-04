"""OpenAI-compatible vendor adapter for grounded answer generation."""

from __future__ import annotations

import httpx

from acgn_rag.config import LLMConfig
from acgn_rag.core.ground.generate import AnswerClient
from acgn_rag.core.ground.validation import ProviderResponseError


def make_openai_answer_client(
    config: LLMConfig,
    *,
    http_client: httpx.Client | None = None,
) -> AnswerClient:
    """Build an answer client from the existing answer-specific vendor settings."""
    try:
        from openai import OpenAI
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
        def complete(self, system: str, user: str, *, max_tokens: int) -> str:
            response = vendor.chat.completions.create(
                model=config.answer_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=config.temperature,
            )
            if not response.choices:
                raise ProviderResponseError("the answer provider returned no completion choices")
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                raise ProviderResponseError(
                    f"the answer provider did not finish normally: {choice.finish_reason}"
                )
            content = choice.message.content
            if content is None or not content.strip():
                raise ProviderResponseError("the answer provider returned no completion content")
            return content

    return _Adapter()
