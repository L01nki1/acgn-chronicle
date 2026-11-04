"""WP4 grounded generation, citation alignment, abstention and compliance clipping."""

from acgn_rag.core.ground.generate import (
    ANSWER_MAX_TOKENS,
    SYSTEM_PROMPT,
    AnswerClient,
    GroundedAnswerer,
    build_grounding_prompt,
    generate_grounded_answer,
)
from acgn_rag.core.ground.provider import make_openai_answer_client
from acgn_rag.core.ground.validation import (
    GroundingComponentError,
    GroundingValidationError,
    InvalidModelResponseError,
    InvalidRetrievedEvidenceError,
    ProviderResponseError,
    validate_grounded_answer,
)

__all__ = [
    "ANSWER_MAX_TOKENS",
    "SYSTEM_PROMPT",
    "AnswerClient",
    "GroundedAnswerer",
    "GroundingComponentError",
    "GroundingValidationError",
    "InvalidModelResponseError",
    "InvalidRetrievedEvidenceError",
    "ProviderResponseError",
    "build_grounding_prompt",
    "generate_grounded_answer",
    "make_openai_answer_client",
    "validate_grounded_answer",
]
