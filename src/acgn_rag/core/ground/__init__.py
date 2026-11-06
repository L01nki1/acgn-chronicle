"""WP4 grounded generation, citation alignment, abstention and compliance clipping."""

from acgn_rag.core.ground.generate import (
    ABSTAIN_ALL_CLAIMS_REJECTED,
    ABSTAIN_MODEL_EMPTY,
    ABSTAIN_NO_EVIDENCE,
    ANSWER_MAX_TOKENS,
    REJECTED_CLAIMS_REASON,
    SYSTEM_PROMPT,
    AnswerClient,
    AnswerDiagnostics,
    Completion,
    GroundedAnswerer,
    GroundedAnswerRun,
    build_grounding_prompt,
    generate_grounded_answer,
    generate_grounded_answer_run,
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
    "ABSTAIN_ALL_CLAIMS_REJECTED",
    "ABSTAIN_MODEL_EMPTY",
    "ABSTAIN_NO_EVIDENCE",
    "ANSWER_MAX_TOKENS",
    "REJECTED_CLAIMS_REASON",
    "SYSTEM_PROMPT",
    "AnswerClient",
    "AnswerDiagnostics",
    "Completion",
    "GroundedAnswerRun",
    "GroundedAnswerer",
    "GroundingComponentError",
    "GroundingValidationError",
    "InvalidModelResponseError",
    "InvalidRetrievedEvidenceError",
    "ProviderResponseError",
    "build_grounding_prompt",
    "generate_grounded_answer",
    "generate_grounded_answer_run",
    "make_openai_answer_client",
    "validate_grounded_answer",
]
