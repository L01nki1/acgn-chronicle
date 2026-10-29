"""Encoder lane: identity, model specs, prefixes/prompts, headers, cache keys, throughput.

Owner: F | Spec: RUNBOOK §3.4-§3.5

Everything runs on deterministic fakes with zero network and no [embed] extra (numpy gates the
module). The real-e5 smoke test at the bottom is `slow` and runs only when ACGN_EMBED_SLOW=1 and
the [embed] extra are present; its offline invocation is documented in its docstring.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

np = pytest.importorskip("numpy")  # [embed] extra; CI installs [dev] only and skips this module

from acgn_rag.config import EmbedConfig  # noqa: E402
from acgn_rag.core.embed import (  # noqa: E402
    HEADER_VERSION,
    MODEL_SPECS,
    Embedder,
    EncoderIdentity,
    PassageInput,
    build_header,
    cache_key,
    measure_throughput,
    model_spec,
)
from acgn_rag.core.embed import encoder as encoder_module  # noqa: E402

DIM = 32
MODEL_E5 = "intfloat/multilingual-e5-base"
PASSAGE = PassageInput(
    title="Monster Hunter Wilds expansion",
    publisher="Capcom",
    published_at="2026-02-28",
    body="Capcom announced the expansion goes on sale worldwide on 2026-02-28.",
)


class FakeModel:
    """Deterministic vectors without a model download; records exactly what encode() received.

    seed = sha256(text) -> 32 standard-normal floats: the same text gives the identical vector,
    different texts give different vectors.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        prompt_name: str | None = None,
        **_kwargs: Any,
    ) -> np.ndarray:
        self.calls.append(
            {"texts": list(texts), "batch_size": batch_size, "prompt_name": prompt_name}
        )
        return np.array([self.vector(text) for text in texts], dtype=np.float32)

    def vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)


class FakeTokenizer:
    """One token per 4 characters: deterministic, and unlike len(text) so a test can tell
    tokenizer tokens from character counts (issue #8)."""

    def __call__(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=list(range(max(1, len(text) // 4))))


def embedder_for(model_name: str, model: FakeModel) -> Embedder:
    return Embedder(EmbedConfig(model_name=model_name, device="cpu", batch_size=8), model=model)


def passage_text(item: PassageInput) -> str:
    """The passage string the model should receive: header + newline + body, before any prefix."""
    header = build_header(item.title, item.publisher, item.published_at)
    return f"{header}\n{item.body}"


def test_model_specs_cover_the_four_registered_families() -> None:
    e5 = model_spec(MODEL_E5)
    assert e5.mode == "prefix"
    assert e5.query == EmbedConfig().query_prefix == "query: "
    assert e5.passage == EmbedConfig().passage_prefix == "passage: "

    qwen = model_spec("Qwen/Qwen3-Embedding-0.6B")
    assert qwen.mode == "prompt"
    assert qwen.query == "query"
    assert qwen.passage is None  # documents carry no prompt (RUNBOOK §3.5)

    bge = model_spec("BAAI/bge-m3")
    assert bge.mode == "none"
    assert bge.query is None
    assert bge.passage is None

    gemma = model_spec("google/embeddinggemma-300m")
    assert gemma.mode == "prompt"
    assert gemma.query == "query"
    assert gemma.passage == "document"

    assert "multilingual-e5" in MODEL_SPECS  # the registry is what the lookup reads


def test_unknown_model_family_is_rejected_not_guessed() -> None:
    with pytest.raises(ValueError, match="no registered model spec"):
        model_spec("BAAI/bge-large-zh-v1.5")


def test_e5_prefixes_reach_the_model_input() -> None:
    fake = FakeModel()
    embedder = embedder_for(MODEL_E5, fake)
    embedder.encode_passages([PASSAGE])
    embedder.encode_queries(["new Monster Hunter release date"])
    assert fake.calls[0]["texts"] == [f"passage: {passage_text(PASSAGE)}"]
    assert fake.calls[0]["prompt_name"] is None
    assert fake.calls[1]["texts"] == ["query: new Monster Hunter release date"]


def test_passage_input_shapes_are_coerced() -> None:
    fake = FakeModel()
    embedder = embedder_for(MODEL_E5, fake)
    embedder.encode_passages(
        [
            ("plain header", "plain body"),
            {"title": "T", "publisher": "P", "published_at": "2026-01-01", "body": "B"},
        ]
    )
    texts = fake.calls[0]["texts"]
    assert texts[0] == "passage: plain header\nplain body"
    assert texts[1] == "passage: T | P | 2026-01-01\nB"


def test_qwen3_query_prompt_is_forwarded_and_passages_get_none() -> None:
    fake = FakeModel()
    embedder = embedder_for("Qwen/Qwen3-Embedding-0.6B", fake)
    embedder.encode_queries(["新作の発売日"])
    embedder.encode_passages([PASSAGE])
    assert fake.calls[0]["prompt_name"] == "query"
    assert fake.calls[0]["texts"] == ["新作の発売日"]  # prompt mode adds no prefix
    assert fake.calls[1]["prompt_name"] is None  # documents carry no prompt (RUNBOOK §3.5)
    assert fake.calls[1]["texts"] == [passage_text(PASSAGE)]


def test_bge_m3_gets_neither_prefix_nor_prompt() -> None:
    fake = FakeModel()
    embedder = embedder_for("BAAI/bge-m3", fake)
    embedder.encode_queries(["query text"])
    embedder.encode_passages([("h", "b")])
    assert fake.calls[0]["texts"] == ["query text"]
    assert fake.calls[0]["prompt_name"] is None
    assert fake.calls[1]["texts"] == ["h\nb"]


def test_vectors_are_l2_normalized_float32() -> None:
    embedder = embedder_for(MODEL_E5, FakeModel())
    passages = embedder.encode_passages([PASSAGE, ("h2", "another body")])
    queries = embedder.encode_queries(["q1", "q2", "q3"])
    assert passages.shape == (2, DIM)
    assert queries.shape == (3, DIM)
    for vectors in (passages, queries):
        assert vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_same_text_same_vector_different_text_different_vector() -> None:
    fake = FakeModel()
    np.testing.assert_array_equal(fake.vector("stable text"), fake.vector("stable text"))
    assert not np.array_equal(fake.vector("stable text"), fake.vector("stable texr"))
    embedder = embedder_for(MODEL_E5, fake)
    np.testing.assert_array_equal(embedder.encode_queries(["x"]), embedder.encode_queries(["x"]))


def test_identity_slug_is_stable_and_filesystem_safe() -> None:
    kwargs: dict[str, Any] = {
        "model": MODEL_E5,
        "backend": "torch",
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 32,
        "lib_versions": (("numpy", "2.4.6"),),
    }
    slug = EncoderIdentity(**kwargs).slug()
    assert slug == EncoderIdentity(**kwargs).slug()  # sha256: stable across runs and processes
    assert re.fullmatch(r"[a-z0-9-]+", slug), slug
    assert slug.startswith("multilingual-e5-base-torch-cpu-fp32-b32-")
    assert EncoderIdentity(**{**kwargs, "device": "cuda"}).slug() != slug
    assert EncoderIdentity(**{**kwargs, "lib_versions": ()}).slug() != slug


def test_to_metadata_is_flat_scalars_only() -> None:
    identity = EncoderIdentity(
        model=MODEL_E5,
        lib_versions=(("sentence-transformers", "6.2.0"), ("numpy", "2.4.6")),
    )
    meta = identity.to_metadata()
    assert meta["model"] == MODEL_E5
    assert isinstance(meta["batch_size"], int)
    assert meta["lib_sentence_transformers"] == "6.2.0"
    for value in meta.values():  # Chroma metadata: scalars only, never arrays (RUNBOOK §3.1)
        assert isinstance(value, (str, int))
        assert not isinstance(value, bool)


def test_build_header_joins_fields_and_drops_empty_ones() -> None:
    assert build_header("T", "P", "2026-01-01") == "T | P | 2026-01-01"
    assert build_header("T", "", "2026-01-01") == "T | 2026-01-01"
    assert build_header("", "", "") == ""


def test_cache_key_changes_with_every_input() -> None:
    identity = EncoderIdentity(model=MODEL_E5)
    base = cache_key(identity, MODEL_E5, "passage: ", "H", "B")
    assert re.fullmatch(r"[0-9a-f]{64}", base)
    assert cache_key(identity, MODEL_E5, "passage: ", "H", "B") == base
    changed = {
        "header": cache_key(identity, MODEL_E5, "passage: ", "H2", "B"),
        "body": cache_key(identity, MODEL_E5, "passage: ", "H", "B2"),
        "model": cache_key(identity, "Qwen/Qwen3-Embedding-0.6B", "passage: ", "H", "B"),
        "prefix": cache_key(identity, MODEL_E5, "query: ", "H", "B"),
        "prompt": cache_key(identity, MODEL_E5, None, "H", "B"),
        "identity": cache_key(
            EncoderIdentity(model=MODEL_E5, device="cuda"), MODEL_E5, "passage: ", "H", "B"
        ),
    }
    for what, key in changed.items():
        assert key != base, what


def test_cache_key_binds_the_header_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # same content, new header template -> new key, or stale vectors would be silently reused
    identity = EncoderIdentity(model=MODEL_E5)
    base = cache_key(identity, MODEL_E5, "passage: ", "H", "B")
    monkeypatch.setattr(encoder_module, "HEADER_VERSION", "h2")
    assert cache_key(identity, MODEL_E5, "passage: ", "H", "B") != base
    assert HEADER_VERSION == "h1"  # the package re-export keeps the documented constant


def test_measure_throughput_reports_tokens_not_characters() -> None:
    fake = FakeModel()
    embedder = embedder_for(MODEL_E5, fake)
    texts = ["abcdef", "ghijklmnop"]  # 6 and 10 chars -> 1 and 2 tokens at 4 chars per token
    result = measure_throughput(embedder, FakeTokenizer(), texts, batch_size=2)
    assert result["tokens"] == 3  # 6 // 4 + 10 // 4, not the 16 characters (issue #8)
    assert result["tokens_per_s"] > 0
    assert result["seconds"] > 0
    assert result["batch_size"] == 2
    assert fake.calls[0]["batch_size"] == 2
    assert result["peak_rss_mb"] is None or result["peak_rss_mb"] > 0
    assert "items_per_s" not in result  # forbidden reporting unit (issue #8)
    # texts reached the model as passages, with the passage prefix and no header
    assert fake.calls[0]["texts"] == [f"passage: {text}" for text in texts]


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("ACGN_EMBED_SLOW") != "1",
    reason="loads the real e5 model; run locally with ACGN_EMBED_SLOW=1 and the [embed] extra",
)
def test_real_e5_smoke() -> None:
    """intfloat/multilingual-e5-base from the local HF cache: dims, norms, prefixes, ranking.

    Offline invocation (from the repo root, no network):
      HF_HUB_OFFLINE=1 \\
      HF_HOME=$HOME/LocalDev/cityu/hf-cache/2026-2027_SemA/cs5481-acgn-chronicle \\
      ACGN_EMBED_SLOW=1 python -m pytest tests/test_embed_encoder.py -m slow -q
    """
    st = pytest.importorskip("sentence_transformers")
    model = st.SentenceTransformer(MODEL_E5, device="cpu")
    embedder = Embedder(EmbedConfig(model_name=MODEL_E5, device="cpu", batch_size=4), model=model)
    matching = PassageInput(
        title="Monster Hunter Wilds expansion",
        publisher="Capcom",
        published_at="2026-02-28",
        body="Capcom announced the Monster Hunter Wilds expansion goes on sale on 2026-02-28.",
    )
    unrelated = PassageInput(
        title="Winter anime season",
        publisher="Anime News Network",
        published_at="2026-01-05",
        body="The studio confirmed a new director for its original winter anime series.",
    )
    passages = embedder.encode_passages([matching, unrelated])
    query = embedder.encode_queries(["When does the Monster Hunter Wilds expansion go on sale?"])

    assert passages.shape == (2, 768)
    assert query.shape == (1, 768)
    assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(query, axis=1), 1.0, atol=1e-5)

    tokenizer = model.tokenizer
    passage_ids = tokenizer(f"passage: {matching.body}").input_ids
    query_ids = tokenizer("query: Monster Hunter expansion release date").input_ids
    # decode (not single tokens): "query" splits into subwords, but the prefix must still sit
    # at the head of the token stream, ahead of the body (RUNBOOK §3.4 content-budget rule)
    assert "passage:" in tokenizer.decode(passage_ids[:4])
    assert "query:" in tokenizer.decode(query_ids[:5])

    # unit vectors: dot product == cosine similarity
    cos_matching = float(np.dot(query[0], passages[0]))
    cos_unrelated = float(np.dot(query[0], passages[1]))
    assert cos_matching > cos_unrelated
