# Embedding bake-off — measured evidence (2026-10-29, WP2/F)

The measured numbers behind the pre-registered model policy in `docs/RUNBOOK.md` §3.5 and the
encoder-identity rules in §3.4. `docs/PROJECT.md` §9 requires every report number to be reproducible
from `notebooks/`; this directory is that artifact for the model selection. The decision itself lives
in `docs/DECISIONS.md` (2026-10-29 entries) — this README only carries evidence.

## What was measured

Pilot bake-off on real trilingual ACGN text: 135 passages (Capcom JP press 70, ANN EN news 45,
Gcores ZH media 20), 16 queries over 8 gold entities, half cross-lingual (zh→JP, en→ZH). Same
machine and lane (torch CPU), same texts; only the encoder varies. Judging mirrors the team rule
(hit@5 = any top-5 passage containing the gold entity; MRR@5 on the first hit). Numbers are a
*selection* sample (n=16), not the G2 evaluation.

| model | dim | CPU tok/s | hit@5 | MRR@5 | zh-query hit | zh-query MRR |
|---|---|---|---|---|---|---|
| multilingual-e5-small | 384 | 5090 | 0.69 | 0.559 | 0.50 | 0.344 |
| multilingual-e5-base (CPU-lane default) | 768 | 1661 | 0.81 | 0.771 | 0.75 | 0.667 |
| multilingual-e5-large | 1024 | 421 | 0.81 | 0.771 | 0.62 | 0.625 |
| **Qwen3-Embedding-0.6B** | 1024 | 53 (bf16) / 206 (fp32) | **1.00** | **0.880** | **1.00** | 0.844 |
| **bge-m3 (dense)** | 1024 | 365 | **1.00** | 0.859 | **1.00** | **0.938** |

All three e5-base misses were cross-lingual queries on JP-source items; both Qwen3 and bge-m3
answered every zh and en query on JP sources directly, with no translation lane. bge-m3 was added
to the menu after a problem-driven community survey (r/LocalLLaMA threads; JP retrieval bench
0.8498 vs e5-base 0.8384); at n=16 its tie with Qwen3 is within noise — the G1a bake-off decides.

## GPU lanes (2026-10-29)

Quality is hardware-independent on this sample: every GPU run (A380 fp32, T4 fp32 and fp16, all
three models) reproduces the CPU hit/MRR exactly; A380 vectors agree with CPU at cosine
≥ 0.9999998 with identical rankings. Throughput is therefore reported **per lane, never across
GPUs**. Each A380 row is measured against its own in-run CPU baseline on the same machine
(i5-13400TEF, 16 threads) — a *different measurement series* from the bake-off's CPU column above
(different batch and passage-length mix), so the two CPU figures for one model are both real and
not interchangeable:

| lane (fp32 → fp16) | e5-base | bge-m3 (dense) | Qwen3-0.6B |
|---|---|---|---|
| this machine, OpenVINO / Arc A380 | 6,244 (3.8× its 1,644 in-run CPU) | 2,967 (6.9× its 431 in-run CPU) | n/a (decoder; run on CUDA) |
| Colab T4, torch 2.11+cu130, fp32 → fp16 | 14,660 → 51,235 | 3,389 → 16,380 | 2,039 → 6,883 |

On a 40k-chunk × 512-token corpus (≈20.5M tokens) even the slowest GPU lane builds one config in
≤ 50 min (Qwen3 fp16); the A380 lane builds e5-base in ~55 min and bge-m3 in ~1.9 h per config.

Earlier OpenVINO probe (2026-10-19, **different machine state**: i3-10105F, 4 torch threads, batch
16 padded to 439 tokens): e5-small measured **8.83×** its CPU reference — quoted in review as an
e5-small GPU row; it is not comparable to the rows above (different CPU, thread count and batching)
and is kept here only as the original proof that the OpenVINO route works.

fp32↔fp16 agreement, measured on T4 (all three models): min vector cosine 0.99976, query top-5
overlap 97.5–100 % (one shuffled slot in one e5-base query), corpus top-1 agreement 99.3–100 %,
max similarity-score delta 5e-4 — fp16 compute is ranking-safe at three orders of magnitude below
any threshold we use. Policy: compute in fp16 on GPU lanes, store vectors as fp32 (40k×1024 fp32
≈ 160 MB per config); fp16 stays a distinct encoder-identity dimension.

## Community-practice validation (2026-10-29 survey, problem-driven)

- e5 prefix/truncation pitfalls are the top documented failure mode (HF model card + discussions
  #5/#33: sentence-transformers needs manual prefixes; wrappers silently omit them; 512 truncation
  is silent) — the RUNBOOK §3.4 prefix and content-budget rules cover both.
- Chroma pain points reported in the wild (cosine-distance vs similarity confusion, filtered
  queries under-returning, hnsw param validation bugs) map one-to-one onto RUNBOOK §3.1's three
  pitfalls and the over-fetch rule; nothing new to guard beyond what §3.1 already has.
- Late chunking cannot apply to last-token-pooling decoder embedders (Qwen3) — confirmed as a known
  limitation in EmbedAnything #176 and RAGFlow #2301, not just our reading of the pooling math.
- News-dedup clustering practice prefers complete/average linkage over single linkage (chaining
  risk is textbook) and entity-aware keys for multilingual streams — independently supports the
  dedup design (entity key primary, complete-linkage text auxiliary, RUNBOOK §3.3).

## Reproducing

Scripts are copied verbatim from the measurement session (absolute paths document where it ran):

- `eval_embeds.py` — the five-model bake-off; reads the pilot corpus, writes `selection_results.json`.
- `gpu_bench.py` — the A380/OpenVINO lane; writes `a380_results.json` including the in-run CPU baselines.
- `t4_results.json` / `t4_fp16_agreement.json` — aggregates of the Colab T4 runs. The T4 scripts
  embed the pilot corpus inline (self-contained notebook style) and are **not committed**: the corpus
  is third-party text and stays out of git (C2 two-zone; `data/` never enters git).

The pilot corpus itself (`corpus/corpus.jsonl`, 135 passages) is likewise not committed. Archived
copy with provenance: semester workspace
`CS5481_Data Engineering/audit/20261029-embed-selection/` (sha256 prefix `03eee015006e909d`),
produced from the same source families the adapters cover (Capcom RSS, ANN archive, Gcores RSS —
`gcores.xml` snapshot archived beside it). Rerun after restoring it and pointing `CORPUS` at the
restored path, with `HF_HUB_OFFLINE=1` and the `[embed]` extra installed.

Environment of record: i5-13400TEF, 16 threads, torch 2.15.0+cpu, sentence-transformers 6.1.0,
transformers 5.20.1, OpenVINO 2026.4.1 (A380 lane), models from the local HF cache.
