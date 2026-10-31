import test from "node:test";
import assert from "node:assert/strict";

import {
  MAX_QUOTE_CODEPOINTS,
  cacheKey,
  clipQuote,
  codePointLength,
  isRenderableCitation,
  selectRenderableExcerpts,
  sourceTypeLabel,
  viewState,
} from "../../src/acgn_rag/web/static/contract.mjs";

const citation = {
  source_type: "official",
  publisher: "Aurora Animation",
  title: "第二季开播公告",
  url: "https://example.com/acgn-chronicle/mock/aurora/season-2",
  published_at: "2026-09-12",
  chunk_id: "doc-90210::A::1",
};

test("clipQuote counts Unicode code points, not UTF-16 units", () => {
  const clipped = clipQuote("😀".repeat(250));
  assert.equal(codePointLength(clipped), MAX_QUOTE_CODEPOINTS);
  assert.ok(clipped.endsWith("\u2026"));
});

test("clipQuote leaves an exact-budget quote unchanged", () => {
  const exact = "😀".repeat(MAX_QUOTE_CODEPOINTS);
  assert.equal(clipQuote(exact), exact);
});

test("clipQuote keeps CJK quotes within the same budget", () => {
  const clipped = clipQuote("星".repeat(260));
  assert.equal(codePointLength(clipped), MAX_QUOTE_CODEPOINTS);
  assert.equal(clipped.at(-1), "\u2026");
});

test("clipQuote rejects a non-positive or fractional limit", () => {
  assert.throws(() => clipQuote("text", 0), RangeError);
  assert.throws(() => clipQuote("text", 2.5), RangeError);
});

test("a complete citation is renderable", () => {
  assert.equal(isRenderableCitation(citation), true);
});

test("a citation without url is never renderable", () => {
  assert.equal(isRenderableCitation({ ...citation, url: "" }), false);
  assert.equal(isRenderableCitation({ ...citation, url: "javascript:alert(1)" }), false);
});

test("a citation with an unknown source type is not rendered", () => {
  assert.equal(isRenderableCitation({ ...citation, source_type: "blog" }), false);
});

test("selectRenderableExcerpts drops unattributable cards and reports the count", () => {
  const result = selectRenderableExcerpts([
    { chunk_id: citation.chunk_id, score: 0.9, quote: citation.title, citation },
    {
      chunk_id: "doc-90210::A::2",
      score: 0.8,
      quote: "no url",
      citation: { ...citation, url: undefined },
    },
    { chunk_id: "broken", score: 0.7, quote: 42, citation },
  ]);
  assert.equal(result.items.length, 1);
  assert.equal(result.hiddenCount, 2);
  assert.equal(result.items[0].quote, citation.title);
});

test("selectRenderableExcerpts clips a hostile oversized quote before rendering", () => {
  const result = selectRenderableExcerpts([
    { chunk_id: citation.chunk_id, score: 0.9, quote: "x".repeat(260), citation },
  ]);
  assert.equal(result.items.length, 1);
  assert.equal(codePointLength(result.items[0].quote), MAX_QUOTE_CODEPOINTS);
  assert.equal(result.items[0].wasClipped, true);
});

test("cacheKey changes when the snapshot changes", () => {
  const base = { text: "第二季什么时候开播？", top_k: 5, config: "A", filters: null };
  const current = cacheKey({ ...base, snapshot_id: "20261030-abcdef012345" });
  const retained = cacheKey({ ...base, snapshot_id: "20261023-012345abcdef" });
  assert.notEqual(current, retained);
});

test("cacheKey canonicalizes filter order and drops empty families", () => {
  const left = cacheKey({
    snapshot_id: "s",
    text: "q",
    top_k: 5,
    config: "A",
    filters: { source_types: ["media", "official"], media_ids: [3, 1], since: "", until: "" },
  });
  const right = cacheKey({
    snapshot_id: "s",
    text: "q",
    top_k: 5,
    config: "A",
    filters: { source_types: ["official", "media"], media_ids: [1, 3] },
  });
  assert.equal(left, right);
});

test("empty retrieval and abstention are distinct states", () => {
  assert.equal(viewState("retrieve", []), "retrieve-empty");
  assert.equal(viewState("retrieve", [{ chunk_id: "x" }]), "retrieve-results");
  assert.equal(viewState("ask", { abstained: true, answer: "no evidence" }), "ask-abstained");
  assert.equal(viewState("ask", { abstained: false, answer: "answer" }), "ask-answer");
});

test("source type labels are stable", () => {
  assert.equal(sourceTypeLabel("official"), "官方");
  assert.equal(sourceTypeLabel("media"), "媒体");
  assert.equal(sourceTypeLabel("metadata"), "元数据");
});
