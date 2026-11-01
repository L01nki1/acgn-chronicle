/**
 * Contract-facing front-end rules, kept pure so they can be tested without a DOM.
 *
 * Owner: C | Spec: contracts/KnowledgePort.md §2.4 and §6, docs/COMPLIANCE.md C3
 *
 * The server clips to 200 characters; this module is the client's second line of defence. It also
 * enforces "no URL, no render" before a card ever reaches the DOM.
 */

export const MAX_QUOTE_CODEPOINTS = 200;
export const ELLIPSIS = "\u2026";
export const SOURCE_TYPES = Object.freeze(["official", "media", "metadata"]);

const SOURCE_TYPE_SET = new Set(SOURCE_TYPES);
const CITATION_FIELDS = Object.freeze([
  "source_type",
  "publisher",
  "title",
  "url",
  "published_at",
  "chunk_id",
]);

export function codePointLength(value) {
  return Array.from(String(value ?? "")).length;
}

/**
 * Clip to a number of Unicode code points, not UTF-16 units. JS `.length`/`.slice()` would count an
 * emoji as two characters and produce a different budget from the Python server's `len(str)`.
 */
export function clipQuote(text, limit = MAX_QUOTE_CODEPOINTS) {
  if (!Number.isInteger(limit) || limit <= 0) {
    throw new RangeError("limit must be a positive integer");
  }
  const characters = Array.from(String(text ?? ""));
  if (characters.length <= limit) {
    return characters.join("");
  }
  return characters.slice(0, limit - 1).join("") + ELLIPSIS;
}

export function isRenderableCitation(citation) {
  if (!citation || typeof citation !== "object") {
    return false;
  }
  for (const field of CITATION_FIELDS) {
    if (typeof citation[field] !== "string" || citation[field].trim().length === 0) {
      return false;
    }
  }
  if (!SOURCE_TYPE_SET.has(citation.source_type)) {
    return false;
  }
  try {
    const parsed = new URL(citation.url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * Drop any excerpt whose citation cannot be attributed, and clip the quote again before rendering.
 * `hiddenCount` lets the page explain that content was withheld without showing it.
 */
export function selectRenderableExcerpts(excerpts) {
  const items = [];
  let hiddenCount = 0;
  for (const excerpt of Array.isArray(excerpts) ? excerpts : []) {
    if (
      !excerpt ||
      typeof excerpt !== "object" ||
      typeof excerpt.quote !== "string" ||
      !isRenderableCitation(excerpt.citation)
    ) {
      hiddenCount += 1;
      continue;
    }
    const quote = excerpt.quote;
    const clippedQuote = clipQuote(quote);
    items.push({
      ...excerpt,
      quote: clippedQuote,
      quoteLength: codePointLength(clippedQuote),
      wasClipped: codePointLength(quote) > codePointLength(clippedQuote),
    });
  }
  return { items, hiddenCount };
}

export function canonicalFilters(filters) {
  if (!filters || typeof filters !== "object") {
    return null;
  }
  const canonical = {};
  if (Array.isArray(filters.source_types) && filters.source_types.length > 0) {
    canonical.source_types = [...filters.source_types].sort();
  }
  if (Array.isArray(filters.media_ids) && filters.media_ids.length > 0) {
    canonical.media_ids = [...filters.media_ids].sort((left, right) => left - right);
  }
  if (typeof filters.since === "string" && filters.since.length > 0) {
    canonical.since = filters.since;
  }
  if (typeof filters.until === "string" && filters.until.length > 0) {
    canonical.until = filters.until;
  }
  return Object.keys(canonical).length > 0 ? canonical : null;
}

/**
 * Contract §6.5: the snapshot is part of the cache key. A question-only key would silently serve an
 * answer built from another corpus.
 */
export function cacheKey(parts = {}) {
  return JSON.stringify([
    parts.snapshot_id ?? "",
    String(parts.text ?? "").trim(),
    canonicalFilters(parts.filters),
    parts.top_k ?? null,
    parts.config ?? null,
  ]);
}

/** "Empty retrieval" and "the system refused" are different states, not two labels for one. */
export function viewState(kind, payload) {
  if (kind === "retrieve") {
    return Array.isArray(payload) && payload.length > 0 ? "retrieve-results" : "retrieve-empty";
  }
  if (kind === "ask") {
    if (!payload || typeof payload !== "object") {
      return "ask-error";
    }
    if (payload.abstained === true) {
      return "ask-abstained";
    }
    return typeof payload.answer === "string" ? "ask-answer" : "ask-error";
  }
  return "error";
}

export function sourceTypeLabel(sourceType) {
  return { official: "官方", media: "媒体", metadata: "元数据" }[sourceType] ?? "未知来源";
}
