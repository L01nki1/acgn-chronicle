import {
  MAX_QUOTE_CODEPOINTS,
  cacheKey,
  isRenderableCitation,
  selectRenderableExcerpts,
  sourceTypeLabel,
  viewState,
} from "./contract.mjs";
import { createQueryQueue } from "./queue.mjs";
import {
  applyDocumentI18n,
  applyStatic,
  bindLangToggle,
  seedContentValue,
  syncToggle,
  t,
} from "./i18n.mjs";

const ACTIVE_CONFIG = "A";
const STRENGTH = { official: 3, media: 2, metadata: 1 };

class ApiError extends Error {
  constructor(status, detail, retryAfter = null) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.retryAfter = retryAfter;
  }
}

class FormError extends Error {
  constructor(message) {
    super(message);
    this.name = "FormError";
  }
}

const dom = {
  form: document.querySelector("#query-form"),
  question: document.querySelector("#question"),
  submit: document.querySelector("#submit-button"),
  formNote: document.querySelector("#form-note"),
  snapshotSelect: document.querySelector("#snapshot-select"),
  snapshotId: document.querySelector("#snapshot-id"),
  resultState: document.querySelector("#result-state"),
  resultStatus: document.querySelector("#result-status"),
  resultMeta: document.querySelector("#result-meta"),
  resultsTitle: document.querySelector("#results-title"),
  contractNotes: document.querySelector("#contract-notes"),
  questionCount: document.querySelector("#question-count"),
  since: document.querySelector("#since"),
  until: document.querySelector("#until"),
  mediaId: document.querySelector("#media-id"),
  topK: document.querySelector("#top-k"),
  modeInputs: document.querySelectorAll('input[name="mode"]'),
  sourceInputs: document.querySelectorAll('input[name="source_type"]'),
};

const state = {
  inventory: null,
  current: null,
  lastRequest: null,
  lastRender: null,
  lastResponse: null,
  stale: false,
  // One cache per endpoint: the contract cache tuple does not include the route, so sharing one
  // map between /retrieve and /ask would mix two response shapes under the same key.
  cache: { retrieve: new Map(), ask: new Map() },
};

const queryQueue = createQueryQueue();

function element(tag, { className, text, attrs } = {}, children = []) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = String(text);
  }
  for (const [name, value] of Object.entries(attrs ?? {})) {
    node.setAttribute(name, String(value));
  }
  for (const child of children) {
    node.append(child);
  }
  return node;
}

function replace(node, ...children) {
  node.replaceChildren(...children);
}

function currentMode() {
  const checked = [...dom.modeInputs].find((input) => input.checked);
  return checked?.value === "ask" ? "ask" : "retrieve";
}

function selectedSourceTypes() {
  return [...dom.sourceInputs].filter((input) => input.checked).map((input) => input.value);
}

function setFormNote(message, isError = false) {
  dom.formNote.textContent = message;
  dom.formNote.classList.toggle("form-note-error", isError);
}

function setBusy(isBusy) {
  dom.submit.disabled = isBusy;
  dom.resultState.setAttribute("aria-busy", String(isBusy));
}

/** Announce the outcome in one short sentence instead of re-reading the whole panel. */
function setResultStatus(message) {
  dom.resultStatus.textContent = message;
}

// The page runs one query on load, but a visitor who just arrived did not ask for it; scrolling
// them into the middle of the page would be a worse first impression than the missing feedback.
let userIntent = false;

/**
 * On a narrow viewport the results sit far below the form, so a submit used to look like nothing
 * happened. Bring the result region into view; desktop and the initial load are left alone.
 */
function revealResults() {
  if (!userIntent || !window.matchMedia("(max-width: 860px)").matches) {
    return;
  }
  dom.resultState.scrollIntoView({ behavior: "smooth", block: "start" });
}

/**
 * The form now describes something other than what the panel shows. Say so instead of leaving
 * the previous response looking like the answer to the current controls.
 */
function markFormChanged() {
  if (!state.lastRender || state.stale) {
    return;
  }
  state.stale = true;
  dom.resultMeta.textContent = `${dom.resultMeta.textContent} · ${t("form.staleSuffix")}`;
}

function updateModeCopy() {
  const mode = currentMode();
  dom.submit.textContent = mode === "ask" ? t("form.modeAsk") : t("form.modeRetrieve");
  dom.resultsTitle.textContent = mode === "ask" ? t("answer.title") : t("form.modeRetrieve");
  if (mode === "ask") {
    setFormNote(t("form.noteAsk"));
  } else {
    setFormNote(t("form.noteRetrieve"));
  }
}

function filtersSummary(filters) {
  if (!filters) {
    return t("filter.none");
  }
  const parts = [];
  if (filters.source_types?.length) {
    parts.push(
      t("filter.sources", {
        list: filters.source_types.map((value) => sourceTypeLabel(value)).join(" / "),
      }),
    );
  }
  if (filters.since || filters.until) {
    parts.push(
      t("filter.dates", {
        since: filters.since || t("filter.unlimited"),
        until: filters.until || t("filter.unlimited"),
      }),
    );
  }
  if (filters.media_ids?.length) {
    parts.push(t("filter.mediaIds", { list: filters.media_ids.join(", ") }));
  }
  return parts.length > 0 ? parts.join(t("filter.join")) : t("filter.none");
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
      ...options,
    });
  } catch (error) {
    throw new ApiError(0, t("error.network", { message: error.message }));
  }

  const raw = await response.text();
  let body = null;
  try {
    body = raw ? JSON.parse(raw) : null;
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : `HTTP ${response.status}`;
    throw new ApiError(response.status, detail, response.headers.get("Retry-After"));
  }
  return body;
}

function buildRequest() {
  const text = dom.question.value.trim();
  if (!text) {
    throw new FormError(t("form.error.empty"));
  }
  if (!state.current) {
    throw new FormError(t("form.error.noSnapshot"));
  }

  const filters = {};
  const sourceTypes = selectedSourceTypes();
  // No checkbox is not "no filter": omitting source_types widens the request to everything, the
  // opposite of the user's "I selected nothing". Rejecting it keeps the form's meaning honest.
  if (sourceTypes.length === 0) {
    throw new FormError(t("form.error.noSource"));
  }
  filters.source_types = sourceTypes;
  if (dom.since.value) {
    filters.since = dom.since.value;
  }
  if (dom.until.value) {
    filters.until = dom.until.value;
  }
  if (dom.since.value && dom.until.value && dom.since.value > dom.until.value) {
    throw new FormError(t("form.error.dateRange"));
  }
  if (dom.mediaId.value.trim()) {
    const mediaId = Number(dom.mediaId.value);
    if (!Number.isInteger(mediaId) || mediaId < 1) {
      throw new FormError(t("form.error.mediaId"));
    }
    filters.media_ids = [mediaId];
  }

  const topK = Number(dom.topK.value) || 5;
  const query = {
    text,
    top_k: topK,
    snapshot_id: state.current,
  };
  if (Object.keys(filters).length > 0) {
    query.filters = filters;
  }

  const mode = currentMode();
  return {
    mode,
    body: query,
    key: cacheKey({
      snapshot_id: query.snapshot_id,
      text: query.text,
      filters: query.filters,
      top_k: query.top_k,
      config: ACTIVE_CONFIG,
    }),
  };
}

function renderSnapshotOptions(inventory) {
  const ids = [...new Set([inventory.current, ...(inventory.retained ?? [])])].filter(Boolean);
  replace(dom.snapshotSelect);
  for (const id of ids) {
    const suffix = id === inventory.current ? t("snapshot.optionCurrent") : t("snapshot.optionRetained");
    dom.snapshotSelect.append(
      element("option", {
        text: `${id} · ${suffix}`,
        attrs: { value: id },
      }),
    );
  }
  dom.snapshotSelect.disabled = false;
  dom.snapshotSelect.value = state.current ?? inventory.current;
}

function syncSnapshotDisplay() {
  dom.snapshotId.textContent = state.current
    ? t("snapshot.bound", { id: state.current })
    : t("snapshot.none");
}

async function loadInventory() {
  const inventory = await requestJson("/snapshots");
  state.inventory = inventory;
  if (!state.current || !(inventory.retained ?? []).includes(state.current)) {
    state.current = inventory.current;
  }
  renderSnapshotOptions(inventory);
  syncSnapshotDisplay();
}

async function runQuery() {
  if (!queryQueue.begin()) {
    // The submit button is disabled while busy, but example buttons, mode radios and retry never
    // are. Dropping them silently left the new question in the box next to the old response;
    // queue the newest intent and run it when the current request settles instead.
    queryQueue.queue(runQuery);
    setFormNote(t("note.queued"));
    return;
  }

  setBusy(true);
  let request = null;
  try {
    try {
      request = buildRequest();
    } catch (error) {
      if (error instanceof FormError) {
        setFormNote(error.message, true);
        return;
      }
      throw error;
    }

    state.lastRequest = request;
    setFormNote(t("note.requesting"));
    const cache = state.cache[request.mode];
    if (cache.has(request.key)) {
      renderResponse(request.mode, cache.get(request.key), request, true);
      setFormNote(t("note.cacheHit"));
      return;
    }
    const path = request.mode === "ask" ? "/ask" : "/retrieve";
    const payload = await requestJson(path, {
      method: "POST",
      body: JSON.stringify(request.body),
    });
    cache.set(request.key, payload);
    renderResponse(request.mode, payload, request, false);
    setFormNote(t("note.done"));
  } catch (error) {
    renderApiError(error, request);
    setFormNote(t("note.failed"), true);
  } finally {
    setBusy(false);
    const next = queryQueue.finish();
    if (next) {
      next();
    }
  }
}

function renderResponse(mode, payload, request, fromCache) {
  const stateName = viewState(mode, payload);
  dom.resultMeta.textContent = [
    mode === "ask" ? t("form.modeAsk") : t("form.modeRetrieve"),
    t("meta.snapshot", { id: request.body.snapshot_id }),
    fromCache ? t("meta.fromCache") : t("meta.fresh"),
  ].join(" · ");
  state.lastRender = true;
  state.stale = false;
  // Kept so a language switch can re-render the same response without refetching.
  state.lastResponse = { mode, payload, request, fromCache };

  if (mode === "retrieve") {
    renderRetrieve(payload, request, stateName);
    revealResults();
    return;
  }
  if (stateName === "ask-abstained") {
    renderRefusal(payload);
    revealResults();
    return;
  }
  renderAnswer(payload, request);
  revealResults();
}

function renderRetrieve(payload, request, stateName) {
  const { items, hiddenCount } = selectRenderableExcerpts(payload);
  if (stateName === "retrieve-empty") {
    renderEmpty(request, t("retrieve.empty"));
    return;
  }
  if (items.length === 0) {
    renderComplianceOnly(hiddenCount);
    return;
  }

  const container = element("div");
  container.append(
    element("p", {
      className: "compliance-note",
      text: t("retrieve.compliance", { n: items.length, max: MAX_QUOTE_CODEPOINTS }),
    }),
  );
  items.forEach((excerpt) => {
    container.append(renderExcerptCard(excerpt));
  });
  if (hiddenCount > 0) {
    container.append(renderHiddenNotice(hiddenCount));
  }
  replace(dom.resultState, container);
  setResultStatus(t("retrieve.shown", { n: items.length }));
}

function renderExcerptCard(excerpt) {
  const citation = excerpt.citation;
  const sourceType = citation.source_type;
  const sourceClass = `source-${sourceType}`;
  const strength = STRENGTH[sourceType] ?? 1;
  const quote = excerpt.quote;
  const quoteLength = excerpt.quoteLength;

  const badge = element(
    "span",
    { className: "source-badge", attrs: { "data-strength": strength } },
    [
      element("span", { text: sourceTypeLabel(sourceType) }),
      element("span", { className: "strength", attrs: { "aria-hidden": "true" } }, [
        element("span"),
        element("span"),
        element("span"),
      ]),
    ],
  );

  const head = element("div", { className: "result-card-head" }, [
    badge,
    element("span", {
      className: "result-publisher",
      text: `${citation.publisher} · ${citation.published_at}`,
    }),
  ]);

  const meta = element("div", { className: "result-meta" }, [
    element("span", { text: `score ${Number(excerpt.score).toFixed(3)}` }),
    element("span", {
      text: t("card.chars", { n: quoteLength, max: MAX_QUOTE_CODEPOINTS }),
    }),
    element("span", { text: citation.chunk_id }),
  ]);
  if (excerpt.wasClipped) {
    meta.append(element("span", { className: "clip-flag", text: t("card.clipped") }));
  }

  return element("article", { className: `result-card ${sourceClass}` }, [
    head,
    element("h3", { text: citation.title }),
    element("p", { className: "quote", text: quote }),
    meta,
    renderCitationUrl(citation),
  ]);
}

function renderCitationUrl(citation) {
  // `isRenderableCitation` already rejected non-http(s) URLs before this card was built, so the
  // anchor can carry the original URL. `noopener` keeps the opened page out of `window.opener`.
  const link = element("a", {
    className: "citation-url-value",
    text: t("citation.open"),
    attrs: {
      href: citation.url,
      target: "_blank",
      rel: "noopener noreferrer",
      title: t("citation.openTitle", { url: citation.url }),
    },
  });
  link.append(
    element("span", { className: "jump-mark", attrs: { "aria-hidden": "true" }, text: " ↗" }),
  );
  return element("div", { className: "citation-url" }, [
    element("span", { className: "citation-url-label", text: t("citation.source") }),
    link,
    element("span", { className: "citation-url-raw", text: citation.url }),
  ]);
}

function renderAnswer(payload, request) {
  const citations = Array.isArray(payload.citations) ? payload.citations : [];
  const visible = citations.filter((citation) => isRenderableCitation(citation));
  const hiddenCount = citations.length - visible.length;

  const container = element("div");
  const meta = element("div", { className: "answer-meta" }, [
    element("span", { text: t("meta.snapshot", { id: request.body.snapshot_id }) }),
    element("span", { text: t("answer.citedCount", { n: payload.cited_chunk_ids?.length ?? 0 }) }),
    element("span", { text: t("answer.displayOnly") }),
  ]);
  container.append(
    element("section", { className: "answer-panel", attrs: { "aria-label": t("answer.title") } }, [
      element("h3", { text: t("answer.title") }),
      element("p", { className: "answer-text", text: payload.answer ?? "" }),
      meta,
      element("p", {
        className: "answer-caveat",
        text: t("answer.caveat"),
      }),
    ]),
  );

  if (visible.length > 0) {
    container.append(
      element("p", {
        className: "compliance-note",
        text: t("answer.compliance", { n: visible.length }),
      }),
    );
    visible.forEach((citation) => {
      container.append(renderCitationCard(citation));
    });
  } else if (citations.length === 0) {
    container.append(
      element("p", {
        className: "compliance-note",
        text: t("answer.noCitations"),
      }),
    );
  } else {
    container.append(renderHiddenNotice(hiddenCount));
  }
  if (hiddenCount > 0 && visible.length > 0) {
    container.append(renderHiddenNotice(hiddenCount));
  }
  replace(dom.resultState, container);
  setResultStatus(
    visible.length > 0
      ? t("answer.shown", { n: visible.length })
      : t("answer.noVisible"),
  );
}

function renderCitationCard(citation) {
  const sourceType = citation.source_type;
  const strength = STRENGTH[sourceType] ?? 1;
  const badge = element(
    "span",
    { className: "source-badge", attrs: { "data-strength": strength } },
    [
      element("span", { text: sourceTypeLabel(sourceType) }),
      element("span", { className: "strength", attrs: { "aria-hidden": "true" } }, [
        element("span"),
        element("span"),
        element("span"),
      ]),
    ],
  );

  return element("article", { className: `result-card source-${sourceType}` }, [
    element("div", { className: "result-card-head" }, [
      badge,
      element("span", {
        className: "result-publisher",
        text: `${citation.publisher} · ${citation.published_at}`,
      }),
    ]),
    element("h3", { text: citation.title }),
    element("div", { className: "result-meta" }, [
      element("span", { text: citation.chunk_id }),
      element("span", { text: t("card.fromEvidence") }),
    ]),
    renderCitationUrl(citation),
  ]);
}

function renderEmpty(request, detail) {
  const filters = request.body.filters;
  const clearFilters = element("button", {
    text: t("empty.action"),
    attrs: { type: "button" },
  });
  clearFilters.addEventListener("click", () => {
    userIntent = true;
    dom.sourceInputs.forEach((input) => {
      input.checked = true;
    });
    dom.since.value = "";
    dom.until.value = "";
    dom.mediaId.value = "";
    runQuery();
  });

  replace(
    dom.resultState,
    element("section", { className: "state-panel empty-panel" }, [
      element("h3", { text: t("empty.title") }),
      element("p", { text: detail }),
      element("p", { text: t("empty.body") }),
      element("p", { className: "state-detail", text: t("empty.filters", { summary: filtersSummary(filters) }) }),
      element("p", { className: "state-detail", text: t("empty.snapshot", { id: request.body.snapshot_id }) }),
      element("div", { className: "state-actions" }, [clearFilters]),
    ]),
  );
  setResultStatus(t("empty.status"));
}

function renderRefusal(payload) {
  const reason =
    typeof payload.abstain_reason === "string" && payload.abstain_reason.trim()
      ? payload.abstain_reason
      : t("refuse.noReason");
  const answer =
    typeof payload.answer === "string" && payload.answer.trim() ? payload.answer : reason;
  // `answer` and `abstain_reason` may legitimately be the same string; printing both reads like a
  // rendering bug. Show the sentence once and keep the reason only when it adds information.
  const reasonAddsDetail = answer !== reason;

  const rows = [
    element("h3", { text: t("refuse.title") }),
    element("p", { text: answer }),
    element("p", { text: t("refuse.body") }),
  ];
  if (reasonAddsDetail) {
    rows.push(element("p", { className: "state-detail", text: t("refuse.reason", { reason }) }));
  }
  rows.push(element("p", { className: "state-detail", text: t("refuse.counts") }));

  replace(dom.resultState, element("section", { className: "state-panel refusal-panel" }, rows));
  setResultStatus(t("refuse.status"));
}

function renderComplianceOnly(hiddenCount) {
  replace(
    dom.resultState,
    element("section", { className: "state-panel empty-panel" }, [
      element("h3", { text: t("hidden.title") }),
      element("p", { text: t("hidden.body", { n: hiddenCount }) }),
      element("p", { text: t("hidden.principle") }),
    ]),
  );
  setResultStatus(t("hidden.status", { n: hiddenCount }));
}

function renderHiddenNotice(count) {
  return element("p", {
    className: "compliance-note",
    text: t("hidden.notice", { n: count }),
  });
}

function renderApiError(error, request = state.lastRequest) {
  const status = error instanceof ApiError ? error.status : 0;
  const detail = error instanceof ApiError ? error.detail : error.message;
  const panel = element("section", { className: "state-panel error-panel" });
  const actions = [];

  if (status === 400) {
    panel.append(
      element("h3", { text: t("err.400.title") }),
      element("p", { text: t("err.400.body") }),
      element("p", { className: "state-detail", text: detail }),
    );
  } else if (status === 404) {
    panel.append(
      element("h3", { text: t("err.404.title") }),
      element("p", { text: t("err.404.body") }),
      element("p", { className: "state-detail", text: detail }),
    );
    actions.push(
      element("button", {
        text: t("action.refresh"),
        attrs: { type: "button", "data-action": "refresh" },
      }),
    );
  } else if (status === 410) {
    panel.append(
      element("h3", { text: t("err.410.title") }),
      element("p", { text: t("err.410.body") }),
      element("p", { className: "state-detail", text: detail }),
    );
    actions.push(
      element("button", {
        text: t("action.refresh"),
        attrs: { type: "button", "data-action": "refresh" },
      }),
      element("button", {
        text: t("action.rerunCurrent"),
        attrs: { type: "button", "data-action": "rerun-current" },
      }),
    );
  } else if (status === 422) {
    panel.append(
      element("h3", { text: t("err.422.title") }),
      element("p", { text: t("err.422.body") }),
      element("p", { className: "state-detail", text: detail }),
    );
    actions.push(
      element("button", {
        text: t("action.clearDates"),
        attrs: { type: "button", "data-action": "clear-dates" },
      }),
    );
  } else if (status === 503) {
    panel.append(
      element("h3", { text: t("err.503.title") }),
      element("p", { text: t("err.503.body") }),
      element("p", { className: "state-detail", text: detail }),
    );
    if (error.retryAfter) {
      panel.append(
        element("p", {
          className: "state-detail",
          text: t("err.retryAfter", { n: error.retryAfter }),
        }),
      );
    }
    actions.push(
      element("button", {
        text: t("action.retry"),
        attrs: { type: "button", "data-action": "retry" },
      }),
    );
  } else {
    panel.append(
      element("h3", { text: t("err.500.title") }),
      element("p", { text: t("err.500.body") }),
      element("p", { className: "state-detail", text: detail || t("err.unknown") }),
    );
    actions.push(
      element("button", {
        text: t("action.retry"),
        attrs: { type: "button", "data-action": "retry" },
      }),
    );
  }

  if (actions.length > 0) {
    const actionBar = element("div", { className: "state-actions" }, actions);
    actionBar.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) {
        return;
      }
      handleAction(button.dataset.action, request);
    });
    panel.append(actionBar);
  }

  if (request) {
    panel.append(
      element("p", {
        className: "state-detail",
        text: t("err.requestDetail", {
          id: request.body.snapshot_id,
          filters: filtersSummary(request.body.filters),
        }),
      }),
    );
  }
  dom.resultMeta.textContent = t("err.meta", { status: status || t("err.networkWord") });
  replace(dom.resultState, panel);
  setResultStatus(t("err.status", { status: status || t("err.networkWord") }));
  revealResults();
}

async function handleAction(action, request) {
  userIntent = true;
  if (action === "refresh") {
    try {
      await loadInventory();
      replace(
        dom.resultState,
        element("section", { className: "state-panel placeholder-panel" }, [
          element("h3", { text: t("refreshed.title") }),
          element("p", { text: t("refreshed.body") }),
          element("p", {
            className: "state-detail",
            text: t("refreshed.current", { id: state.current ?? "—" }),
          }),
        ]),
      );
      setFormNote(t("note.snapshotRefreshed"));
    } catch (error) {
      renderApiError(error);
    }
    return;
  }
  if (action === "rerun-current" && state.inventory?.current) {
    state.current = state.inventory.current;
    dom.snapshotSelect.value = state.current;
    syncSnapshotDisplay();
    await runQuery();
    return;
  }
  if (action === "clear-dates") {
    dom.since.value = "";
    dom.until.value = "";
    await runQuery();
    return;
  }
  if (action === "retry") {
    await runQuery();
    return;
  }
  if (request) {
    await runQuery();
  }
}

function updateQuestionCount() {
  const used = [...dom.question.value].length;
  const limit = Number(dom.question.maxLength) || 500;
  dom.questionCount.textContent = `${used} / ${limit}`;
}

/**
 * Re-render everything the current language owns: static chrome, the document metadata and the
 * last response (re-rendered from the stored payload — never refetched).
 */
function reapplyLanguage() {
  applyStatic();
  applyDocumentI18n();
  syncToggle();
  updateModeCopy();
  updateQuestionCount();
  if (state.inventory) {
    renderSnapshotOptions(state.inventory);
  }
  syncSnapshotDisplay();
  const last = state.lastResponse;
  if (last) {
    renderResponse(last.mode, last.payload, last.request, last.fromCache);
    return;
  }
  dom.resultMeta.textContent = t("results.metaWaiting");
  replace(
    dom.resultState,
    element("div", { className: "state-panel placeholder-panel" }, [
      element("p", { text: t("results.placeholder") }),
    ]),
  );
}

function bindEvents() {
  bindLangToggle();
  document.addEventListener("acgn:langchange", reapplyLanguage);
  updateQuestionCount();
  dom.question.addEventListener("input", updateQuestionCount);
  // The contract notes are context, not something the query needs; on a phone they pushed the
  // form's submit button off the first screen.
  if (window.matchMedia("(max-width: 860px)").matches) {
    dom.contractNotes.open = false;
  }
  dom.form.addEventListener("submit", (event) => {
    event.preventDefault();
    userIntent = true;
    runQuery();
  });
  dom.form.addEventListener("input", markFormChanged);
  dom.form.addEventListener("change", markFormChanged);
  dom.modeInputs.forEach((input) => {
    input.addEventListener("change", updateModeCopy);
  });
  dom.snapshotSelect.addEventListener("change", () => {
    state.current = dom.snapshotSelect.value;
    syncSnapshotDisplay();
    markFormChanged();
    setFormNote(t("note.snapshotChanged"));
  });
  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      userIntent = true;
      dom.question.value = button.dataset.example ?? "";
      dom.question.focus();
      runQuery();
    });
  });
}

async function init() {
  bindEvents();
  applyStatic();
  seedContentValue();
  applyDocumentI18n();
  updateModeCopy();
  try {
    await loadInventory();
    await runQuery();
  } catch (error) {
    renderApiError(error);
    setFormNote(t("init.failed"), true);
  }
}

init();
