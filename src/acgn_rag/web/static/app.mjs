import {
  MAX_QUOTE_CODEPOINTS,
  cacheKey,
  isRenderableCitation,
  selectRenderableExcerpts,
  viewState,
} from "./contract.mjs";
import { getLang, setLang, sourceLabel, t } from "./i18n.mjs";
import { createQueryQueue } from "./queue.mjs";

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
  questionCount: document.querySelector("#question-count"),
  submit: document.querySelector("#submit-button"),
  formNote: document.querySelector("#form-note"),
  snapshotSelect: document.querySelector("#snapshot-select"),
  snapshotId: document.querySelector("#snapshot-id"),
  resultState: document.querySelector("#result-state"),
  resultStatus: document.querySelector("#result-status"),
  resultMeta: document.querySelector("#result-meta"),
  resultsTitle: document.querySelector("#results-title"),
  contractNotes: document.querySelector("#contract"),
  since: document.querySelector("#since"),
  until: document.querySelector("#until"),
  mediaId: document.querySelector("#media-id"),
  topK: document.querySelector("#top-k"),
  modeInputs: document.querySelectorAll('input[name="mode"]'),
  sourceInputs: document.querySelectorAll('input[name="source_type"]'),
  langButtons: document.querySelectorAll("[data-lang]"),
  statsMount: document.querySelector("#stats-mount"),
  overviewStrip: document.querySelector("#overview-strip"),
  scrollProgress: document.querySelector("#scroll-progress"),
  detailDialog: document.querySelector("#detail-dialog"),
  detailBody: document.querySelector("#detail-body"),
  detailClose: document.querySelector("#detail-close"),
};

const state = {
  inventory: null,
  current: null,
  lastRequest: null,
  lastRender: null,
  stale: false,
  // Repaints the last outcome from memory; a language switch reuses it instead of refetching.
  lastPaint: null,
  // Presentation-only state. `view` picks which of the four readers paints the same response;
  // `viewData` holds the last renderable dataset so switching views never touches the network.
  view: "cards",
  viewData: null,
  viewSeries: 0,
  // One cache per endpoint: the contract cache tuple does not include the route, so sharing one
  // map between /retrieve and /ask would mix two response shapes under the same key.
  cache: { retrieve: new Map(), ask: new Map() },
};

const detailStore = new WeakMap();
let lastFocusedCard = null;

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
  if (dom.resultStatus) {
    dom.resultStatus.textContent = message;
  }
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
  dom.resultMeta.textContent = `${dom.resultMeta.textContent} · ${t("meta.formChanged")}`;
}

function updateModeCopy() {
  const mode = currentMode();
  dom.submit.textContent = mode === "ask" ? t("form.submit.ask") : t("form.submit.retrieve");
  dom.resultsTitle.textContent =
    mode === "ask" ? t("results.title.ask") : t("results.title.retrieve");
  setFormNote(mode === "ask" ? t("form.note.ask") : t("form.note.retrieve"));
}

function filtersSummary(filters) {
  if (!filters) {
    return t("filters.none");
  }
  const parts = [];
  if (filters.source_types?.length) {
    parts.push(
      t("filters.sources", {
        list: filters.source_types.map((value) => sourceLabel(value)).join(" / "),
      }),
    );
  }
  if (filters.since || filters.until) {
    parts.push(
      t("filters.dates", {
        since: filters.since || t("filters.any"),
        until: filters.until || t("filters.any"),
      }),
    );
  }
  if (filters.media_ids?.length) {
    parts.push(t("filters.media", { ids: filters.media_ids.join(", ") }));
  }
  return parts.length > 0 ? parts.join(t("list.sep")) : t("filters.none");
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
      ...options,
    });
  } catch (error) {
    throw new ApiError(0, t("error.connect", { message: error.message }));
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
    throw new FormError(t("err.form.empty"));
  }
  if (!state.current) {
    throw new FormError(t("err.form.noSnapshot"));
  }

  const filters = {};
  const sourceTypes = selectedSourceTypes();
  // No checkbox is not "no filter": omitting source_types widens the request to everything, the
  // opposite of the user's "I selected nothing". Rejecting it keeps the form's meaning honest.
  if (sourceTypes.length === 0) {
    throw new FormError(t("err.form.sources"));
  }
  filters.source_types = sourceTypes;
  if (dom.since.value) {
    filters.since = dom.since.value;
  }
  if (dom.until.value) {
    filters.until = dom.until.value;
  }
  if (dom.since.value && dom.until.value && dom.since.value > dom.until.value) {
    throw new FormError(t("err.form.dateOrder"));
  }
  if (dom.mediaId.value.trim()) {
    const mediaId = Number(dom.mediaId.value);
    if (!Number.isInteger(mediaId) || mediaId < 1) {
      throw new FormError(t("err.form.mediaId"));
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
    const suffix = id === inventory.current ? t("snapshot.current") : t("snapshot.retained");
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
    setStatsLoading();
    setFormNote(t("note.requesting"));
    const cache = state.cache[request.mode];
    if (cache.has(request.key)) {
      renderResponse(request.mode, cache.get(request.key), request, true);
      setFormNote(t("note.cache"));
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

/* ------------------------------------------------------------------ */
/* Quantified profile (derived only from the current response)         */
/* ------------------------------------------------------------------ */

const SOURCE_ORDER = ["official", "media", "metadata"];

function formatScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? score.toFixed(3) : "—";
}

function dayValue(value) {
  const parsed = new Date(`${value}T00:00:00`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function shortId(value) {
  const text = String(value ?? "");
  if (text.length <= 16) {
    return text || "—";
  }
  return `${text.slice(0, 9)}…${text.slice(-4)}`;
}

function percentage(part, total) {
  return total > 0 ? Math.round((part / total) * 100) : 0;
}

function metricBlock(label, value, hint) {
  const children = [element("dt", { text: label }), element("dd", { text: value })];
  if (hint) {
    children.push(element("small", { text: hint }));
  }
  return element("div", { className: "metric" }, children);
}

function normaliseRows(mode, items) {
  return items.map((item) =>
    mode === "retrieve"
      ? {
          citation: item.citation,
          score: Number(item.score),
          quote: item.quote,
          quoteLength: item.quoteLength,
          wasClipped: item.wasClipped,
        }
      : { citation: item, score: null, quote: null, quoteLength: null, wasClipped: false },
  );
}

/** Shared source chip with the three-bar strength meter, reused by every view and the drawer. */
function sourceBadge(sourceType) {
  const strength = STRENGTH[sourceType] ?? 1;
  return element(
    "span",
    { className: "source-badge", attrs: { "data-strength": strength } },
    [
      element("span", { text: sourceLabel(sourceType) }),
      element("span", { className: "strength", attrs: { "aria-hidden": "true" } }, [
        element("span"),
        element("span"),
        element("span"),
      ]),
    ],
  );
}

function statCard(title, countLabel, children) {
  return element("section", { className: "stat-card" }, [
    element("div", { className: "stat-card-head" }, [
      element("h4", { text: title }),
      element("span", { className: "stat-count", text: countLabel }),
    ]),
    ...children,
  ]);
}

/* Carousel / CountUp / capsule tabs ported from Animal Island UI
 * (Carousel, CountUp, Tag, Progress) to keep the profile on the same
 * component language as the rest of the page. */

function statTabs() {
  return [
    t("stats.tab.source"),
    t("stats.tab.metrics"),
    t("stats.tab.distribution"),
    t("stats.tab.time"),
  ];
}

let statCarouselTimer = null;

function easeOutCubic(progress) {
  return 1 - (1 - progress) ** 3;
}

function countUpNode(node) {
  const original = node.textContent ?? "";
  const match = original.match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return;
  }
  const target = Number(match[0]);
  if (!Number.isFinite(target) || target === 0) {
    return;
  }
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return;
  }
  const decimals = (match[0].split(".")[1] ?? "").length;
  const prefix = original.slice(0, match.index);
  const suffix = original.slice(match.index + match[0].length);
  const started = performance.now();
  const tick = (now) => {
    const progress = Math.min(1, (now - started) / 820);
    if (progress >= 1) {
      node.textContent = original;
      return;
    }
    node.textContent = `${prefix}${(target * easeOutCubic(progress)).toFixed(decimals)}${suffix}`;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function buildStatCarousel(cards) {
  clearInterval(statCarouselTimer);
  statCarouselTimer = null;

  const tabsCopy = statTabs();
  let current = 0;

  const slides = cards.map((card, index) =>
    element(
      "div",
      {
        className: `stat-slide${index === 0 ? " is-active" : ""}`,
        attrs: {
          role: "tabpanel",
          "aria-label": t("stats.panelAria", {
            tab: tabsCopy[index] ?? t("stats.tabFallback", { n: index + 1 }),
            index: index + 1,
            total: cards.length,
          }),
          "aria-hidden": String(index !== 0),
        },
      },
      [card],
    ),
  );

  const tabButtons = cards.map((_, index) =>
    element("button", {
      className: `stat-tab${index === 0 ? " is-active" : ""}`,
      text: tabsCopy[index] ?? t("stats.tabFallback", { n: index + 1 }),
      attrs: {
        type: "button",
        role: "tab",
        "aria-selected": String(index === 0),
        tabindex: index === 0 ? "0" : "-1",
      },
    }),
  );

  const goTo = (next) => {
    current = (next + cards.length) % cards.length;
    slides.forEach((slide, index) => {
      const active = index === current;
      slide.classList.toggle("is-active", active);
      slide.setAttribute("aria-hidden", String(!active));
    });
    tabButtons.forEach((tab, index) => {
      const active = index === current;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
  };

  tabButtons.forEach((tab, index) => {
    tab.addEventListener("click", () => goTo(index));
  });

  const previous = element("button", {
    className: "stat-arrow previous",
    attrs: { type: "button", "aria-label": t("stats.prev") },
  });
  const next = element("button", {
    className: "stat-arrow next",
    attrs: { type: "button", "aria-label": t("stats.next") },
  });
  previous.addEventListener("click", () => goTo(current - 1));
  next.addEventListener("click", () => goTo(current + 1));

  const tabs = element(
    "div",
    { className: "stat-tabs", attrs: { role: "tablist", "aria-label": t("stats.group") } },
    tabButtons,
  );
  tabs.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }
    event.preventDefault();
    goTo(current + (event.key === "ArrowRight" ? 1 : -1));
    tabButtons[current].focus();
  });

  const root = element("div", { className: "stat-carousel" }, [
    element("div", { className: "stat-carousel-top" }, [
      tabs,
      element("div", { className: "stat-arrows" }, [previous, next]),
    ]),
    element("div", { className: "stat-viewport" }, slides),
  ]);

  slides.forEach((slide) => {
    slide
      .querySelectorAll(".metric dd, .source-legend strong, .timeline-row strong")
      .forEach((node) => countUpNode(node));
  });

  if (cards.length > 1 && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const start = () => {
      clearInterval(statCarouselTimer);
      statCarouselTimer = setInterval(() => goTo(current + 1), 7000);
    };
    const stop = () => clearInterval(statCarouselTimer);
    root.addEventListener("pointerenter", stop);
    root.addEventListener("pointerleave", start);
    root.addEventListener("focusin", stop);
    root.addEventListener("focusout", start);
    start();
  }

  return root;
}

function resetStats(message) {
  if (!dom.statsMount) {
    return;
  }
  clearInterval(statCarouselTimer);
  statCarouselTimer = null;
  replace(
    dom.statsMount,
    element("div", { className: "stat-empty" }, [
      element("span", { className: "stat-empty-mark", attrs: { "aria-hidden": "true" }, text: "◇" }),
      element("p", { text: message }),
    ]),
  );
}

function setStatsLoading() {
  if (!dom.statsMount) {
    return;
  }
  clearInterval(statCarouselTimer);
  statCarouselTimer = null;
  replace(
    dom.statsMount,
    element("div", { className: "stat-empty is-loading" }, [
      element("span", { className: "stat-empty-mark", attrs: { "aria-hidden": "true" }, text: "◌" }),
      element("p", { text: t("stats.loading") }),
    ]),
  );
}

/**
 * The single place that turns response rows into numbers. The profile carousel, the overview
 * strip and the board view all read from here, so a figure never differs between them.
 */
function summariseRows(mode, rows) {
  const total = rows.length;
  const sourceRows = SOURCE_ORDER.map((type) => {
    const group = rows.filter((row) => row.citation.source_type === type);
    const scored = group.map((row) => row.score).filter(Number.isFinite);
    return {
      type,
      count: group.length,
      share: percentage(group.length, total),
      average: scored.length
        ? scored.reduce((sum, value) => sum + value, 0) / scored.length
        : null,
    };
  }).filter((row) => row.count > 0);

  const scoreValues = rows.map((row) => row.score).filter(Number.isFinite);
  const averageScore = scoreValues.length
    ? scoreValues.reduce((sum, value) => sum + value, 0) / scoreValues.length
    : null;
  const maxScore = scoreValues.length ? Math.max(...scoreValues) : null;
  const dates = rows
    .map((row) => dayValue(row.citation.published_at))
    .filter(Boolean)
    .sort((left, right) => left - right);
  const spanDays =
    dates.length > 1
      ? Math.max(0, Math.round((dates[dates.length - 1] - dates[0]) / 86400000))
      : 0;
  const averageLength = mode === "retrieve"
    ? Math.round(
        rows.reduce((sum, row) => sum + (Number(row.quoteLength) || 0), 0) / total,
      )
    : 0;
  const publishers = new Set(rows.map((row) => row.citation.publisher)).size;
  return {
    total,
    sourceRows,
    scoreValues,
    averageScore,
    maxScore,
    dates,
    spanDays,
    averageLength,
    publishers,
  };
}

function renderStats(mode, items, request, payload) {
  if (!dom.statsMount) {
    return;
  }
  const rows = normaliseRows(mode, items);
  if (rows.length === 0) {
    resetStats(t("stats.noCitations"));
    return;
  }

  const {
    total,
    sourceRows,
    scoreValues,
    averageScore,
    maxScore,
    dates,
    spanDays,
    averageLength,
    publishers,
  } = summariseRows(mode, rows);

  const flowSegments = sourceRows.map((row, index) =>
    element("span", {
      className: `source-seg source-${row.type}`,
      attrs: {
        style: `--w:${row.share}%;--i:${index}`,
        title: `${sourceLabel(row.type)} ${row.share}%`,
      },
    }),
  );
  const legendRows = sourceRows.map((row) =>
    element("li", {}, [
      element("span", { className: `legend-dot source-${row.type}` }),
      element("span", { className: "legend-name", text: sourceLabel(row.type) }),
      element("strong", { text: String(row.count) }),
      element("em", { text: `${row.share}%` }),
      element("small", {
        text:
          row.average === null
            ? t("stats.idleSource")
            : t("stats.avgScore", { score: formatScore(row.average) }),
      }),
    ]),
  );

  const sourceCard = statCard(t("stats.sourcesTitle"), t("stats.countLabel", { n: total }), [
    element("div", {
      className: "source-flow",
      attrs: {
        role: "img",
        "aria-label": sourceRows
          .map((row) =>
            t("stats.sourceShare", {
              label: sourceLabel(row.type),
              n: row.count,
              share: row.share,
            }),
          )
          .join(t("list.sep")),
      },
    }, flowSegments),
    element("ul", { className: "source-legend" }, legendRows),
  ]);

  const metricValues = mode === "retrieve"
    ? [
        metricBlock(
          t("stats.avg"),
          averageScore === null ? "—" : formatScore(averageScore),
          t("stats.avgHint"),
        ),
        metricBlock(
          t("stats.max"),
          maxScore === null ? "—" : formatScore(maxScore),
          t("stats.maxHint"),
        ),
        metricBlock(
          t("stats.span"),
          spanDays > 0 ? t("stats.days", { n: spanDays }) : t("stats.singleDay"),
          t("stats.dateCount", { n: dates.length }),
        ),
        metricBlock(
          t("stats.avgChars"),
          t("stats.chars", { n: averageLength }),
          t("stats.limit", { n: MAX_QUOTE_CODEPOINTS }),
        ),
      ]
    : [
        metricBlock(t("stats.citations"), t("stats.countLabel", { n: total }), t("overview.passed")),
        metricBlock(
          t("stats.publishers"),
          t("stats.publisherCount", { n: publishers }),
          t("stats.dedup"),
        ),
        metricBlock(
          t("stats.span"),
          spanDays > 0 ? t("stats.days", { n: spanDays }) : t("stats.singleDay"),
          t("stats.dateCount", { n: dates.length }),
        ),
        metricBlock(
          t("stats.answerChars"),
          t("stats.chars", { n: (payload.answer ?? "").length }),
          t("stats.displayOnly"),
        ),
      ];
  const metricCard = statCard(
    mode === "retrieve" ? t("stats.metricsTitle.retrieve") : t("stats.metricsTitle.ask"),
    mode === "retrieve" ? t("stats.metricsCount.retrieve") : t("stats.metricsCount.ask"),
    [element("dl", { className: "metric-grid" }, metricValues)],
  );

  const maxBar = mode === "retrieve"
    ? Math.max(...scoreValues, 0.0001)
    : 3;
  const meterRows = rows.map((row, index) => {
    const value = mode === "retrieve"
      ? row.score
      : STRENGTH[row.citation.source_type] ?? 1;
    const width = Math.max(6, Math.round((value / maxBar) * 100));
    return element("li", { className: "stat-meter" }, [
      element("span", { className: "meter-label", text: shortId(row.citation.chunk_id) }),
      element("span", { className: "meter-track" }, [
        element("span", {
          className: "meter-fill",
          attrs: { style: `--w:${width}%;--i:${index}` },
        }),
      ]),
      element("span", {
        className: "meter-value",
        text: mode === "retrieve" ? formatScore(value) : `${value}/3`,
      }),
    ]);
  });
  const meterCard = statCard(
    mode === "retrieve" ? t("stats.meterTitle.retrieve") : t("stats.meterTitle.ask"),
    mode === "retrieve" ? t("stats.meterCount.retrieve") : t("stats.meterCount.ask"),
    [element("ol", { className: "meter-list" }, meterRows)],
  );

  const monthCounts = new Map();
  for (const date of dates) {
    const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
    monthCounts.set(key, (monthCounts.get(key) ?? 0) + 1);
  }
  const monthKeys = [...monthCounts.keys()].sort();
  const maxMonth = Math.max(1, ...monthCounts.values());
  const timelineRows = monthKeys.map((key, index) => {
    const count = monthCounts.get(key);
    return element("li", { className: "timeline-row" }, [
      element("time", { text: key }),
      element("span", { className: "timeline-track" }, [
        element("span", {
          className: "timeline-fill",
          attrs: { style: `--w:${Math.max(8, Math.round((count / maxMonth) * 100))}%;--i:${index}` },
        }),
      ]),
      element("strong", { text: String(count) }),
    ]);
  });
  const timelineCard = statCard(
    t("stats.published"),
    monthKeys.length ? t("stats.months", { n: monthKeys.length }) : t("stats.noDates"),
    [
      monthKeys.length
        ? element("ol", { className: "timeline-list" }, timelineRows)
        : element("p", { className: "stat-note", text: t("stats.noDatesNote") }),
    ],
  );

  replace(dom.statsMount, buildStatCarousel([sourceCard, metricCard, meterCard, timelineCard]));
}

/* ------------------------------------------------------------------ */
/* Detail drawer                                                       */
/* ------------------------------------------------------------------ */

function detailSection(title, ...children) {
  return element("section", { className: "detail-section" }, [
    element("h4", { text: title }),
    ...children,
  ]);
}

function attachCardDetail(card, detail, index, cta = t("detail.cta.cards")) {
  // `detail-item` is the delegation hook every view shares; `.result-card` stays the cards-view
  // skin so the photo-background rules keep applying only where they were designed to.
  card.classList.add("is-interactive", "detail-item");
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.setAttribute("aria-label", t("detail.aria", { title: detail.citation.title }));
  card.style.setProperty("--i", String(index));
  if (cta) {
    card.append(
      element("span", {
        className: "card-cta",
        attrs: { "aria-hidden": "true" },
        text: cta,
      }),
    );
  }
  detailStore.set(card, detail);
}

function detailField(label, value, meterPercent = null) {
  const tile = element("div", { className: "detail-field" }, [
    element("span", { className: "detail-field-label", text: label }),
    element("strong", { className: "detail-field-value", text: value ?? "—" }),
  ]);
  if (Number.isFinite(meterPercent)) {
    tile.append(
      element("span", { className: "detail-meter", attrs: { "aria-hidden": "true" } }, [
        element("span", { attrs: { style: `--w:${Math.round(Math.max(0, Math.min(100, meterPercent)))}%` } }),
      ]),
    );
  }
  return tile;
}

function openDetail(detail) {
  if (!dom.detailDialog || typeof dom.detailDialog.showModal !== "function") {
    return;
  }
  const { citation } = detail;
  const sourceType = citation.source_type;
  const children = [];

  const hero = element("div", { className: `detail-hero source-${sourceType}` }, [
    element("div", { className: "detail-hero-top" }, [
      element("span", { className: `detail-badge source-${sourceType}` }, [
        element("span", { text: sourceLabel(sourceType) }),
      ]),
      Number.isFinite(detail.score)
        ? element("span", { className: "score-chip on-dark", text: `score ${formatScore(detail.score)}` })
        : null,
    ].filter(Boolean)),
    element("h3", { text: citation.title, attrs: { id: "detail-title" } }),
    element("p", {
      className: "detail-publisher",
      text: `${citation.publisher ?? "—"} · ${citation.published_at ?? "—"}`,
    }),
  ]);
  // Photo banner cycles the four card backgrounds by position within the batch.
  hero.style.setProperty(
    "--hero-photo",
    `url("/static/assets/cards/bg-0${(detail.index % 4) + 1}.jpg")`,
  );
  children.push(hero);

  if (detail.quote) {
    children.push(element("blockquote", { className: "detail-quote", text: detail.quote }));
    if (detail.wasClipped) {
      children.push(
        element("p", {
          className: "detail-note",
          text: t("detail.clipped", { limit: MAX_QUOTE_CODEPOINTS }),
        }),
      );
    }
  } else if (detail.kind === "citation") {
    children.push(
      element("p", {
        className: "detail-note",
        text: t("detail.citationOnly"),
      }),
    );
  }

  children.push(
    detailSection(
      t("detail.fields"),
      element("div", { className: "detail-fields" }, [
        detailField("chunk_id", citation.chunk_id ?? "—"),
        detailField(t("detail.sourceType"), sourceLabel(sourceType)),
        detailField(t("detail.publisher"), citation.publisher ?? "—"),
        detailField(t("detail.publishedAt"), citation.published_at ?? "—"),
        detailField(
          t("detail.score"),
          Number.isFinite(detail.score) ? formatScore(detail.score) : "—",
          Number.isFinite(detail.score) ? detail.score * 100 : null,
        ),
        detailField(
          t("detail.quoteChars"),
          Number.isFinite(detail.quoteLength)
            ? `${detail.quoteLength} / ${MAX_QUOTE_CODEPOINTS}`
            : "—",
          Number.isFinite(detail.quoteLength)
            ? (detail.quoteLength / MAX_QUOTE_CODEPOINTS) * 100
            : null,
        ),
        detailField(t("detail.snapshot"), detail.request?.body?.snapshot_id ?? "—"),
      ]),
    ),
  );

  if (detail.request) {
    const chips = [
      [t("detail.question"), detail.request.body.text],
      [t("detail.mode"), detail.request.mode === "ask" ? t("meta.mode.ask") : t("meta.mode.retrieve")],
      ["top_k", detail.request.body.top_k],
      [t("detail.filters"), filtersSummary(detail.request.body.filters)],
      [t("detail.config"), ACTIVE_CONFIG],
      [t("detail.cacheKey"), shortId(detail.request.key)],
    ].map(([label, value]) =>
      element("span", { className: "detail-chip" }, [
        element("em", { text: label }),
        element("span", { text: String(value ?? "—") }),
      ]),
    );
    children.push(detailSection(t("detail.context"), element("div", { className: "detail-chips" }, chips)));
  }

  children.push(
    detailSection(
      t("detail.links"),
      element("div", { className: "detail-link-surface" }, [renderCitationUrl(citation)]),
      element("p", {
        className: "detail-note",
        text: t("detail.linksNote"),
      }),
    ),
  );

  const peers = (detail.peers ?? []).filter((peer) => peer !== detail);
  if (peers.length > 0) {
    const related = peers.map((peer) => {
      const chip = element(
        "button",
        {
          className: `detail-related-item source-${peer.citation.source_type}`,
          attrs: { type: "button" },
        },
        [
          element("span", { className: "detail-related-badge", text: sourceLabel(peer.citation.source_type) }),
          element("span", { className: "detail-related-title", text: peer.citation.title }),
        ],
      );
      // Swapping the drawer body keeps the dialog open; focus restore still points at the
      // original card because `lastFocusedCard` is only captured on the first show.
      chip.addEventListener("click", () => openDetail(peer));
      return chip;
    });
    children.push(detailSection(t("detail.peers"), element("div", { className: "detail-related" }, related)));
  }

  replace(dom.detailBody, ...children);
  if (!dom.detailDialog.open) {
    lastFocusedCard = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dom.detailDialog.showModal();
  }
  dom.detailBody.scrollTop = 0;
  dom.detailClose.focus();
}

function closeDetail() {
  if (dom.detailDialog?.open) {
    dom.detailDialog.close();
  }
}

function setResultMeta(mode, request, fromCache) {
  dom.resultMeta.textContent = [
    mode === "ask" ? t("meta.mode.ask") : t("meta.mode.retrieve"),
    t("answer.snapshot", { id: request.body.snapshot_id }),
    fromCache ? t("meta.cached") : t("meta.thisResponse"),
  ].join(" · ");
}

function renderResponse(mode, payload, request, fromCache) {
  state.lastRequest = request;
  const stateName = viewState(mode, payload);
  setResultMeta(mode, request, fromCache);
  state.lastRender = true;
  state.stale = false;

  if (mode === "retrieve") {
    renderRetrieve(payload, request, stateName, fromCache);
    return;
  }
  if (stateName === "ask-abstained") {
    state.viewData = null;
    renderRefusal(payload);
    return;
  }
  renderAnswer(payload, request, fromCache);
}

function renderRetrieve(payload, request, stateName, fromCache) {
  const { items, hiddenCount } = selectRenderableExcerpts(payload);
  if (stateName === "retrieve-empty") {
    state.viewData = null;
    renderEmpty(request);
    return;
  }
  if (items.length === 0) {
    state.viewData = null;
    renderComplianceOnly(hiddenCount);
    return;
  }

  renderStats("retrieve", items, request, payload);
  state.viewSeries += 1;
  state.viewData = {
    mode: "retrieve",
    rows: normaliseRows("retrieve", items),
    request,
    payload,
    hiddenCount,
    fromCache,
  };
  state.viewData.details = buildDetails(state.viewData);
  renderOverview(state.viewData);
  paintResults();
  state.lastPaint = () => {
    setResultMeta("retrieve", request, fromCache);
    renderStats("retrieve", items, request, payload);
    renderOverview(state.viewData);
    paintResults();
  };
  revealResults();
  setResultStatus(
    hiddenCount > 0
      ? `${t("retrieve.compliance", { n: items.length, limit: MAX_QUOTE_CODEPOINTS })} ${t(
          "compliance.hidden",
          { n: hiddenCount },
        )}`
      : t("retrieve.compliance", { n: items.length, limit: MAX_QUOTE_CODEPOINTS }),
  );
}

function renderExcerptCard(row, detail, index) {
  const citation = row.citation;
  const sourceType = citation.source_type;
  const sourceClass = `source-${sourceType}`;
  const quote = row.quote;
  const quoteLength = row.quoteLength;

  const head = element("div", { className: "result-card-head" }, [
    sourceBadge(sourceType),
    element("span", {
      className: "result-publisher",
      text: `${citation.publisher} · ${citation.published_at}`,
    }),
  ]);

  const meta = element("div", { className: "result-meta" }, [
    element("span", { text: `score ${Number(row.score).toFixed(3)}` }),
    element("span", {
      text: t("retrieve.chars", { n: quoteLength, limit: MAX_QUOTE_CODEPOINTS }),
    }),
    element("span", { text: citation.chunk_id }),
  ]);
  if (row.wasClipped) {
    meta.append(element("span", { className: "clip-flag", text: t("retrieve.clipped") }));
  }

  const card = element("article", { className: `result-card ${sourceClass}` }, [
    head,
    element("h3", { text: citation.title }),
    element("p", { className: "quote", text: quote }),
    meta,
    renderCitationUrl(citation),
  ]);
  attachCardDetail(card, detail, index);
  return card;
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

function buildAnswerPanel(payload, request) {
  const meta = element("div", { className: "answer-meta" }, [
    element("span", { text: t("answer.snapshot", { id: request.body.snapshot_id }) }),
    element("span", {
      text: t("answer.cited", { n: payload.cited_chunk_ids?.length ?? 0 }),
    }),
    element("span", { text: t("answer.displayOnly") }),
  ]);
  return element("section", { className: "answer-panel", attrs: { "aria-label": t("answer.title") } }, [
    element("h3", { text: t("answer.title") }),
    element("p", { className: "answer-text", text: payload.answer ?? "" }),
    meta,
    element("p", { className: "answer-caveat", text: t("answer.caveat") }),
  ]);
}

function renderAnswer(payload, request, fromCache) {
  const citations = Array.isArray(payload.citations) ? payload.citations : [];
  const visible = citations.filter((citation) => isRenderableCitation(citation));
  const hiddenCount = citations.length - visible.length;
  renderStats("ask", visible, request, payload);

  state.viewSeries += 1;
  state.viewData = {
    mode: "ask",
    rows: normaliseRows("ask", visible),
    request,
    payload,
    hiddenCount,
    fromCache,
  };
  state.viewData.details = buildDetails(state.viewData);
  renderOverview(state.viewData);
  paintResults();
  state.lastPaint = () => {
    setResultMeta("ask", request, fromCache);
    renderStats("ask", visible, request, payload);
    renderOverview(state.viewData);
    paintResults();
  };
  revealResults();
  if (visible.length > 0) {
    setResultStatus(t("answer.compliance", { n: visible.length }));
  } else if (citations.length === 0) {
    setResultStatus(t("answer.noCitations"));
  } else {
    setResultStatus(t("compliance.hidden", { n: hiddenCount }));
  }
}

function renderCitationCard(citation, detail, index) {
  const sourceType = citation.source_type;
  const card = element("article", { className: `result-card source-${sourceType}` }, [
    element("div", { className: "result-card-head" }, [
      sourceBadge(sourceType),
      element("span", {
        className: "result-publisher",
        text: `${citation.publisher} · ${citation.published_at}`,
      }),
    ]),
    element("h3", { text: citation.title }),
    element("div", { className: "result-meta" }, [
      element("span", { text: citation.chunk_id }),
      element("span", { text: t("answer.citationMeta") }),
    ]),
    renderCitationUrl(citation),
  ]);
  attachCardDetail(card, detail, index);
  return card;
}

function renderEmpty(request) {
  resetStats(t("stats.noExcerpts"));
  setOverviewHint(t("overview.empty"));
  const filters = request.body.filters;
  const clearFilters = element("button", { text: t("empty.clear"), attrs: { type: "button" } });
  clearFilters.addEventListener("click", () => {
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
      element("p", { text: t("retrieve.empty") }),
      element("p", { text: t("empty.explainer") }),
      element("p", {
        className: "state-detail",
        text: t("empty.filters", { summary: filtersSummary(filters) }),
      }),
      element("p", { className: "state-detail", text: t("empty.snapshot", { id: request.body.snapshot_id }) }),
      element("div", { className: "state-actions" }, [clearFilters]),
    ]),
  );
  state.lastPaint = () => renderEmpty(request);
  revealResults();
  setResultStatus(t("empty.title"));
}

function renderRefusal(payload) {
  resetStats(t("stats.noAnswerSources"));
  setOverviewHint(t("overview.refusal"));
  const rawReason =
    typeof payload.abstain_reason === "string" ? payload.abstain_reason.trim() : "";
  const reason = rawReason || t("refusal.missingReason");
  const answer =
    typeof payload.answer === "string" && payload.answer.trim() ? payload.answer : reason;
  // B's review: with a bare reason the body repeated the reason line word for word. Only keep
  // the body when it actually adds detail beyond the reason chip.
  const reasonAddsDetail =
    answer !== reason && answer !== rawReason;

  replace(
    dom.resultState,
    element("section", { className: "state-panel refusal-panel" }, [
      element("h3", { text: t("refusal.title") }),
      ...(reasonAddsDetail ? [element("p", { text: answer })] : []),
      element("p", { text: t("refusal.explainer") }),
      element("p", { className: "state-detail", text: t("refusal.reason", { reason }) }),
      element("p", { className: "state-detail", text: t("refusal.counts") }),
    ]),
  );
  state.lastPaint = () => renderRefusal(payload);
  revealResults();
  setResultStatus(t("refusal.title"));
}

function renderComplianceOnly(hiddenCount) {
  resetStats(t("stats.paused"));
  setOverviewHint(t("overview.hidden", { n: hiddenCount }));
  replace(
    dom.resultState,
    element("section", { className: "state-panel empty-panel" }, [
      element("h3", { text: t("compliance.title") }),
      element("p", { text: t("compliance.detail", { n: hiddenCount }) }),
      element("p", { text: t("compliance.explainer") }),
    ]),
  );
  state.lastPaint = () => renderComplianceOnly(hiddenCount);
  revealResults();
  setResultStatus(t("compliance.title"));
}

function renderHiddenNotice(count) {
  return element("p", {
    className: "compliance-note",
    text: t("compliance.hidden", { n: count }),
  });
}

/* ------------------------------------------------------------------ */
/* Results views: one dataset, four readings                           */
/* ------------------------------------------------------------------ */

// CTA copy per view; compact layouts get shorter labels, the board row is itself the button.
function viewCtas() {
  return {
    cards: t("detail.cta.cards"),
    timeline: t("detail.cta.timeline"),
    magazine: t("detail.cta.magazine"),
    board: "",
  };
}

/**
 * One detail object per row, shared by every view and the drawer. `peers` lets the drawer offer
 * the rest of the batch without re-deriving anything from the payload.
 */
function buildDetails(data) {
  const details = data.rows.map((row, index) => ({
    kind: data.mode === "retrieve" ? "excerpt" : "citation",
    citation: row.citation,
    quote: row.quote ?? null,
    score: Number.isFinite(row.score) ? row.score : null,
    quoteLength: Number.isFinite(row.quoteLength) ? row.quoteLength : null,
    wasClipped: Boolean(row.wasClipped),
    request: data.request,
    index,
    peers: [],
  }));
  details.forEach((detail) => {
    detail.peers = details;
  });
  return details;
}

function syncViewButtons() {
  document.querySelectorAll(".view-switch [data-view]").forEach((button) => {
    const active = button.dataset.view === state.view;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function paintResults() {
  const data = state.viewData;
  if (!data) {
    return;
  }
  syncViewButtons();
  const children = [];
  if (data.mode === "ask") {
    children.push(buildAnswerPanel(data.payload, data.request));
  }
  const complianceNote = data.mode === "retrieve"
    ? t("retrieve.compliance", { n: data.rows.length, limit: MAX_QUOTE_CODEPOINTS })
    : data.rows.length > 0
      ? t("answer.compliance", { n: data.rows.length })
      : (data.payload.citations ?? []).length === 0
        ? t("answer.noCitations")
        : null;
  if (complianceNote) {
    children.push(element("p", { className: "compliance-note", text: complianceNote }));
  }
  if (data.rows.length > 0) {
    const view = element("div", { className: `results-view results-view--${state.view}` });
    view.append(VIEW_BUILDERS[state.view](data));
    children.push(view);
  }
  if (data.hiddenCount > 0) {
    children.push(renderHiddenNotice(data.hiddenCount));
  }
  replace(dom.resultState, ...children);
  observeReveals(dom.resultState);
}

function buildCardsView(data) {
  // `.result-card` already fades and rises through the `card-in` keyframes; stacking `.reveal`
  // on top would fight the animation fill, so the cards view keeps its native entrance.
  const list = element("div", { className: "result-list" });
  data.rows.forEach((row, index) => {
    const detail = data.details[index];
    list.append(
      data.mode === "retrieve"
        ? renderExcerptCard(row, detail, index)
        : renderCitationCard(row.citation, detail, index),
    );
  });
  return list;
}

function buildTimelineView(data) {
  const rail = element("ol", { className: "timeline-rail" });
  data.rows.forEach((row, index) => {
    const citation = row.citation;
    const headChildren = [sourceBadge(citation.source_type)];
    if (Number.isFinite(row.score)) {
      headChildren.push(
        element("span", { className: "score-chip", text: `score ${formatScore(row.score)}` }),
      );
    }
    const cardChildren = [
      element("time", {
        className: "timeline-date",
        text: citation.published_at,
        attrs: { datetime: citation.published_at },
      }),
      element("div", { className: "timeline-head" }, headChildren),
      element("h3", { text: citation.title }),
      element("p", { className: "timeline-publisher", text: citation.publisher }),
    ];
    if (row.quote) {
      cardChildren.push(element("p", { className: "timeline-quote", text: row.quote }));
    }
    const card = element("article", { className: "timeline-card" }, cardChildren);
    attachCardDetail(card, data.details[index], index, viewCtas().timeline);
    card.classList.add("reveal");
    rail.append(
      element("li", { className: `timeline-entry source-${citation.source_type}` }, [
        element("span", { className: "timeline-marker", attrs: { "aria-hidden": "true" } }, [
          element("span", { className: "timeline-dot" }),
        ]),
        card,
      ]),
    );
  });
  return rail;
}

function buildMagazineView(data) {
  const wrap = element("div", { className: "magazine" });
  const feature = data.rows[0];
  const rightChildren = [sourceBadge(feature.citation.source_type)];
  if (Number.isFinite(feature.score)) {
    const percent = Math.round(Math.max(0, Math.min(1, feature.score)) * 100);
    rightChildren.push(
      element("span", { className: "score-ring", attrs: { style: `--p:${percent}%`, "aria-hidden": "true" } }, [
        element("span", { text: formatScore(feature.score) }),
      ]),
    );
  }
  const featureChildren = [
    element("div", { className: "mag-feature-top" }, [
      element("span", { className: "mag-kicker", text: "FOCUS 01" }),
      element("div", { className: "mag-feature-right" }, rightChildren),
    ]),
    element("h3", { className: "mag-title", text: feature.citation.title }),
  ];
  if (feature.quote) {
    featureChildren.push(element("blockquote", { className: "mag-pull", text: feature.quote }));
  }
  featureChildren.push(
    element("p", {
      className: "mag-meta",
      text: `${feature.citation.publisher} · ${feature.citation.published_at}`,
    }),
  );
  const featureCard = element(
    "article",
    { className: `mag-feature source-${feature.citation.source_type}` },
    featureChildren,
  );
  // The photo banner cycles the four card backgrounds once per response.
  featureCard.style.setProperty(
    "--card-photo",
    `url("/static/assets/cards/bg-0${(state.viewSeries % 4) + 1}.jpg")`,
  );
  attachCardDetail(featureCard, data.details[0], 0, viewCtas().magazine);
  featureCard.classList.add("reveal");
  wrap.append(featureCard);

  if (data.rows.length > 1) {
    const grid = element("div", { className: "mag-grid" });
    data.rows.slice(1).forEach((row, offset) => {
      const index = offset + 1;
      const bodyChildren = [
        element("div", { className: "mag-item-head" }, [
          sourceBadge(row.citation.source_type),
          element("span", {
            className: "mag-item-meta",
            text: `${row.citation.publisher} · ${row.citation.published_at}`,
          }),
        ]),
        element("h4", { text: row.citation.title }),
      ];
      if (row.quote) {
        bodyChildren.push(element("p", { className: "mag-item-quote", text: row.quote }));
      }
      const item = element("article", { className: `mag-item source-${row.citation.source_type}` }, [
        element("span", {
          className: "mag-index",
          text: String(index + 1).padStart(2, "0"),
          attrs: { "aria-hidden": "true" },
        }),
        element("div", { className: "mag-item-body" }, bodyChildren),
      ]);
      attachCardDetail(item, data.details[index], index, viewCtas().magazine);
      item.classList.add("reveal");
      grid.append(item);
    });
    wrap.append(grid);
  }
  return wrap;
}

function buildBoardView(data) {
  const summary = summariseRows(data.mode, data.rows);
  const board = element("div", { className: "board" });

  const ranked = data.rows
    .map((row, index) => ({
      row,
      index,
      value: Number.isFinite(row.score) ? row.score : STRENGTH[row.citation.source_type] ?? 1,
    }))
    .sort((left, right) => right.value - left.value);
  const maxBar = data.mode === "retrieve"
    ? Math.max(...ranked.map((entry) => entry.value), 0.0001)
    : 3;
  const bars = element("ol", { className: "board-bars" });
  ranked.forEach((entry, rank) => {
    const width = Math.max(6, Math.round((entry.value / maxBar) * 100));
    const barRow = element("li", { className: `board-row${rank === 0 ? " is-top" : ""}` }, [
      element("span", { className: "board-rank", text: String(rank + 1).padStart(2, "0") }),
      element("span", { className: "board-id", text: shortId(entry.row.citation.chunk_id) }),
      element("span", { className: "board-track", attrs: { "aria-hidden": "true" } }, [
        element("span", { className: "board-fill", attrs: { style: `--w:${width}%;--i:${rank}` } }),
      ]),
      element("span", {
        className: "board-value",
        text: data.mode === "retrieve" ? formatScore(entry.value) : `${entry.value}/3`,
      }),
    ]);
    attachCardDetail(barRow, data.details[entry.index], entry.index, viewCtas().board);
    barRow.classList.add("reveal");
    bars.append(barRow);
  });
  board.append(
    element("section", { className: "board-panel" }, [
      element("h4", {
        text: data.mode === "retrieve" ? t("board.rank.retrieve") : t("board.rank.ask"),
      }),
      element("p", {
        className: "board-sub",
        text: data.mode === "retrieve" ? t("board.sub.retrieve") : t("board.sub.ask"),
      }),
      bars,
    ]),
  );

  const donutColors = { official: "var(--accent)", media: "var(--sky)", metadata: "var(--gold)" };
  let cursor = 0;
  const stops = summary.sourceRows.map((source, index) => {
    const start = cursor;
    cursor = index === summary.sourceRows.length - 1 ? 100 : cursor + source.share;
    return `${donutColors[source.type]} ${start}% ${cursor}%`;
  });
  const donut = element("div", {
    className: "donut",
    attrs: {
      role: "img",
      "aria-label": summary.sourceRows
        .map((source) =>
          t("stats.sourceShare", {
            label: sourceLabel(source.type),
            n: source.count,
            share: source.share,
          }),
        )
        .join(t("list.sep")),
    },
  }, [
    element("div", { className: "donut-hole" }, [
      element("strong", { text: String(summary.total) }),
      element("span", { text: t("board.unit") }),
    ]),
  ]);
  donut.style.background = `conic-gradient(${stops.join(", ")})`;
  const legend = element(
    "ul",
    { className: "donut-legend" },
    summary.sourceRows.map((source) =>
      element("li", {}, [
        element("span", { className: `legend-dot source-${source.type}` }),
        element("span", { className: "legend-name", text: sourceLabel(source.type) }),
        element("strong", { text: String(source.count) }),
        element("em", { text: `${source.share}%` }),
      ]),
    ),
  );
  board.append(
    element("section", { className: "board-panel" }, [
      element("h4", { text: t("board.sourceMix") }),
      element("div", { className: "donut-wrap" }, [donut, legend]),
    ]),
  );

  const tiles = [
    tileNode("board-tile", String(summary.total), t("board.total"), t("board.thisResponse"), true),
    data.mode === "retrieve"
      ? tileNode(
          "board-tile",
          summary.averageScore === null ? "—" : formatScore(summary.averageScore),
          t("stats.avg"),
          t("stats.avgHint"),
          summary.averageScore !== null,
        )
      : tileNode(
          "board-tile",
          String(summary.publishers),
          t("board.publishers"),
          t("board.dedup"),
          true,
        ),
    tileNode(
      "board-tile",
      summary.spanDays > 0 ? t("stats.days", { n: summary.spanDays }) : t("stats.singleDay"),
      t("board.span"),
      t("stats.dateCount", { n: summary.dates.length }),
    ),
    tileNode(
      "board-tile",
      shortId(data.request.body.snapshot_id),
      t("board.snapshot"),
      data.fromCache ? t("meta.cached") : t("meta.thisResponse"),
    ),
    tileNode("board-tile", ACTIVE_CONFIG, t("board.config"), t("board.mockCorpus")),
  ];
  board.append(element("section", { className: "board-tiles" }, tiles));
  board.querySelectorAll(".is-count strong").forEach((node) => countUpNode(node));
  return board;
}

const VIEW_BUILDERS = {
  cards: buildCardsView,
  timeline: buildTimelineView,
  magazine: buildMagazineView,
  board: buildBoardView,
};

/* ------------------------------------------------------------------ */
/* 本期速览 overview strip                                             */
/* ------------------------------------------------------------------ */

function tileNode(className, value, label, hint, isCount = false) {
  const children = [
    element("strong", { text: value }),
    element("span", { className: "tile-label", text: label }),
  ];
  if (hint) {
    children.push(element("small", { className: "tile-hint", text: hint }));
  }
  return element("div", { className: `${className}${isCount ? " is-count" : ""}` }, children);
}

function setOverviewHint(message) {
  if (!dom.overviewStrip) {
    return;
  }
  replace(
    dom.overviewStrip,
    element("div", { className: "overview-tile is-muted" }, [
      element("strong", { text: "◇" }),
      element("span", { className: "tile-label", text: t("overview.title") }),
      element("small", { className: "tile-hint", text: message }),
    ]),
  );
}

function renderOverview(data) {
  if (!dom.overviewStrip) {
    return;
  }
  if (data.rows.length === 0) {
    setOverviewHint(t("overview.nothing"));
    return;
  }
  const summary = summariseRows(data.mode, data.rows);
  const tiles = [
    tileNode(
      "overview-tile",
      String(summary.total),
      data.mode === "retrieve" ? t("overview.excerpts") : t("overview.citations"),
      t("overview.passed"),
      true,
    ),
    tileNode(
      "overview-tile",
      String(summary.sourceRows.length),
      t("overview.sourceTypes"),
      t("overview.inResponse"),
      true,
    ),
  ];
  if (summary.averageScore !== null) {
    tiles.push(
      tileNode("overview-tile", formatScore(summary.averageScore), t("overview.avg"), t("stats.avgHint"), true),
    );
  }
  tiles.push(
    tileNode(
      "overview-tile",
      summary.spanDays > 0 ? t("stats.days", { n: summary.spanDays }) : t("stats.singleDay"),
      t("overview.span"),
      t("stats.dateCount", { n: summary.dates.length }),
    ),
    tileNode(
      "overview-tile",
      shortId(data.request.body.snapshot_id),
      t("overview.snapshot"),
      data.fromCache ? t("meta.cached") : t("meta.thisResponse"),
    ),
  );
  if (data.fromCache) {
    tiles.push(tileNode("overview-tile", t("overview.hit"), t("overview.cache"), t("overview.sameKey")));
  }
  replace(dom.overviewStrip, ...tiles);
  dom.overviewStrip.querySelectorAll(".is-count strong").forEach((node) => countUpNode(node));
}

/* ------------------------------------------------------------------ */
/* Global atmosphere: scroll progress + reveal-on-scroll               */
/* ------------------------------------------------------------------ */

let revealObserver = null;

function initAtmosphere() {
  if (dom.scrollProgress) {
    let ticking = false;
    const update = () => {
      ticking = false;
      const doc = document.documentElement;
      const scrollable = doc.scrollHeight - doc.clientHeight;
      const ratio = scrollable > 0 ? Math.min(1, Math.max(0, doc.scrollTop / scrollable)) : 0;
      dom.scrollProgress.style.setProperty("--progress", `${(ratio * 100).toFixed(2)}%`);
    };
    window.addEventListener(
      "scroll",
      () => {
        if (!ticking) {
          ticking = true;
          requestAnimationFrame(update);
        }
      },
      { passive: true },
    );
    window.addEventListener("resize", update, { passive: true });
    update();
  }

  // Reduced motion: skip the observer entirely; the stylesheet shows every `.reveal` node.
  if (
    window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
    !("IntersectionObserver" in window)
  ) {
    return;
  }
  revealObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) {
          continue;
        }
        entry.target.classList.add("is-revealed");
        revealObserver.unobserve(entry.target);
      }
    },
    { threshold: 0.15 },
  );
  document.querySelectorAll(".reveal").forEach((node) => revealObserver.observe(node));
}

function observeReveals(root) {
  if (!revealObserver) {
    return;
  }
  root.querySelectorAll(".reveal").forEach((node) => revealObserver.observe(node));
}

function renderApiError(error, request = state.lastRequest) {
  resetStats(t("stats.noCitations"));
  state.viewData = null;
  setOverviewHint(t("overview.error"));
  const status = error instanceof ApiError ? error.status : 0;
  const detail = error instanceof ApiError ? error.detail : error.message;
  const panel = element("section", { className: "state-panel error-panel" });
  const actions = [];

  if (status === 400) {
    panel.append(
      element("h3", { text: t("error.badRequest.title") }),
      element("p", { text: t("error.badRequest.text") }),
      element("p", { className: "state-detail", text: detail }),
    );
  } else if (status === 404) {
    panel.append(
      element("h3", { text: t("error.notFound.title") }),
      element("p", { text: t("error.notFound.text") }),
      element("p", { className: "state-detail", text: detail }),
    );
    actions.push(
      element("button", {
        text: t("error.refresh"),
        attrs: { type: "button", "data-action": "refresh" },
      }),
    );
  } else if (status === 410) {
    panel.append(
      element("h3", { text: t("error.retired.title") }),
      element("p", { text: t("error.retired.text") }),
      element("p", { className: "state-detail", text: detail }),
    );
    actions.push(
      element("button", {
        text: t("error.refresh"),
        attrs: { type: "button", "data-action": "refresh" },
      }),
      element("button", {
        text: t("error.rerun"),
        attrs: { type: "button", "data-action": "rerun-current" },
      }),
    );
  } else if (status === 422) {
    panel.append(
      element("h3", { text: t("error.invalid.title") }),
      element("p", { text: t("error.invalid.text") }),
      element("p", { className: "state-detail", text: detail }),
    );
    actions.push(
      element("button", {
        text: t("error.clearDates"),
        attrs: { type: "button", "data-action": "clear-dates" },
      }),
    );
  } else if (status === 503) {
    panel.append(
      element("h3", { text: t("error.unavailable.title") }),
      element("p", { text: t("error.unavailable.text") }),
      element("p", { className: "state-detail", text: detail }),
    );
    if (error.retryAfter) {
      panel.append(
        element("p", {
          className: "state-detail",
          text: t("error.retryAfter", { seconds: error.retryAfter }),
        }),
      );
    }
    actions.push(
      element("button", {
        text: t("error.retry"),
        attrs: { type: "button", "data-action": "retry" },
      }),
    );
  } else {
    panel.append(
      element("h3", { text: t("error.generic.title") }),
      element("p", { text: t("error.generic.text") }),
      element("p", { className: "state-detail", text: detail || t("error.unknown") }),
    );
    actions.push(
      element("button", {
        text: t("error.retry"),
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
        text: t("error.requestLine", {
          snapshot: request.body.snapshot_id,
          filters: filtersSummary(request.body.filters),
        }),
      }),
    );
  }
  dom.resultMeta.textContent = t("meta.failure", { status: status || t("error.network") });
  replace(dom.resultState, panel);
  state.lastPaint = () => renderApiError(error, request);
  revealResults();
  setResultStatus(t("error.summary", { status: status || t("error.network") }));
}

async function handleAction(action, request) {
  userIntent = true;
  if (action === "refresh") {
    try {
      await loadInventory();
      replace(
        dom.resultState,
        element("section", { className: "state-panel placeholder-panel" }, [
          element("h3", { text: t("refresh.done.title") }),
          element("p", { text: t("refresh.done.text") }),
          element("p", {
            className: "state-detail",
            text: t("refresh.done.current", { id: state.current ?? t("refresh.unknown") }),
          }),
        ]),
      );
      setFormNote(t("note.refreshed"));
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

function autoGrowQuestion() {
  dom.question.style.height = "auto";
  dom.question.style.height = `${Math.min(dom.question.scrollHeight, 132)}px`;
}

function syncQuestionCount() {
  if (!dom.questionCount) {
    return;
  }
  dom.questionCount.textContent = t("form.count", {
    used: dom.question.value.length,
    limit: dom.question.maxLength > 0 ? dom.question.maxLength : 500,
  });
}

/**
 * Static chrome carries `data-i18n` keys so one pass re-labels the whole page. Dynamic panels
 * are re-painted from the stored response by `state.lastPaint`; nothing is re-requested.
 */
function applyStaticCopy() {
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
    node.setAttribute("aria-label", t(node.dataset.i18nAria));
  });
  document.querySelectorAll("[data-i18n-title]").forEach((node) => {
    node.setAttribute("title", t(node.dataset.i18nTitle));
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder));
  });
  document.title = t("site.title");
  const description = document.querySelector('meta[name="description"]');
  if (description) {
    description.setAttribute("content", t("site.description"));
  }
  document.documentElement.lang = getLang() === "zh" ? "zh-CN" : "en";
  dom.langButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.lang === getLang()));
  });
}

function switchLanguage(next) {
  if (setLang(next) !== next) {
    return;
  }
  applyStaticCopy();
  updateModeCopy();
  syncQuestionCount();
  if (state.inventory) {
    renderSnapshotOptions(state.inventory);
    syncSnapshotDisplay();
  }
  if (dom.detailDialog?.open) {
    closeDetail();
  }
  if (state.lastPaint) {
    state.lastPaint();
  }
}

function bindEvents() {
  const contractSection = document.querySelector("#contract");
  if (contractSection instanceof HTMLDetailsElement) {
    const openContract = () => {
      if (window.location.hash === "#contract") {
        contractSection.open = true;
      }
    };
    openContract();
    window.addEventListener("hashchange", openContract);
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
    setFormNote(t("note.switched"));
  });
  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      dom.question.value = button.dataset.example ?? "";
      dom.question.focus();
      autoGrowQuestion();
      syncQuestionCount();
      userIntent = true;
      runQuery();
    });
  });

  dom.question.addEventListener("input", () => {
    autoGrowQuestion();
    syncQuestionCount();
  });
  autoGrowQuestion();
  syncQuestionCount();

  dom.langButtons.forEach((button) => {
    button.addEventListener("click", () => switchLanguage(button.dataset.lang));
  });

  // The contract notes are reference material, not the answer; on a phone they would push the
  // results another screen down, so the summary starts collapsed there.
  if (window.matchMedia("(max-width: 860px)").matches && dom.contractNotes) {
    dom.contractNotes.open = false;
  }

  // One delegation pair on the result container covers every view: cards, timeline, magazine
  // and board items all carry `detail-item` via `attachCardDetail`.
  dom.resultState.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      return;
    }
    const card = event.target.closest(".result-card, .detail-item");
    const detail = card ? detailStore.get(card) : null;
    if (detail) {
      openDetail(detail);
    }
  });
  dom.resultState.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    const card = event.target.closest(".result-card, .detail-item");
    if (!card || event.target.closest("a")) {
      return;
    }
    const detail = detailStore.get(card);
    if (!detail) {
      return;
    }
    event.preventDefault();
    openDetail(detail);
  });

  // View switching is pure re-painting of the cached response; it never issues a request.
  const viewSwitch = document.querySelector(".view-switch");
  viewSwitch?.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-view]");
    if (!button || !state.viewData) {
      return;
    }
    state.view = button.dataset.view;
    paintResults();
  });

  dom.detailClose?.addEventListener("click", closeDetail);
  dom.detailDialog?.addEventListener("click", (event) => {
    if (event.target === dom.detailDialog) {
      closeDetail();
    }
  });
  dom.detailDialog?.addEventListener("close", () => {
    if (lastFocusedCard && document.contains(lastFocusedCard)) {
      lastFocusedCard.focus();
    }
    lastFocusedCard = null;
  });
}

async function init() {
  bindEvents();
  applyStaticCopy();
  updateModeCopy();
  initAtmosphere();
  setOverviewHint(t("overview.initial"));
  try {
    await loadInventory();
    await runQuery();
  } catch (error) {
    renderApiError(error);
    setFormNote(t("note.inventoryFailed"), true);
  }
}

init();
