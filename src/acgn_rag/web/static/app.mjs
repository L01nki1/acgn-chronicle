import {
  MAX_QUOTE_CODEPOINTS,
  cacheKey,
  isRenderableCitation,
  selectRenderableExcerpts,
  sourceTypeLabel,
  viewState,
} from "./contract.mjs";
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
  submit: document.querySelector("#submit-button"),
  formNote: document.querySelector("#form-note"),
  snapshotSelect: document.querySelector("#snapshot-select"),
  snapshotId: document.querySelector("#snapshot-id"),
  resultState: document.querySelector("#result-state"),
  resultMeta: document.querySelector("#result-meta"),
  resultsTitle: document.querySelector("#results-title"),
  since: document.querySelector("#since"),
  until: document.querySelector("#until"),
  mediaId: document.querySelector("#media-id"),
  topK: document.querySelector("#top-k"),
  modeInputs: document.querySelectorAll('input[name="mode"]'),
  sourceInputs: document.querySelectorAll('input[name="source_type"]'),
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

/**
 * The form now describes something other than what the panel shows. Say so instead of leaving
 * the previous response looking like the answer to the current controls.
 */
function markFormChanged() {
  if (!state.lastRender || state.stale) {
    return;
  }
  state.stale = true;
  dom.resultMeta.textContent = `${dom.resultMeta.textContent} · 表单已修改，结果对应上一次请求`;
}

function updateModeCopy() {
  const mode = currentMode();
  dom.submit.textContent = mode === "ask" ? "带引用问答" : "检索片段";
  dom.resultsTitle.textContent = mode === "ask" ? "带引用回答" : "检索结果";
  if (mode === "ask") {
    setFormNote("拒答会显示原因与空引用；空结果不会被说成拒答。");
  } else {
    setFormNote("结果绑定到下方快照；空结果与拒答是两种状态。");
  }
}

function filtersSummary(filters) {
  if (!filters) {
    return "无筛选";
  }
  const parts = [];
  if (filters.source_types?.length) {
    parts.push(
      `来源 ${filters.source_types.map((value) => sourceTypeLabel(value)).join(" / ")}`,
    );
  }
  if (filters.since || filters.until) {
    parts.push(`日期 ${filters.since || "不限"} → ${filters.until || "不限"}`);
  }
  if (filters.media_ids?.length) {
    parts.push(`作品 ID ${filters.media_ids.join(", ")}`);
  }
  return parts.length > 0 ? parts.join("；") : "无筛选";
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
      ...options,
    });
  } catch (error) {
    throw new ApiError(0, `无法连接本地服务：${error.message}`);
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
    throw new FormError("请输入问题或检索词。");
  }
  if (!state.current) {
    throw new FormError("还没有选择快照；请先刷新快照列表。");
  }

  const filters = {};
  const sourceTypes = selectedSourceTypes();
  if (sourceTypes.length > 0) {
    filters.source_types = sourceTypes;
  }
  if (dom.since.value) {
    filters.since = dom.since.value;
  }
  if (dom.until.value) {
    filters.until = dom.until.value;
  }
  if (dom.mediaId.value.trim()) {
    const mediaId = Number(dom.mediaId.value);
    if (!Number.isInteger(mediaId) || mediaId < 1) {
      throw new FormError("作品 ID 必须是正整数。");
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
    const suffix = id === inventory.current ? "当前" : "保留";
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
  dom.snapshotId.textContent = state.current ? `绑定 ${state.current}` : "未选择快照";
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
    setFormNote("上一个请求尚未完成；这次操作已排队，完成后自动运行。");
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
    setFormNote("正在请求 mock 快照…");
    const cache = state.cache[request.mode];
    if (cache.has(request.key)) {
      renderResponse(request.mode, cache.get(request.key), request, true);
      setFormNote("缓存命中；缓存键包含快照与筛选条件。");
      return;
    }
    const path = request.mode === "ask" ? "/ask" : "/retrieve";
    const payload = await requestJson(path, {
      method: "POST",
      body: JSON.stringify(request.body),
    });
    cache.set(request.key, payload);
    renderResponse(request.mode, payload, request, false);
    setFormNote("请求完成。");
  } catch (error) {
    renderApiError(error, request);
    setFormNote("请求失败：原因见结果区。", true);
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
      element("span", { text: sourceTypeLabel(sourceType) }),
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

const STAT_TABS = ["来源", "量化", "分布", "时间"];

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

  let current = 0;

  const slides = cards.map((card, index) =>
    element(
      "div",
      {
        className: `stat-slide${index === 0 ? " is-active" : ""}`,
        attrs: {
          role: "tabpanel",
          "aria-label": `${STAT_TABS[index] ?? "剖面"} ${index + 1} / ${cards.length}`,
          "aria-hidden": String(index !== 0),
        },
      },
      [card],
    ),
  );

  const tabButtons = cards.map((_, index) =>
    element("button", {
      className: `stat-tab${index === 0 ? " is-active" : ""}`,
      text: STAT_TABS[index] ?? `第 ${index + 1} 页`,
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
    attrs: { type: "button", "aria-label": "上一张剖面" },
  });
  const next = element("button", {
    className: "stat-arrow next",
    attrs: { type: "button", "aria-label": "下一张剖面" },
  });
  previous.addEventListener("click", () => goTo(current - 1));
  next.addEventListener("click", () => goTo(current + 1));

  const tabs = element(
    "div",
    { className: "stat-tabs", attrs: { role: "tablist", "aria-label": "剖面分组" } },
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
      element("p", { text: "正在计算本次响应剖面…" }),
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
    resetStats("本次没有可统计的引用。");
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
        title: `${sourceTypeLabel(row.type)} ${row.share}%`,
      },
    }),
  );
  const legendRows = sourceRows.map((row) =>
    element("li", {}, [
      element("span", { className: `legend-dot source-${row.type}` }),
      element("span", { className: "legend-name", text: sourceTypeLabel(row.type) }),
      element("strong", { text: String(row.count) }),
      element("em", { text: `${row.share}%` }),
      element("small", {
        text: row.average === null ? "引用来源" : `均分 ${formatScore(row.average)}`,
      }),
    ]),
  );

  const sourceCard = statCard("来源构成", `${total} 条`, [
    element("div", {
      className: "source-flow",
      attrs: {
        role: "img",
        "aria-label": sourceRows
          .map((row) => `${sourceTypeLabel(row.type)} ${row.count} 条，占 ${row.share}%`)
          .join("；"),
      },
    }, flowSegments),
    element("ul", { className: "source-legend" }, legendRows),
  ]);

  const metricValues = mode === "retrieve"
    ? [
        metricBlock("平均分", averageScore === null ? "—" : formatScore(averageScore), "本次响应内"),
        metricBlock("最高分", maxScore === null ? "—" : formatScore(maxScore), "仅用于排序"),
        metricBlock("时间跨度", spanDays > 0 ? `${spanDays} 天` : "单日", `${dates.length} 个日期`),
        metricBlock("平均字符", `${averageLength} 字`, `上限 ${MAX_QUOTE_CODEPOINTS}`),
      ]
    : [
        metricBlock("引用", `${total} 条`, "通过 URL 检查"),
        metricBlock("来源方", `${publishers} 个`, "发布方去重"),
        metricBlock("时间跨度", spanDays > 0 ? `${spanDays} 天` : "单日", `${dates.length} 个日期`),
        metricBlock("回答", `${(payload.answer ?? "").length} 字`, "只展示，不补全"),
      ];
  const metricCard = statCard(
    mode === "retrieve" ? "量化切片" : "回答切片",
    mode === "retrieve" ? "5 项" : "4 项",
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
    mode === "retrieve" ? "分数分布" : "证据等级",
    mode === "retrieve" ? "按 score" : "按来源",
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
    "发布时间线",
    monthKeys.length ? `${monthKeys.length} 个月` : "无日期",
    [
      monthKeys.length
        ? element("ol", { className: "timeline-list" }, timelineRows)
        : element("p", { className: "stat-note", text: "本次引用没有可解析的发布日期。" }),
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

function attachCardDetail(card, detail, index, cta = "展开完整档案 →") {
  // `detail-item` is the delegation hook every view shares; `.result-card` stays the cards-view
  // skin so the photo-background rules keep applying only where they were designed to.
  card.classList.add("is-interactive", "detail-item");
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.setAttribute("aria-label", `查看「${detail.citation.title}」的完整档案`);
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
        element("span", { text: sourceTypeLabel(sourceType) }),
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
          text: `该引文已按 ${MAX_QUOTE_CODEPOINTS} 字符上限裁剪；客户端不会展开或补全。`,
        }),
      );
    }
  } else if (detail.kind === "citation") {
    children.push(
      element("p", {
        className: "detail-note",
        text: "本次回答的引用只携带来源元数据，不包含正文摘录；正文摘录只在片段检索中返回。",
      }),
    );
  }

  children.push(
    detailSection(
      "档案字段",
      element("div", { className: "detail-fields" }, [
        detailField("chunk_id", citation.chunk_id ?? "—"),
        detailField("来源类型", sourceTypeLabel(sourceType)),
        detailField("发布方", citation.publisher ?? "—"),
        detailField("发布日期", citation.published_at ?? "—"),
        detailField(
          "排序分数",
          Number.isFinite(detail.score) ? formatScore(detail.score) : "—",
          Number.isFinite(detail.score) ? detail.score * 100 : null,
        ),
        detailField(
          "引文字符",
          Number.isFinite(detail.quoteLength)
            ? `${detail.quoteLength} / ${MAX_QUOTE_CODEPOINTS}`
            : "—",
          Number.isFinite(detail.quoteLength)
            ? (detail.quoteLength / MAX_QUOTE_CODEPOINTS) * 100
            : null,
        ),
        detailField("快照", detail.request?.body?.snapshot_id ?? "—"),
      ]),
    ),
  );

  if (detail.request) {
    const chips = [
      ["问题", detail.request.body.text],
      ["模式", detail.request.mode === "ask" ? "带引用问答" : "片段检索"],
      ["top_k", detail.request.body.top_k],
      ["筛选", filtersSummary(detail.request.body.filters)],
      ["配置", ACTIVE_CONFIG],
      ["缓存键", shortId(detail.request.key)],
    ].map(([label, value]) =>
      element("span", { className: "detail-chip" }, [
        element("em", { text: label }),
        element("span", { text: String(value ?? "—") }),
      ]),
    );
    children.push(detailSection("请求上下文", element("div", { className: "detail-chips" }, chips)));
  }

  children.push(
    detailSection(
      "来源链接",
      element("div", { className: "detail-link-surface" }, [renderCitationUrl(citation)]),
      element("p", {
        className: "detail-note",
        text: "仅展示服务端返回的字段；缺失字段不会在客户端补全。",
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
          element("span", { className: "detail-related-badge", text: sourceTypeLabel(peer.citation.source_type) }),
          element("span", { className: "detail-related-title", text: peer.citation.title }),
        ],
      );
      // Swapping the drawer body keeps the dialog open; focus restore still points at the
      // original card because `lastFocusedCard` is only captured on the first show.
      chip.addEventListener("click", () => openDetail(peer));
      return chip;
    });
    children.push(detailSection("同批结果", element("div", { className: "detail-related" }, related)));
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

function renderResponse(mode, payload, request, fromCache) {
  state.lastRequest = request;
  const stateName = viewState(mode, payload);
  dom.resultMeta.textContent = [
    mode === "ask" ? "带引用问答" : "片段检索",
    `快照 ${request.body.snapshot_id}`,
    fromCache ? "缓存命中" : "本次响应",
  ].join(" · ");
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
    renderEmpty(request, "当前筛选条件下没有检索到片段。");
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
    complianceNote: `${items.length} 条片段通过引用检查；每条引文最多 ${MAX_QUOTE_CODEPOINTS} 个字符。`,
  };
  state.viewData.details = buildDetails(state.viewData);
  renderOverview(state.viewData);
  paintResults();
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
    element("span", { text: "仅本次响应内排序" }),
    element("span", {
      text: `字符 ${quoteLength} / ${MAX_QUOTE_CODEPOINTS}`,
    }),
    element("span", { text: citation.chunk_id }),
  ]);
  if (row.wasClipped) {
    meta.append(element("span", { className: "clip-flag", text: "客户端二次裁剪" }));
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
    text: citation.url,
    attrs: {
      href: citation.url,
      target: "_blank",
      rel: "noopener noreferrer",
      title: "在新窗口打开原文",
    },
  });
  link.append(
    element("span", { className: "jump-mark", attrs: { "aria-hidden": "true" }, text: " ↗" }),
  );
  return element("div", { className: "citation-url" }, [
    element("span", { className: "citation-url-label", text: "来源" }),
    link,
  ]);
}

function buildAnswerPanel(payload, request) {
  const meta = element("div", { className: "answer-meta" }, [
    element("span", { text: `快照 ${request.body.snapshot_id}` }),
    element("span", { text: `cited_chunk_ids ${payload.cited_chunk_ids?.length ?? 0} 条` }),
    element("span", { text: "回答只展示，不在客户端补全引文" }),
  ]);
  return element("section", { className: "answer-panel", attrs: { "aria-label": "带引用回答" } }, [
    element("h3", { text: "带引用回答" }),
    element("p", { className: "answer-text", text: payload.answer ?? "" }),
    meta,
    element("p", {
      className: "answer-caveat",
      text: "当前回答来自本地 mock 数据；服务端负责 cited_chunk_ids ⊆ retrieved，客户端只检查引用能否展示。",
    }),
  ]);
}

function renderAnswer(payload, request, fromCache) {
  const citations = Array.isArray(payload.citations) ? payload.citations : [];
  const visible = citations.filter((citation) => isRenderableCitation(citation));
  const hiddenCount = citations.length - visible.length;
  renderStats("ask", visible, request, payload);

  let complianceNote = null;
  if (visible.length > 0) {
    complianceNote = `${visible.length} 条引用带 URL 与来源字段，按契约展示；点击来源可在新窗口打开原文。`;
  } else if (citations.length === 0) {
    complianceNote = "本次回答没有附带引用；引用为空时页面不渲染任何来源卡片。";
  }

  state.viewSeries += 1;
  state.viewData = {
    mode: "ask",
    rows: normaliseRows("ask", visible),
    request,
    payload,
    hiddenCount,
    fromCache,
    complianceNote,
  };
  state.viewData.details = buildDetails(state.viewData);
  renderOverview(state.viewData);
  paintResults();
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
      element("span", { text: "引用来自本次回答的检索证据" }),
    ]),
    renderCitationUrl(citation),
  ]);
  attachCardDetail(card, detail, index);
  return card;
}

function renderEmpty(request, detail) {
  resetStats("本次筛选没有可统计的片段。");
  setOverviewHint("空结果是有效回答，不是拒答。");
  const filters = request.body.filters;
  const clearFilters = element("button", { text: "改用无筛选检索", attrs: { type: "button" } });
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
      element("h3", { text: "没有检索到片段" }),
      element("p", { text: detail }),
      element("p", {
        text: "空结果表示当前筛选下没有证据，不是系统拒答。页面会保持这两种状态的区别。",
      }),
      element("p", { className: "state-detail", text: `筛选：${filtersSummary(filters)}` }),
      element("p", { className: "state-detail", text: `快照：${request.body.snapshot_id}` }),
      element("div", { className: "state-actions" }, [clearFilters]),
    ]),
  );
}

function renderRefusal(payload) {
  resetStats("本次回答没有引用，无法统计来源。");
  setOverviewHint("系统拒答；没有引用可速览。");
  const reason =
    typeof payload.abstain_reason === "string" && payload.abstain_reason.trim()
      ? payload.abstain_reason
      : "服务端没有提供拒答原因。";
  const answer =
    typeof payload.answer === "string" && payload.answer.trim() ? payload.answer : reason;

  replace(
    dom.resultState,
    element("section", { className: "state-panel refusal-panel" }, [
      element("h3", { text: "系统拒绝回答" }),
      element("p", { text: answer }),
      element("p", {
        text: "拒答不是空结果：它表示知识库没有足够依据，因此没有引用可展示。",
      }),
      element("p", { className: "state-detail", text: `原因：${reason}` }),
      element("p", { className: "state-detail", text: "引用数量：0；cited_chunk_ids：0" }),
    ]),
  );
}

function renderComplianceOnly(hiddenCount) {
  resetStats("没有可展示的引用，统计暂停。");
  setOverviewHint(`${hiddenCount} 条结果因缺少来源信息未渲染。`);
  replace(
    dom.resultState,
    element("section", { className: "state-panel empty-panel" }, [
      element("h3", { text: "结果因缺少来源信息而未渲染" }),
      element("p", {
        text: `${hiddenCount} 条结果缺少 URL 或必需来源字段；按“无 URL，不渲染”整条隐藏。`,
      }),
      element("p", {
        text: "页面不会灰显、不会把缺失字段补成占位符，也不会为了好看而展示不可追溯的内容。",
      }),
    ]),
  );
}

function renderHiddenNotice(count) {
  return element("p", {
    className: "compliance-note",
    text: `${count} 条结果缺少 URL 或必需来源字段，已按“无 URL，不渲染”整体隐藏。`,
  });
}

/* ------------------------------------------------------------------ */
/* Results views: one dataset, four readings                           */
/* ------------------------------------------------------------------ */

// CTA copy per view; compact layouts get shorter labels, the board row is itself the button.
const VIEW_CTAS = {
  cards: "展开完整档案 →",
  timeline: "展开档案 →",
  magazine: "展开全文 →",
  board: "",
};

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
  if (data.complianceNote) {
    children.push(element("p", { className: "compliance-note", text: data.complianceNote }));
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
    attachCardDetail(card, data.details[index], index, VIEW_CTAS.timeline);
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
  attachCardDetail(featureCard, data.details[0], 0, VIEW_CTAS.magazine);
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
      attachCardDetail(item, data.details[index], index, VIEW_CTAS.magazine);
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
    attachCardDetail(barRow, data.details[entry.index], entry.index, VIEW_CTAS.board);
    barRow.classList.add("reveal");
    bars.append(barRow);
  });
  board.append(
    element("section", { className: "board-panel" }, [
      element("h4", { text: data.mode === "retrieve" ? "分数排行" : "证据强度" }),
      element("p", {
        className: "board-sub",
        text: data.mode === "retrieve" ? "仅本次响应内排序" : "官方 3 / 媒体 2 / 元数据 1",
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
        .map((source) => `${sourceTypeLabel(source.type)} ${source.count} 条，占 ${source.share}%`)
        .join("；"),
    },
  }, [
    element("div", { className: "donut-hole" }, [
      element("strong", { text: String(summary.total) }),
      element("span", { text: "条" }),
    ]),
  ]);
  donut.style.background = `conic-gradient(${stops.join(", ")})`;
  const legend = element(
    "ul",
    { className: "donut-legend" },
    summary.sourceRows.map((source) =>
      element("li", {}, [
        element("span", { className: `legend-dot source-${source.type}` }),
        element("span", { className: "legend-name", text: sourceTypeLabel(source.type) }),
        element("strong", { text: String(source.count) }),
        element("em", { text: `${source.share}%` }),
      ]),
    ),
  );
  board.append(
    element("section", { className: "board-panel" }, [
      element("h4", { text: "来源构成" }),
      element("div", { className: "donut-wrap" }, [donut, legend]),
    ]),
  );

  const tiles = [
    tileNode("board-tile", String(summary.total), "条数", "本次响应", true),
    data.mode === "retrieve"
      ? tileNode(
          "board-tile",
          summary.averageScore === null ? "—" : formatScore(summary.averageScore),
          "平均分",
          "仅本次响应内",
          summary.averageScore !== null,
        )
      : tileNode("board-tile", String(summary.publishers), "来源方", "发布方去重", true),
    tileNode(
      "board-tile",
      summary.spanDays > 0 ? `${summary.spanDays} 天` : "单日",
      "时间跨度",
      `${summary.dates.length} 个日期`,
    ),
    tileNode("board-tile", shortId(data.request.body.snapshot_id), "快照", data.fromCache ? "缓存命中" : "本次响应"),
    tileNode("board-tile", ACTIVE_CONFIG, "配置", "mock 语料"),
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
      element("span", { className: "tile-label", text: "本期速览" }),
      element("small", { className: "tile-hint", text: message }),
    ]),
  );
}

function renderOverview(data) {
  if (!dom.overviewStrip) {
    return;
  }
  if (data.rows.length === 0) {
    setOverviewHint("本次响应没有可速览的条目。");
    return;
  }
  const summary = summariseRows(data.mode, data.rows);
  const tiles = [
    tileNode(
      "overview-tile",
      String(summary.total),
      data.mode === "retrieve" ? "片段" : "引用",
      "通过引用检查",
      true,
    ),
    tileNode("overview-tile", String(summary.sourceRows.length), "来源类型", "本次响应内", true),
  ];
  if (summary.averageScore !== null) {
    tiles.push(tileNode("overview-tile", formatScore(summary.averageScore), "平均分", "仅本次响应内", true));
  }
  tiles.push(
    tileNode(
      "overview-tile",
      summary.spanDays > 0 ? `${summary.spanDays} 天` : "单日",
      "时间跨度",
      `${summary.dates.length} 个日期`,
    ),
    tileNode("overview-tile", shortId(data.request.body.snapshot_id), "快照", data.fromCache ? "缓存命中" : "本次响应"),
  );
  if (data.fromCache) {
    tiles.push(tileNode("overview-tile", "命中", "缓存", "同一缓存键"));
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
  resetStats("请求失败，剖面统计不可用。");
  state.viewData = null;
  setOverviewHint("请求失败；速览不可用。");
  const status = error instanceof ApiError ? error.status : 0;
  const detail = error instanceof ApiError ? error.detail : error.message;
  const panel = element("section", { className: "state-panel error-panel" });
  const actions = [];

  if (status === 400) {
    panel.append(
      element("h3", { text: "请求格式有误" }),
      element("p", { text: "服务端拒绝了请求体；请修正输入后重新提交。" }),
      element("p", { className: "state-detail", text: detail }),
    );
  } else if (status === 404) {
    panel.append(
      element("h3", { text: "快照不存在" }),
      element("p", { text: "这个 id 从未被本机服务知道；页面不会自动猜测另一个快照。" }),
      element("p", { className: "state-detail", text: detail }),
    );
    actions.push(
      element("button", {
        text: "刷新快照列表",
        attrs: { type: "button", "data-action": "refresh" },
      }),
    );
  } else if (status === 410) {
    panel.append(
      element("h3", { text: "快照已退役" }),
      element("p", {
        text: "该快照超出保留范围。页面不会自动切到当前快照；要用新快照，必须由你显式重跑。",
      }),
      element("p", { className: "state-detail", text: detail }),
    );
    actions.push(
      element("button", {
        text: "刷新快照列表",
        attrs: { type: "button", "data-action": "refresh" },
      }),
      element("button", {
        text: "改用当前快照显式重跑",
        attrs: { type: "button", "data-action": "rerun-current" },
      }),
    );
  } else if (status === 422) {
    panel.append(
      element("h3", { text: "筛选条件无效" }),
      element("p", { text: "服务端返回 422；请检查日期范围或筛选组合后重试。" }),
      element("p", { className: "state-detail", text: detail }),
    );
    actions.push(
      element("button", {
        text: "清空日期筛选",
        attrs: { type: "button", "data-action": "clear-dates" },
      }),
    );
  } else if (status === 503) {
    panel.append(
      element("h3", { text: "服务暂不可用" }),
      element("p", { text: "本机还没有可服务的完整快照，或正在安全切换。" }),
      element("p", { className: "state-detail", text: detail }),
    );
    if (error.retryAfter) {
      panel.append(
        element("p", {
          className: "state-detail",
          text: `Retry-After：${error.retryAfter} 秒`,
        }),
      );
    }
    actions.push(
      element("button", {
        text: "重试",
        attrs: { type: "button", "data-action": "retry" },
      }),
    );
  } else {
    panel.append(
      element("h3", { text: "请求失败" }),
      element("p", { text: "无法完成这次请求；结果区没有展示任何不可追溯的内容。" }),
      element("p", { className: "state-detail", text: detail || "未知错误" }),
    );
    actions.push(
      element("button", {
        text: "重试",
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
        text: `请求快照：${request.body.snapshot_id} · 筛选：${filtersSummary(request.body.filters)}`,
      }),
    );
  }
  dom.resultMeta.textContent = `请求失败 · HTTP ${status || "network"}`;
  replace(dom.resultState, panel);
}

async function handleAction(action, request) {
  if (action === "refresh") {
    try {
      await loadInventory();
      replace(
        dom.resultState,
        element("section", { className: "state-panel placeholder-panel" }, [
          element("h3", { text: "快照列表已刷新" }),
          element("p", { text: "选择快照后，再显式重跑；页面不会自动改写原请求。" }),
          element("p", {
            className: "state-detail",
            text: `当前快照：${state.current ?? "未知"}`,
          }),
        ]),
      );
      setFormNote("快照列表已刷新；请确认选择后重跑。");
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
    setFormNote("已切换快照；重新提交后才会得到该快照的结果。");
  });
  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      dom.question.value = button.dataset.example ?? "";
      dom.question.focus();
      autoGrowQuestion();
      runQuery();
    });
  });

  dom.question.addEventListener("input", autoGrowQuestion);
  autoGrowQuestion();

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
  updateModeCopy();
  initAtmosphere();
  setOverviewHint("提交检索后，这里给出本期速览。");
  try {
    await loadInventory();
    await runQuery();
  } catch (error) {
    renderApiError(error);
    setFormNote("无法读取快照列表；请确认本地服务已启动。", true);
  }
}

init();
