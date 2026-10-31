import {
  MAX_QUOTE_CODEPOINTS,
  cacheKey,
  isRenderableCitation,
  selectRenderableExcerpts,
  sourceTypeLabel,
  viewState,
} from "./contract.mjs";

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
};

const state = {
  inventory: null,
  current: null,
  busy: false,
  lastRequest: null,
  // One cache per endpoint: the contract cache tuple does not include the route, so sharing one
  // map between /retrieve and /ask would mix two response shapes under the same key.
  cache: { retrieve: new Map(), ask: new Map() },
};

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
  state.busy = isBusy;
  dom.submit.disabled = isBusy;
  dom.resultState.setAttribute("aria-busy", String(isBusy));
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
  if (state.busy) {
    return;
  }
  let request;
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
  setBusy(true);
  setFormNote("正在请求 mock 快照…");
  try {
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
  }
}

function renderResponse(mode, payload, request, fromCache) {
  const stateName = viewState(mode, payload);
  dom.resultMeta.textContent = [
    mode === "ask" ? "带引用问答" : "片段检索",
    `快照 ${request.body.snapshot_id}`,
    fromCache ? "缓存命中" : "本次响应",
  ].join(" · ");

  if (mode === "retrieve") {
    renderRetrieve(payload, request, stateName);
    return;
  }
  if (stateName === "ask-abstained") {
    renderRefusal(payload);
    return;
  }
  renderAnswer(payload, request);
}

function renderRetrieve(payload, request, stateName) {
  const { items, hiddenCount } = selectRenderableExcerpts(payload);
  if (stateName === "retrieve-empty") {
    renderEmpty(request, "当前筛选条件下没有检索到片段。");
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
      text: `${items.length} 条片段通过引用检查；每条引文最多 ${MAX_QUOTE_CODEPOINTS} 个字符。`,
    }),
  );
  items.forEach((excerpt) => {
    container.append(renderExcerptCard(excerpt));
  });
  if (hiddenCount > 0) {
    container.append(renderHiddenNotice(hiddenCount));
  }
  replace(dom.resultState, container);
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
    element("span", { text: "仅本次响应内排序" }),
    element("span", {
      text: `字符 ${quoteLength} / ${MAX_QUOTE_CODEPOINTS}`,
    }),
    element("span", { text: citation.chunk_id }),
  ]);
  if (excerpt.wasClipped) {
    meta.append(element("span", { className: "clip-flag", text: "客户端二次裁剪" }));
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
  return element("div", { className: "citation-url" }, [
    element("span", { className: "citation-url-label", text: "来源" }),
    element("span", { className: "citation-url-value", text: citation.url }),
    element("span", { className: "jump-note", text: "跳转将在 G2 启用" }),
  ]);
}

function renderAnswer(payload, request) {
  const citations = Array.isArray(payload.citations) ? payload.citations : [];
  const visible = citations.filter((citation) => isRenderableCitation(citation));
  const hiddenCount = citations.length - visible.length;

  const container = element("div");
  const meta = element("div", { className: "answer-meta" }, [
    element("span", { text: `快照 ${request.body.snapshot_id}` }),
    element("span", { text: `cited_chunk_ids ${payload.cited_chunk_ids?.length ?? 0} 条` }),
    element("span", { text: "回答只展示，不在客户端补全引文" }),
  ]);
  container.append(
    element("section", { className: "answer-panel", attrs: { "aria-label": "带引用回答" } }, [
      element("h3", { text: "带引用回答" }),
      element("p", { className: "answer-text", text: payload.answer ?? "" }),
      meta,
      element("p", {
        className: "answer-caveat",
        text: "这是 G1a 模拟回答；服务端负责 cited_chunk_ids ⊆ retrieved，客户端只检查引用能否展示。",
      }),
    ]),
  );

  if (visible.length > 0) {
    container.append(
      element("p", {
        className: "compliance-note",
        text: `${visible.length} 条引用带 URL 与来源字段，按契约展示；跳转链接留到 G2。`,
      }),
    );
    visible.forEach((citation) => {
      container.append(renderCitationCard(citation));
    });
  } else if (citations.length === 0) {
    container.append(
      element("p", {
        className: "compliance-note",
        text: "本次回答没有附带引用；引用为空时页面不渲染任何来源卡片。",
      }),
    );
  } else {
    container.append(renderHiddenNotice(hiddenCount));
  }
  if (hiddenCount > 0 && visible.length > 0) {
    container.append(renderHiddenNotice(hiddenCount));
  }
  replace(dom.resultState, container);
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
      element("span", { text: "引用来自本次回答的检索证据" }),
    ]),
    renderCitationUrl(citation),
  ]);
}

function renderEmpty(request, detail) {
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

function renderApiError(error, request = state.lastRequest) {
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

function bindEvents() {
  dom.form.addEventListener("submit", (event) => {
    event.preventDefault();
    runQuery();
  });
  dom.modeInputs.forEach((input) => {
    input.addEventListener("change", updateModeCopy);
  });
  dom.snapshotSelect.addEventListener("change", () => {
    state.current = dom.snapshotSelect.value;
    syncSnapshotDisplay();
    setFormNote("已切换快照；重新提交后才会得到该快照的结果。");
  });
  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      dom.question.value = button.dataset.example ?? "";
      dom.question.focus();
      runQuery();
    });
  });
}

async function init() {
  bindEvents();
  updateModeCopy();
  try {
    await loadInventory();
    await runQuery();
  } catch (error) {
    renderApiError(error);
    setFormNote("无法读取快照列表；请确认本地服务已启动。", true);
  }
}

init();
