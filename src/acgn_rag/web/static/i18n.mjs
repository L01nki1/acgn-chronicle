/**
 * i18n for the static query page — a dependency-free dictionary and a few DOM helpers.
 *
 * Scope boundary: this covers UI chrome only. Retrieved titles, quotes, publishers, dates and
 * server-side refusal reasons are corpus evidence in its source language; "translating" them
 * would falsify the source, so they pass through untouched.
 *
 * Static HTML carries `data-i18n="key"` (textContent) and `data-i18n-attr="attr:key;attr:key"`
 * attributes; dynamic strings go through `t()`. Switching the language stores the choice,
 * updates `<html lang>`, re-applies the static attributes and dispatches `acgn:langchange` so
 * the app can re-render the current view in the new language.
 */

export const LANGS = ["zh", "en"];
const STORAGE_KEY = "acgn:lang";

export const STRINGS = {
  zh: {
    "doc.title": "ACGN Chronicle · 档案检索",
    "meta.description": "ACGN Chronicle 跨时档案问答的 G2 查询页面，当前后端为固定 mock 数据。",
    "skip": "跳到主要内容",
    "header.tagline": "跨时档案问答 · 每条结论都要能回到来源",
    "header.modeChip": "G2 · mock 数据",
    "header.banner":
      "当前页是 G2 界面，后端为固定 mock：数据不是真实语料，引用 URL 指向 example.com；“打开原文”会在新窗口打开来源链接。",
    "panel.title": "问档案",
    "panel.intro":
      "输入一个问题，页面只展示检索片段或明确拒答。空结果与拒答是两种状态，不会混在一起。",
    "form.questionLabel": "问题或检索词",
    "form.questionDefault": "第二季什么时候开播？",
    "form.modeLegend": "输出",
    "form.modeRetrieve": "片段检索",
    "form.modeAsk": "带引用问答",
    "form.filtersLabel": "筛选条件",
    "form.sourceLegend": "来源类型",
    "form.srcOfficial": "官方",
    "form.srcMedia": "媒体",
    "form.srcMetadata": "元数据",
    "form.sinceLabel": "开始日期",
    "form.untilLabel": "结束日期",
    "form.mediaIdLabel": "作品 ID（可选）",
    "form.topKLabel": "返回条数",
    "form.noteDefault": "结果绑定到下方快照；切换快照会改变缓存键，不会复用旧语料的答案。",
    "form.noteRetrieve": "结果绑定到下方快照；空结果与拒答是两种状态。",
    "form.noteAsk": "拒答会显示原因与空引用；空结果不会被说成拒答。",
    "form.examplesTitle": "示例问题",
    "form.example.season": "第二季什么时候开播？",
    "form.example.bluray": "蓝光第一卷什么时候发售？",
    "form.example.movie": "月灯食堂剧场版什么时候上映？",
    "form.example.phone": "导演的私人电话是多少？",
    "form.contractTitle": "页面遵守的契约",
    "form.contract.1": "引用对象没有 URL 就不渲染，不灰显、不藏在提示里。",
    "form.contract.2": "正文在服务端裁剪一次，客户端渲染前再裁剪一次，上限 200 个字符。",
    "form.contract.3": "每次结果都显示对应快照；缓存键包含快照、问题、筛选、top_k 与配置。",
    "form.contract.4": "410 表示快照已退役，页面不会自动切到当前快照。",
    "form.contract.5": "没有检索到片段不是拒答；拒答一定会给出原因。",
    "results.snapshotLabel": "快照",
    "results.snapshotReading": "正在读取…",
    "results.snapshotConfig": "配置 A · mock",
    "results.metaWaiting": "等待请求",
    "results.placeholder": "正在读取 mock 快照…",
    "results.footer":
      "数据为 G2 mock；引用 URL 仅用于检查渲染规则。真实语料与检索通过本地真实后端接入。",
    "site.footer.line1": "ACGN Chronicle · CS5481 Topic 5 · G2 查询页面",
    "site.footer.line2": "无 URL 不渲染 · 客户端二次裁剪 · 快照随结果展示 · 410 不自动回退",
    "site.noscript": "这个页面需要 JavaScript 才能读取 mock 快照；脚本未加载时不会展示任何引用。",
    "error.network": "无法连接本地服务：{message}",
    "form.error.empty": "请输入问题或检索词。",
    "form.error.noSnapshot": "还没有选择快照；请先刷新快照列表。",
    "form.error.noSource": "至少选择一个来源类型；全部不选不等于不筛选。",
    "form.error.dateRange": "开始日期不能晚于结束日期。",
    "form.error.mediaId": "作品 ID 必须是正整数。",
    "form.staleSuffix": "表单已修改，结果对应上一次请求",
    "note.queued": "上一个请求尚未完成；这次操作已排队，完成后自动运行。",
    "note.requesting": "正在请求 mock 快照…",
    "note.cacheHit": "缓存命中；缓存键包含快照与筛选条件。",
    "note.done": "请求完成。",
    "note.failed": "请求失败：原因见结果区。",
    "note.snapshotChanged": "已切换快照；重新提交后才会得到该快照的结果。",
    "note.snapshotRefreshed": "快照列表已刷新；请确认选择后重跑。",
    "init.failed": "无法读取快照列表；请确认本地服务已启动。",
    "meta.snapshot": "快照 {id}",
    "meta.fromCache": "缓存命中",
    "meta.fresh": "本次响应",
    "filter.none": "无筛选",
    "filter.sources": "来源 {list}",
    "filter.dates": "日期 {since} → {until}",
    "filter.unlimited": "不限",
    "filter.mediaIds": "作品 ID {list}",
    "filter.join": "；",
    "snapshot.optionCurrent": "当前",
    "snapshot.optionRetained": "保留",
    "snapshot.bound": "绑定 {id}",
    "snapshot.none": "未选择快照",
    "retrieve.empty": "当前筛选条件下没有检索到片段。",
    "retrieve.compliance":
      "{n} 条片段通过引用检查；按 score 排序，仅在本次响应内有效；每条引文最多 {max} 个字符。",
    "retrieve.shown": "已显示 {n} 条检索片段。",
    "card.chars": "字符 {n} / {max}",
    "card.clipped": "客户端二次裁剪",
    "citation.open": "打开原文",
    "citation.openTitle": "在新窗口打开原文：{url}",
    "citation.source": "来源",
    "answer.title": "带引用回答",
    "answer.citedCount": "cited_chunk_ids {n} 条",
    "answer.displayOnly": "回答只展示，不在客户端补全引文",
    "answer.caveat":
      "当前回答来自本地 mock 数据；服务端负责 cited_chunk_ids ⊆ retrieved，客户端只检查引用能否展示。",
    "answer.compliance":
      "{n} 条引用带 URL 与来源字段，按契约展示；“打开原文”在新窗口打开来源链接。",
    "answer.noCitations": "本次回答没有附带引用；引用为空时页面不渲染任何来源卡片。",
    "answer.shown": "已回答，展示 {n} 条引用。",
    "answer.noVisible": "已回答，但没有可展示的引用。",
    "card.fromEvidence": "引用来自本次回答的检索证据",
    "empty.action": "改用无筛选检索",
    "empty.title": "没有检索到片段",
    "empty.body": "空结果表示当前筛选下没有证据，不是系统拒答。页面会保持这两种状态的区别。",
    "empty.filters": "筛选：{summary}",
    "empty.snapshot": "快照：{id}",
    "empty.status": "没有检索到片段：当前筛选下没有证据，这不是拒答。",
    "refuse.noReason": "服务端没有提供拒答原因。",
    "refuse.title": "系统拒绝回答",
    "refuse.body": "拒答不是空结果：它表示知识库没有足够依据，因此没有引用可展示。",
    "refuse.reason": "原因：{reason}",
    "refuse.counts": "引用数量：0；cited_chunk_ids：0",
    "refuse.status": "系统拒绝回答，没有展示任何引用。",
    "hidden.title": "结果因缺少来源信息而未渲染",
    "hidden.body": "{n} 条结果缺少 URL 或必需来源字段；按“无 URL，不渲染”整条隐藏。",
    "hidden.principle": "页面不会灰显、不会把缺失字段补成占位符，也不会为了好看而展示不可追溯的内容。",
    "hidden.notice": "{n} 条结果缺少 URL 或必需来源字段，已按“无 URL，不渲染”整体隐藏。",
    "hidden.status": "{n} 条结果因缺少来源信息被隐藏。",
    "action.refresh": "刷新快照列表",
    "action.rerunCurrent": "改用当前快照显式重跑",
    "action.clearDates": "清空日期筛选",
    "action.retry": "重试",
    "err.400.title": "请求格式有误",
    "err.400.body": "服务端拒绝了请求体；请修正输入后重新提交。",
    "err.404.title": "快照不存在",
    "err.404.body": "这个 id 从未被本机服务知道；页面不会自动猜测另一个快照。",
    "err.410.title": "快照已退役",
    "err.410.body": "该快照超出保留范围。页面不会自动切到当前快照；要用新快照，必须由你显式重跑。",
    "err.422.title": "筛选条件无效",
    "err.422.body": "服务端返回 422；请检查日期范围或筛选组合后重试。",
    "err.503.title": "服务暂不可用",
    "err.503.body": "本机还没有可服务的完整快照，或正在安全切换。",
    "err.retryAfter": "Retry-After：{n} 秒",
    "err.500.title": "请求失败",
    "err.500.body": "无法完成这次请求；结果区没有展示任何不可追溯的内容。",
    "err.unknown": "未知错误",
    "err.meta": "请求失败 · HTTP {status}",
    "err.status": "请求失败：HTTP {status}。",
    "err.networkWord": "网络错误",
    "err.requestDetail": "请求快照：{id} · 筛选：{filters}",
    "refreshed.title": "快照列表已刷新",
    "refreshed.body": "选择快照后，再显式重跑；页面不会自动改写原请求。",
    "refreshed.current": "当前快照：{id}",
    "lang.toggleLabel": "界面语言",
  },
  en: {
    "doc.title": "ACGN Chronicle · Archive Search",
    "meta.description":
      "The G2 query page for ACGN Chronicle's cross-time archive QA; the backend is fixed mock data.",
    "skip": "Skip to main content",
    "header.tagline": "Cross-time archive Q&A · every claim traces back to its source",
    "header.modeChip": "G2 · mock data",
    "header.banner":
      "This page is the G2 UI over a fixed mock backend: the data is not the real corpus, citation URLs point at example.com, and “Open source” opens the source link in a new window.",
    "panel.title": "Ask the archive",
    "panel.intro":
      "Type one question; the page shows either retrieved snippets or an explicit refusal. An empty result and a refusal are two different states and never merge.",
    "form.questionLabel": "Question or search terms",
    "form.questionDefault": "When does season 2 premiere?",
    "form.modeLegend": "Output",
    "form.modeRetrieve": "Snippet retrieval",
    "form.modeAsk": "Cited Q&A",
    "form.filtersLabel": "Filters",
    "form.sourceLegend": "Source types",
    "form.srcOfficial": "Official",
    "form.srcMedia": "Media",
    "form.srcMetadata": "Metadata",
    "form.sinceLabel": "From",
    "form.untilLabel": "Until",
    "form.mediaIdLabel": "Work ID (optional)",
    "form.topKLabel": "Results per query",
    "form.noteDefault":
      "Results are bound to the snapshot below; switching snapshots changes the cache key and never reuses answers from old corpus data.",
    "form.noteRetrieve": "Results are bound to the snapshot below; an empty result and a refusal are two different states.",
    "form.noteAsk": "A refusal shows its reason and empty citations; an empty result is never presented as a refusal.",
    "form.examplesTitle": "Example questions",
    "form.example.season": "When does season 2 premiere?",
    "form.example.bluray": "When does the first Blu-ray volume go on sale?",
    "form.example.movie": "When does the Moonlit Diner film premiere?",
    "form.example.phone": "What is the director's private phone number?",
    "form.contractTitle": "Contract this page follows",
    "form.contract.1": "A citation without a URL is not rendered — never greyed out, never hidden in a tooltip.",
    "form.contract.2": "Quote text is clipped once server-side and again before rendering, capped at 200 code points.",
    "form.contract.3": "Every result shows its snapshot; the cache key covers snapshot, question, filters, top_k and config.",
    "form.contract.4": "410 means the snapshot is retired; the page never silently switches to the current snapshot.",
    "form.contract.5": "An empty result is not a refusal; a refusal always states its reason.",
    "results.snapshotLabel": "Snapshot",
    "results.snapshotReading": "Reading…",
    "results.snapshotConfig": "Config A · mock",
    "results.metaWaiting": "Awaiting request",
    "results.placeholder": "Reading the mock snapshot…",
    "results.footer":
      "Data is the G2 mock; citation URLs exist only to exercise rendering rules. The real corpus and retrieval arrive through the local real backend.",
    "site.footer.line1": "ACGN Chronicle · CS5481 Topic 5 · G2 query page",
    "site.footer.line2":
      "No URL, no render · clipped again client-side · snapshot shown with results · 410 never falls back silently",
    "site.noscript":
      "This page needs JavaScript to read the mock snapshot; without the script no citation is shown.",
    "error.network": "Cannot reach the local server: {message}",
    "form.error.empty": "Enter a question or search terms first.",
    "form.error.noSnapshot": "No snapshot selected yet; refresh the snapshot list first.",
    "form.error.noSource": "Pick at least one source type; unchecking everything is not “no filter”.",
    "form.error.dateRange": "The start date cannot be after the end date.",
    "form.error.mediaId": "The work ID must be a positive integer.",
    "form.staleSuffix": "form changed — the results answer the previous request",
    "note.queued": "The previous request is still running; this action was queued and will run when it settles.",
    "note.requesting": "Requesting the mock snapshot…",
    "note.cacheHit": "Cache hit; the cache key covers snapshot and filters.",
    "note.done": "Request complete.",
    "note.failed": "Request failed: see the results panel for the reason.",
    "note.snapshotChanged": "Snapshot switched; resubmit to get results for that snapshot.",
    "note.snapshotRefreshed": "Snapshot list refreshed; confirm the selection and rerun.",
    "init.failed": "Cannot read the snapshot list; make sure the local server is running.",
    "meta.snapshot": "Snapshot {id}",
    "meta.fromCache": "cache hit",
    "meta.fresh": "this response",
    "filter.none": "no filters",
    "filter.sources": "Sources {list}",
    "filter.dates": "Dates {since} → {until}",
    "filter.unlimited": "any",
    "filter.mediaIds": "Work IDs {list}",
    "filter.join": "; ",
    "snapshot.optionCurrent": "current",
    "snapshot.optionRetained": "retained",
    "snapshot.bound": "Bound {id}",
    "snapshot.none": "No snapshot selected",
    "retrieve.empty": "No snippets were retrieved under the current filters.",
    "retrieve.compliance":
      "{n} snippets passed the citation checks; ordered by score, valid within this response only; each quote is capped at {max} code points.",
    "retrieve.shown": "Showing {n} retrieved snippets.",
    "card.chars": "{n} / {max} code points",
    "card.clipped": "re-clipped client-side",
    "citation.open": "Open source",
    "citation.openTitle": "Open the source in a new window: {url}",
    "citation.source": "Source",
    "answer.title": "Cited answer",
    "answer.citedCount": "cited_chunk_ids: {n}",
    "answer.displayOnly": "The answer is displayed as-is; citations are never completed client-side",
    "answer.caveat":
      "This answer comes from the local mock data; the server guarantees cited_chunk_ids ⊆ retrieved, the client only checks citability.",
    "answer.compliance":
      "{n} citations carry a URL and source fields and are rendered per the contract; “Open source” opens the source link in a new window.",
    "answer.noCitations": "This answer carries no citations; with empty citations the page renders no source cards.",
    "answer.shown": "Answered, showing {n} citations.",
    "answer.noVisible": "Answered, but no citation is renderable.",
    "card.fromEvidence": "Citation from this answer's retrieved evidence",
    "empty.action": "Retry without filters",
    "empty.title": "No snippets retrieved",
    "empty.body":
      "An empty result means no evidence under the current filters — it is not a refusal. The page keeps the two states distinct.",
    "empty.filters": "Filters: {summary}",
    "empty.snapshot": "Snapshot: {id}",
    "empty.status": "No snippets retrieved: no evidence under the current filters — this is not a refusal.",
    "refuse.noReason": "The server did not provide a refusal reason.",
    "refuse.title": "The system declined to answer",
    "refuse.body":
      "A refusal is not an empty result: the knowledge base lacked sufficient grounding, so no citation can be shown.",
    "refuse.reason": "Reason: {reason}",
    "refuse.counts": "Citations: 0; cited_chunk_ids: 0",
    "refuse.status": "The system declined to answer; no citation is shown.",
    "hidden.title": "Results not rendered for missing source fields",
    "hidden.body": "{n} results lack a URL or required source fields; hidden whole per “no URL, no render”.",
    "hidden.principle":
      "The page never greys them out, never substitutes placeholders for missing fields, and never shows untraceable content for looks.",
    "hidden.notice": "{n} results lack a URL or required source fields and were hidden whole per “no URL, no render”.",
    "hidden.status": "{n} results hidden for missing source fields.",
    "action.refresh": "Refresh snapshot list",
    "action.rerunCurrent": "Rerun explicitly on the current snapshot",
    "action.clearDates": "Clear date filters",
    "action.retry": "Retry",
    "err.400.title": "Malformed request",
    "err.400.body": "The server rejected the request body; fix the input and resubmit.",
    "err.404.title": "Snapshot not found",
    "err.404.body": "This id was never known to the local service; the page will not guess another snapshot.",
    "err.410.title": "Snapshot retired",
    "err.410.body":
      "This snapshot is outside the retention window. The page never switches to the current snapshot by itself; rerunning with a new snapshot is your explicit choice.",
    "err.422.title": "Invalid filters",
    "err.422.body": "The server returned 422; check the date range or filter combination and retry.",
    "err.503.title": "Service temporarily unavailable",
    "err.503.body": "The local machine has no servable complete snapshot yet, or is switching safely.",
    "err.retryAfter": "Retry-After: {n} s",
    "err.500.title": "Request failed",
    "err.500.body": "The request could not be completed; the results panel shows nothing untraceable.",
    "err.unknown": "Unknown error",
    "err.meta": "Request failed · HTTP {status}",
    "err.status": "Request failed: HTTP {status}.",
    "err.networkWord": "network error",
    "err.requestDetail": "Requested snapshot: {id} · Filters: {filters}",
    "refreshed.title": "Snapshot list refreshed",
    "refreshed.body": "Pick a snapshot and rerun explicitly; the page never rewrites the original request by itself.",
    "refreshed.current": "Current snapshot: {id}",
    "lang.toggleLabel": "Interface language",
  },
};

let currentLang = null;

export function getLang() {
  if (currentLang) {
    return currentLang;
  }
  let stored = null;
  try {
    stored = globalThis.localStorage?.getItem(STORAGE_KEY) ?? null;
  } catch {
    stored = null;
  }
  currentLang = LANGS.includes(stored) ? stored : "zh";
  return currentLang;
}

export function t(key, vars) {
  const table = STRINGS[getLang()] ?? STRINGS.zh;
  let text = table[key] ?? STRINGS.zh[key] ?? key;
  if (vars) {
    for (const [name, value] of Object.entries(vars)) {
      text = text.replaceAll(`{${name}}`, String(value));
    }
  }
  return text;
}

/** Replace the textContent of every [data-i18n] node and the attributes named by data-i18n-attr. */
export function applyStatic(root = document) {
  for (const node of root.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  for (const node of root.querySelectorAll("[data-i18n-attr]")) {
    for (const pair of node.dataset.i18nAttr.split(";")) {
      const [attr, key] = pair.split(":");
      if (attr && key) {
        node.setAttribute(attr, t(key));
      }
    }
  }
}

/**
 * The seeded question is a corpus query, not chrome: the mock (and the real corpus) answer
 * Chinese questions, so the seed stays in the corpus language under both UI languages — the
 * same choice the example buttons make (translated label, Chinese `data-example` query).
 */
export function seedContentValue() {
  const node = document.querySelector("[data-i18n-value]");
  if (node && !node.dataset.i18nSeeded) {
    node.value = STRINGS.zh[node.dataset.i18nValue] ?? node.value;
    node.dataset.i18nSeeded = "true";
  }
}

export function applyDocumentI18n() {
  document.title = t("doc.title");
  document
    .querySelector('meta[name="description"]')
    ?.setAttribute("content", t("meta.description"));
}

export function syncToggle() {
  const lang = getLang();
  document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
  // `data-ui-lang`, not `data-lang`: the toggle buttons are selected by `[data-lang]`, and an
  // attribute with the same name on <html> would join that selector and swallow their names.
  document.documentElement.dataset.uiLang = lang;
  for (const button of document.querySelectorAll("button[data-lang]")) {
    button.setAttribute("aria-pressed", String(button.dataset.lang === lang));
  }
}

export function bindLangToggle() {
  syncToggle();
  for (const button of document.querySelectorAll("button[data-lang]")) {
    button.addEventListener("click", () => setLang(button.dataset.lang));
  }
}

export function setLang(lang) {
  if (!LANGS.includes(lang) || lang === getLang()) {
    return;
  }
  currentLang = lang;
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, lang);
  } catch {
    // Private-mode storage; the choice then lives for this page view only.
  }
  document.dispatchEvent(new CustomEvent("acgn:langchange", { detail: { lang } }));
}
