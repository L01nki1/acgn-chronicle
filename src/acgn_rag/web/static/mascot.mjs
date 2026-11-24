/*
 * Ai-chan decoration layer (presentation only).
 *
 * Everything here is decoration painted around what app.mjs renders: it never fetches, never
 * changes a request, and never edits result text. It watches the result region for two signals
 * app.mjs already emits - `aria-busy` on #result-state and the panels it paints - and reacts:
 *
 * - a "thinking stage" with an animated character, inner monologue, step list and question
 *   tokens while a query is in flight, then a short reaction to the outcome;
 * - characters and pixel stickers resting on result cards, the answer panel and state panels;
 * - a hero figure and the Q-version white-haired GPT girl resting both hands on the query
 *   board, which steps aside when the board is clicked, a ribbon marquee, and an AI girl
 *   peeking over the results sheet.
 *
 * All characters are cut from ai-chan-sheets; pixel sprites come from Sprite Catalog
 * (see assets/sprites/CREDITS.txt).
 */
import { getLang } from "./i18n.mjs";

const ASSET = "/static/assets/";
const face = (name) => `${ASSET}mascot/${name}.webp`;
const sprite = (name) => `${ASSET}sprites/${name}.png`;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const COPY = {
  en: {
    hero: [
      "Hi! I'm Ai-chan, your archive guide ✦",
      "Ask about premieres, Blu-rays, films…",
      "Every answer I give comes with a source.",
      "Poke me if you get bored~",
    ],
    ribbon: ["ACGN CHRONICLE", "DAILY ARCHIVE", "TRACEABLE SOURCES", "REPLAYABLE SNAPSHOTS", "ASK · READ · CITE"],
    kicker: "Thinking",
    title: "Ai-chan is thinking…",
    stepsAsk: ["Reading your question", "Searching the archive", "Weighing the sources", "Writing a cited answer"],
    stepsRetrieve: ["Reading your question", "Searching the archive", "Ranking excerpts", "Clipping quotes to 200"],
    monologue: [
      "Hmm… let me read that once more, carefully.",
      "Flipping through today's snapshot…",
      "This one looks official — that's a strong source!",
      "Do the dates line up? Let me double-check…",
      "Almost there, tidying up the citations ✦",
    ],
    react: {
      success: "Found it!",
      empty: "Nothing matched…",
      refusal: "I can't answer that one.",
      error: "Something went wrong…",
    },
    answerTag: "Ai-chan's answer",
    peekShow: "Show the archive assistant",
    peekHide: "Put the assistant back on the board",
    perchPoke: "Poke the chibi",
  },
  zh: {
    hero: [
      "你好！我是 Ai 酱，你的档案向导 ✦",
      "首播、蓝光、剧场版……尽管问我～",
      "我给出的每个回答都带着出处哦。",
      "无聊的话可以戳戳我～",
    ],
    ribbon: ["ACGN 编年史", "每日档案", "来源可追溯", "快照可回放", "提问 · 阅读 · 引用"],
    kicker: "思考中",
    title: "Ai 酱正在思考……",
    stepsAsk: ["理解你的问题", "翻找档案库", "权衡信息来源", "撰写带引用的回答"],
    stepsRetrieve: ["理解你的问题", "翻找档案库", "给摘录排序", "把引文裁到 200 字"],
    monologue: [
      "嗯……让我再仔细读一遍问题。",
      "正在翻今天的快照……",
      "这条是官方来源，可信度很高！",
      "日期对得上吗？我再核对一下……",
      "马上就好，正在整理引用 ✦",
    ],
    react: {
      success: "找到啦！",
      empty: "没有匹配的内容……",
      refusal: "这个问题我不能回答。",
      error: "出了点问题……",
    },
    answerTag: "Ai 酱的回答",
    peekShow: "唤出档案助手",
    peekHide: "让助手回到问答板上",
    perchPoke: "戳一下看板娘",
  },
};

function copy(key) {
  const dict = COPY[getLang()] ?? COPY.en;
  return dict[key] ?? COPY.en[key];
}

function pick(list) {
  return list[Math.floor(Math.random() * list.length)];
}

function el(tag, className, attrs = {}, children = []) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  for (const [name, value] of Object.entries(attrs)) {
    if (name === "text") {
      node.textContent = value;
    } else if (name === "style") {
      node.setAttribute("style", value);
    } else {
      node.setAttribute(name, value);
    }
  }
  for (const child of children) {
    node.append(child);
  }
  return node;
}

function img(src, className, attrs = {}) {
  return el("img", className, { src, alt: "", decoding: "async", draggable: "false", ...attrs });
}

// Warm the cache so faces swap without a flash.
for (const name of [
  "gpt-adult-full", "gpt-adult-smile", "gpt-adult-surprised", "gpt-adult-pout", "gpt-adult-sad",
  "gpt-adult-calm", "gpt-adult-focus",
  "gpt-q-peek", "ai-gemini",
  "q-deepseek-smile", "q-deepseek-calm", "q-deepseek-surprised", "q-deepseek-shy",
  "q-deepseek-pout", "q-deepseek-happy", "q-deepseek-serious", "q-deepseek-sad",
]) {
  const preload = new Image();
  preload.src = face(name);
}

/* ------------------------------------------------------------------ */
/* Static decorations: hero figure, ribbon, sheet peek, rail cat       */
/* ------------------------------------------------------------------ */

const heroBubble = el("p", "hero-figure-bubble");
const heroQuery = window.matchMedia("(min-width: 701px)");
let heroLine = 0;
let heroRoot = null;
let heroPeek = null;

function setHeroStage(open) {
  if (!heroRoot) {
    return;
  }
  const next = Boolean(open && heroQuery.matches);
  heroRoot.classList.toggle("is-stage-open", next);
  if (heroPeek) {
    heroPeek.setAttribute("aria-expanded", String(next));
    heroPeek.setAttribute("aria-label", copy(next ? "peekHide" : "peekShow"));
  }
}

function mountHero() {
  const hero = document.querySelector(".hero");
  const card = hero?.querySelector(".query-card");
  if (!hero) {
    return;
  }
  heroRoot = hero;
  const figureImg = img(face("gpt-adult-full"), "hero-figure-img", { width: "478", height: "760" });
  const figure = el("div", "hero-figure", { "aria-hidden": "true" }, [
    el("span", "hero-figure-ring"),
    figureImg,
    heroBubble,
    img(sprite("heart"), "hero-sprite s1"),
    img(sprite("gem"), "hero-sprite s2"),
    img(sprite("moon"), "hero-sprite s3"),
    img(sprite("flower"), "hero-sprite s4"),
    img(sprite("candy"), "hero-sprite s5"),
  ]);
  hero.append(figure);
  heroBubble.textContent = copy("hero")[0];

  // Default pose: only the head and both paws show, resting on the full-width board edge.
  if (card) {
    heroPeek = el(
      "button",
      "query-peek",
      { type: "button", "aria-expanded": "false", "aria-label": copy("peekShow") },
      [
        el("span", "query-peek-clip", {}, [img(face("gpt-q-peek"), "query-peek-face")]),
        el("span", "paw query-peek-paw l"),
        el("span", "paw query-peek-paw r"),
      ],
    );
    card.append(heroPeek);
    heroPeek.addEventListener("click", (event) => {
      event.stopPropagation();
      setHeroStage(true);
    });
    // Clicking bare board surface toggles the stage; controls keep working as before.
    card.addEventListener("click", (event) => {
      if (!heroQuery.matches) {
        return;
      }
      if (
        event.target instanceof Element &&
        event.target.closest("input, textarea, select, button, label, a, summary, details")
      ) {
        return;
      }
      setHeroStage(!hero.classList.contains("is-stage-open"));
    });
  }

  figureImg.addEventListener("click", () => {
    setHeroStage(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setHeroStage(false);
    }
  });
  heroQuery.addEventListener("change", (event) => {
    if (!event.matches) {
      setHeroStage(false);
    }
  });

  const nextLine = () => {
    heroBubble.classList.add("is-swapping");
    window.setTimeout(() => {
      heroLine = (heroLine + 1) % copy("hero").length;
      heroBubble.textContent = copy("hero")[heroLine];
      heroBubble.classList.remove("is-swapping");
    }, 260);
  };
  window.setInterval(() => {
    if (!document.hidden) {
      nextLine();
    }
  }, 5200);
}

const ribbonTrack = el("div", "ribbon-track");
const RIBBON_SPRITES = ["heart", "gem", "moon", "flower", "potion"];

function paintRibbon() {
  const words = copy("ribbon");
  const half = () =>
    words.map((word, index) =>
      el("span", "", {}, [img(sprite(RIBBON_SPRITES[index % RIBBON_SPRITES.length]), ""), document.createTextNode(word)]),
    );
  // Two identical halves: the track slides by exactly one half, so the loop has no seam.
  ribbonTrack.replaceChildren(...half(), ...half(), ...half(), ...half());
}

function mountRibbon() {
  const hero = document.querySelector(".hero");
  if (!hero) {
    return;
  }
  const ribbon = el("div", "ribbon", { "aria-hidden": "true" }, [ribbonTrack]);
  hero.after(ribbon);
  paintRibbon();
}

function mountSheetPeek() {
  const main = document.querySelector("main#main");
  if (!main) {
    return;
  }
  const peek = el("div", "sheet-peek", { "aria-hidden": "true" }, [
    el("div", "peek-clip", {}, [img(face("ai-gemini"), "")]),
    el("span", "paw l"),
    el("span", "paw r"),
  ]);
  main.prepend(peek);
}

function mountRailCat() {
  const head = document.querySelector(".rail-head");
  if (!head) {
    return;
  }
  head.append(
    el("span", "rail-cat", { "aria-hidden": "true" }, [img(sprite("cat"), ""), el("span", "zz", { text: "z" })]),
  );
}

/* ------------------------------------------------------------------ */
/* Thinking stage                                                      */
/* ------------------------------------------------------------------ */

const MIN_THINK_MS = 1600;
const REACT_MS = 1000;
const OUT_MS = 420;
const STEP_MS = 460;
const CLOUD_SPRITES = ["book", "scroll", "potion", "gem", "moon", "heart"];
const REACT_FACE = {
  success: "gpt-adult-smile",
  empty: "gpt-adult-surprised",
  refusal: "gpt-adult-pout",
  error: "gpt-adult-sad",
};

const think = {
  active: false,
  painted: false,
  start: 0,
  mode: "retrieve",
  run: 0,
  timers: new Set(),
};

const stage = {};

function later(fn, ms, run = think.run) {
  const id = window.setTimeout(() => {
    think.timers.delete(id);
    if (run === think.run) {
      fn();
    }
  }, ms);
  think.timers.add(id);
  return id;
}

function every(fn, ms, run = think.run) {
  const id = window.setInterval(() => {
    if (run !== think.run) {
      window.clearInterval(id);
      return;
    }
    fn();
  }, ms);
  think.timers.add(id);
  return id;
}

function clearTimers() {
  for (const id of think.timers) {
    window.clearTimeout(id);
    window.clearInterval(id);
  }
  think.timers.clear();
}

function wavePath(amplitude, period, phase) {
  let d = "";
  for (let x = 0; x <= 700; x += 5) {
    const y = 17 + amplitude * Math.sin((x / period) * Math.PI * 2 + phase);
    d += `${x === 0 ? "M" : "L"}${x} ${y.toFixed(1)}`;
  }
  return d;
}

function buildStage() {
  const sky = el("div", "ts-sky", {}, Array.from({ length: 8 }, () => el("i")));

  const ring = el(
    "span",
    "ts-orbit-ring",
    {},
    ["book", "potion", "gem", "moon", "heart"].map((name) =>
      el("span", "ts-orbit-item", {}, [img(sprite(name), "")]),
    ),
  );
  stage.face = img(face("gpt-adult-focus"), "ts-face");
  stage.cloudSprite = img(sprite("book"), "ts-cloud-sprite");
  const cloud = el("div", "ts-cloud", {}, [
    el("div", "ts-cloud-main", {}, [stage.cloudSprite, el("span", "ts-dots", {}, [el("i"), el("i"), el("i")])]),
    el("span", "c1"),
    el("span", "c2"),
  ]);
  const figure = el("div", "ts-figure", {}, [
    el("span", "ts-halo"),
    el("div", "ts-orbit", {}, [ring]),
    stage.face,
    img(face("prop-book"), "ts-book"),
    cloud,
  ]);

  stage.kicker = el("span", "ts-kicker-text");
  stage.timer = el("span", "ts-timer", { text: "0.0s" });
  stage.title = el("h3", "ts-title");
  stage.line = el("span", "ts-line");
  stage.steps = el("ol", "ts-steps");
  stage.tokens = el("div", "ts-tokens");
  const wave = el("div", "ts-wave");
  // Period 96.6 = 13.8% of the 700-unit viewBox, matching the CSS slide so the loop is seamless.
  wave.innerHTML = `<svg viewBox="0 0 700 34" preserveAspectRatio="none" aria-hidden="true">
    <defs><linearGradient id="ts-wave-grad" x1="0" x2="1"><stop offset="0" stop-color="#ff8fb1"/>
    <stop offset=".5" stop-color="#c6a8ff"/><stop offset="1" stop-color="#8fd3ff"/></linearGradient></defs>
    <path d="${wavePath(10, 96.6, 0)}" stroke="url(#ts-wave-grad)"/>
    <path d="${wavePath(6, 48.3, 1.2)}" stroke="url(#ts-wave-grad)"/></svg>`;

  const mind = el("div", "ts-mind", {}, [
    el("p", "ts-kicker", {}, [el("span", "ts-pulse"), stage.kicker, stage.timer]),
    stage.title,
    el("p", "ts-monologue", {}, [stage.line, el("span", "ts-caret")]),
    stage.steps,
    stage.tokens,
    wave,
  ]);

  const palette = ["#ff8fb1", "#c6a8ff", "#8fd3ff", "#ffc56b", "#6fe3c8", "#ff5f8f"];
  const confetti = el(
    "div",
    "ts-confetti",
    {},
    Array.from({ length: 22 }, (_, index) => {
      const angle = (index / 22) * Math.PI * 2 + Math.random() * 0.3;
      const distance = 110 + Math.random() * 120;
      return el("i", "", {
        style: [
          `--dx:${Math.round(Math.cos(angle) * distance)}px`,
          `--dy:${Math.round(Math.sin(angle) * distance * 0.75 - 30)}px`,
          `--r:${Math.round(Math.random() * 720 - 360)}deg`,
          `--c:${palette[index % palette.length]}`,
          `--d:${Math.round(Math.random() * 120)}ms`,
        ].join(";"),
      });
    }),
  );
  stage.reactFace = img(face("gpt-adult-smile"), "ts-reaction-face");
  stage.reactText = el("p", "ts-reaction-text");
  stage.reaction = el("div", "ts-reaction", {}, [confetti, stage.reactFace, stage.reactText]);

  stage.root = el("div", "think-stage", { "aria-hidden": "true", "data-phase": "" }, [
    sky,
    figure,
    mind,
    stage.reaction,
  ]);
  paintStageCopy();
  return stage.root;
}

function paintStageCopy() {
  if (!stage.root) {
    return;
  }
  stage.kicker.textContent = copy("kicker");
  stage.title.textContent = copy("title");
  const labels = think.mode === "ask" ? copy("stepsAsk") : copy("stepsRetrieve");
  const current = [...stage.steps.children].map((li) => li.className);
  stage.steps.replaceChildren(
    ...labels.map((label, index) =>
      el("li", current[index] ?? "", {}, [el("span", "ts-step-mark"), document.createTextNode(label)]),
    ),
  );
}

function questionTokens() {
  const text = (document.querySelector("#question")?.value ?? "").trim();
  if (!text) {
    return [];
  }
  let words = [];
  if (typeof Intl !== "undefined" && Intl.Segmenter) {
    const segmenter = new Intl.Segmenter(getLang() === "zh" ? "zh" : "en", { granularity: "word" });
    words = [...segmenter.segment(text)].filter((part) => part.isWordLike).map((part) => part.segment);
  } else {
    words = text.split(/[\s,.?!，。？！、]+/);
  }
  const stop = new Set(["the", "a", "an", "of", "is", "does", "do", "when", "what", "to", "go", "on"]);
  const seen = new Set();
  return words
    .map((word) => word.trim())
    .filter((word) => word && !stop.has(word.toLowerCase()) && !seen.has(word) && seen.add(word))
    .slice(0, 7);
}

function currentMode() {
  return document.querySelector('input[name="mode"]:checked')?.value === "ask" ? "ask" : "retrieve";
}

function typeMonologue(run) {
  const lines = copy("monologue");
  let lineIndex = 0;
  const typeLine = () => {
    const text = lines[lineIndex % lines.length];
    lineIndex += 1;
    let shown = 0;
    stage.line.textContent = "";
    const typer = every(() => {
      shown += 1;
      stage.line.textContent = text.slice(0, shown);
      if (shown >= text.length) {
        window.clearInterval(typer);
        think.timers.delete(typer);
        later(typeLine, 900, run);
      }
    }, 32, run);
  };
  typeLine();
}

function startThinking() {
  clearTimers();
  think.run += 1;
  const run = think.run;
  think.active = true;
  think.painted = false;
  think.start = performance.now();
  think.mode = currentMode();

  if (reducedMotion.matches || !stage.root) {
    return;
  }

  stage.steps.replaceChildren();
  paintStageCopy();
  stage.tokens.replaceChildren(
    ...questionTokens().map((word, index) => el("span", "", { text: word, style: `--i:${index}` })),
  );
  stage.face.src = face("gpt-adult-focus");
  stage.reaction.removeAttribute("data-outcome");
  stage.host.classList.add("is-thinking");
  stage.root.dataset.phase = "";
  void stage.root.offsetWidth;
  stage.root.dataset.phase = "think";

  const steps = () => [...stage.steps.children];
  const tick = () => {
    const elapsed = performance.now() - think.start;
    stage.timer.textContent = `${(elapsed / 1000).toFixed(1)}s`;
    const active = Math.min(Math.floor(elapsed / STEP_MS), steps().length - 1);
    steps().forEach((li, index) => {
      li.className = index < active ? "is-done" : index === active ? "is-active" : "";
    });
  };
  tick();
  every(tick, 100, run);

  let spriteIndex = 0;
  every(() => {
    spriteIndex = (spriteIndex + 1) % CLOUD_SPRITES.length;
    stage.cloudSprite.src = sprite(CLOUD_SPRITES[spriteIndex]);
    stage.cloudSprite.classList.remove("is-pop");
    void stage.cloudSprite.offsetWidth;
    stage.cloudSprite.classList.add("is-pop");
  }, 720, run);

  let calm = false;
  every(() => {
    calm = !calm;
    stage.face.src = face(calm ? "gpt-adult-calm" : "gpt-adult-focus");
  }, 1700, run);

  typeMonologue(run);
}

function detectOutcome() {
  const region = stage.region;
  if (region.querySelector(".error-panel")) {
    return "error";
  }
  if (region.querySelector(".refusal-panel")) {
    return "refusal";
  }
  if (region.querySelector(".answer-panel, .result-card, .results-view")) {
    return "success";
  }
  if (region.querySelector(".empty-panel")) {
    return "empty";
  }
  return null;
}

function endStage() {
  stage.host.classList.remove("is-thinking");
  stage.root.dataset.phase = "out";
  later(() => {
    stage.root.dataset.phase = "";
  }, OUT_MS);
}

function finishThinking() {
  if (!think.active) {
    return;
  }
  const run = think.run;
  const outcome = think.painted ? detectOutcome() : null;
  const finish = () => {
    think.active = false;
  };

  if (reducedMotion.matches || !stage.root || !stage.root.dataset.phase) {
    finish();
    return;
  }
  const wait = Math.max(0, MIN_THINK_MS - (performance.now() - think.start));
  later(() => {
    [...stage.steps.children].forEach((li) => {
      li.className = "is-done";
    });
    if (!outcome) {
      finish();
      endStage();
      return;
    }
    stage.reaction.dataset.outcome = outcome;
    stage.reactFace.src = face(REACT_FACE[outcome]);
    stage.reactText.textContent = copy("react")[outcome];
    stage.root.dataset.phase = "react";
    finish();
    later(endStage, REACT_MS, run);
  }, wait, run);
}

function mountStage() {
  const region = document.querySelector("#result-state");
  if (!region) {
    return;
  }
  stage.region = region;
  stage.host = el("div", "result-host");
  region.before(stage.host);
  stage.host.append(region, buildStage());

  new MutationObserver(() => {
    const busy = region.getAttribute("aria-busy") === "true";
    if (busy && !think.active) {
      startThinking();
    } else if (busy && think.active && !stage.root.dataset.phase) {
      startThinking();
    } else if (!busy && think.active) {
      finishThinking();
    }
  }).observe(region, { attributes: true, attributeFilter: ["aria-busy"] });

  new MutationObserver(() => {
    if (think.active) {
      think.painted = true;
    }
    decorate(region);
  }).observe(region, { childList: true, subtree: true });
}

/* ------------------------------------------------------------------ */
/* Result decorations                                                  */
/* ------------------------------------------------------------------ */

const PERCH_FACES = [
  "q-deepseek-smile",
  "q-deepseek-calm",
  "q-deepseek-surprised",
  "q-deepseek-shy",
  "q-deepseek-pout",
  "q-deepseek-happy",
  "q-deepseek-serious",
  "q-deepseek-sad",
];
const STICKERS = [
  ["cat"],
  ["book", "heart"],
  ["potion", "gem"],
  ["coffee", "cake"],
  ["scroll", "flower"],
  ["moon", "candy"],
];
const PANEL_FACE = {
  "empty-panel": "gpt-adult-surprised",
  "refusal-panel": "gpt-adult-pout",
  "error-panel": "gpt-adult-sad",
};

function decorateCard(card) {
  card.dataset.deco = "1";
  // Decorations are prepended so the card's own last child keeps its `:last-child` styling.
  const extras = [el("span", "card-tape", { "aria-hidden": "true" })];
  if (card.classList.contains("source-official")) {
    // Official cards get an ai-chan-sheet chibi, never Claude or GPT.
    card.classList.add("has-perch");
    let current = pick(PERCH_FACES);
    const perchFace = img(face(current), "card-perch-face");
    const perch = el("span", "card-perch", { role: "button", tabindex: "0", "aria-label": copy("perchPoke") }, [
      perchFace,
      el("span", "card-perch-spark", { text: "✦" }),
    ]);
    // Poke: the head sits up over the paper and each poke brings a new face.
    const poke = (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!perch.classList.toggle("is-awake")) {
        return;
      }
      const next = pick(PERCH_FACES.filter((name) => name !== current));
      current = next;
      perchFace.src = face(next);
    };
    perch.addEventListener("click", poke);
    perch.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        poke(event);
      }
    });
    extras.push(perch);
  } else {
    // Every other source type gets a Sprite Catalog sticker set.
    card.classList.add("has-sticker");
    const set = pick(STICKERS);
    const sticker = el("span", "card-sticker", { "aria-hidden": "true" }, set.map((name) => img(sprite(name), "")));
    if (set[0] === "cat") {
      sticker.append(el("span", "zz", { text: "z z" }));
    }
    extras.push(sticker);
  }
  card.prepend(...extras);
}

function decorateAnswer(panel) {
  panel.dataset.deco = "1";
  panel.classList.add("has-avatar");
  panel.prepend(
    el("span", "answer-avatar", { "aria-hidden": "true" }, [img(face("gpt-adult-smile"), "")]),
    el("span", "answer-tag", { "aria-hidden": "true" }, [
      img(sprite("staff"), ""),
      el("span", "answer-tag-text", { text: copy("answerTag") }),
    ]),
  );
}

function decoratePanel(panel) {
  panel.dataset.deco = "1";
  const kind = Object.keys(PANEL_FACE).find((name) => panel.classList.contains(name));
  if (!kind) {
    return;
  }
  panel.classList.add("has-mascot");
  panel.prepend(img(face(PANEL_FACE[kind]), "panel-mascot", { "aria-hidden": "true" }));
}

function decorate(root) {
  root.querySelectorAll(".result-card:not([data-deco])").forEach(decorateCard);
  root.querySelectorAll(".answer-panel:not([data-deco])").forEach(decorateAnswer);
  root.querySelectorAll(".state-panel:not(.placeholder-panel):not([data-deco])").forEach(decoratePanel);
}

/* ------------------------------------------------------------------ */
/* Language + boot                                                     */
/* ------------------------------------------------------------------ */

function refreshCopy() {
  heroBubble.textContent = copy("hero")[heroLine % copy("hero").length];
  paintRibbon();
  paintStageCopy();
  document.querySelectorAll(".answer-tag-text").forEach((node) => {
    node.textContent = copy("answerTag");
  });
  if (heroPeek) {
    heroPeek.setAttribute("aria-label", copy(heroRoot?.classList.contains("is-stage-open") ? "peekHide" : "peekShow"));
  }
}

function boot() {
  mountHero();
  mountRibbon();
  mountSheetPeek();
  mountRailCat();
  mountStage();
  const region = document.querySelector("#result-state");
  if (region) {
    decorate(region);
  }

  document.addEventListener("click", (event) => {
    if (event.target instanceof Element && event.target.closest("[data-lang]")) {
      window.setTimeout(refreshCopy, 0);
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
