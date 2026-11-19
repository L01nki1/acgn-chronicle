import test from "node:test";
import assert from "node:assert/strict";
import { LANGS, STRINGS } from "../../src/acgn_rag/web/static/i18n.mjs";

test("both dictionaries define exactly the same keys", () => {
  const keyLists = LANGS.map((lang) => Object.keys(STRINGS[lang]).sort());
  assert.deepEqual(keyLists[1], keyLists[0]);
});

test("every dictionary value is a non-empty string", () => {
  for (const lang of LANGS) {
    for (const [key, value] of Object.entries(STRINGS[lang])) {
      assert.equal(typeof value, "string", `${lang}/${key} must be a string`);
      assert.ok(value.trim().length > 0, `${lang}/${key} must not be empty`);
    }
  }
});

test("the strings the frontend contract relies on exist in both languages", () => {
  assert.equal(STRINGS.zh["citation.open"], "打开原文");
  assert.equal(STRINGS.en["citation.open"], "Open source");
  assert.ok(
    STRINGS.zh["form.error.noSource"].includes("全部不选不等于不筛选"),
    "the no-source-type FormError keeps its contract wording",
  );
});
