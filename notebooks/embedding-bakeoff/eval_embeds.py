"""Mini selection eval: 4 open embedders on a trilingual ACGN corpus (135 passages).

Gold = 8 entities, 2 queries each (en+zh). Hit@5 = any top-5 passage containing
the entity; MRR@5 on the first hit. Same machine, same lane (torch CPU), same
texts — the only variable is the encoder.
"""
import json, os, re, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from sentence_transformers import SentenceTransformer

E5_SMALL_CACHE = str(Path.home() / "LocalDev/cityu/model-cache/2026-2027_SemA/cs5481-rag-pilot")
CORPUS = Path.home() / "OneDrive/CityU/2026-2027_SemA/CS5481_Data Engineering/audit/20261029-embed-selection/corpus/corpus.jsonl"

GOLDS = [
    {"entity": "アセンダンス", "src_lang": "ja", "q_en": "Capcom announces a new Monster Hunter game releasing in 2027",
     "q_zh": "卡普空宣布怪物猎人新作会在2027年发售的消息"},
    {"entity": "レクイエム", "src_lang": "ja", "q_en": "Resident Evil Requiem release date on Nintendo Switch 2",
     "q_zh": "生化危机安魂曲在NS2上的发售日期是什么时候"},
    {"entity": "CAPCOM CUP", "src_lang": "ja", "q_en": "Capcom Cup fighting game esports finals to be held in Japan",
     "q_zh": "卡普空杯格斗大赛首次在日本举办的决定"},
    {"entity": "gamescom", "src_lang": "ja", "q_en": "Onimusha game winning two awards at gamescom",
     "q_zh": "鬼武者新作在欧洲游戏展上获得两个奖项"},
    {"entity": "Georgie", "src_lang": "en", "q_en": "which publisher licensed the Georgie manga in English",
     "q_zh": "乔治这部漫画的英文版授权给了哪家出版社"},
    {"entity": "AniBiz", "src_lang": "en", "q_en": "anime licensing marketplace receives six million dollar investment",
     "q_zh": "动画授权交易平台获得六百万美元投资"},
    {"entity": "赛博朋克2077", "src_lang": "zh", "q_en": "Cyberpunk 2077 version 2.5 update",
     "q_zh": "赛博朋克2077的2.5版本更新上线"},
    {"entity": "潜渊症", "src_lang": "zh", "q_en": "Barotrauma Home and Harbor expansion release date",
     "q_zh": "潜渊症的新扩展包定档在什么时候"},
]

MODELS = [
    {"key": "e5-small", "name": "intfloat/multilingual-e5-small", "family": "e5", "cache_folder": E5_SMALL_CACHE},
    {"key": "e5-base", "name": "intfloat/multilingual-e5-base", "family": "e5", "cache_folder": None},
    {"key": "e5-large", "name": "intfloat/multilingual-e5-large", "family": "e5", "cache_folder": None},
    {"key": "qwen3-0.6b", "name": "Qwen/Qwen3-Embedding-0.6B", "family": "qwen3", "cache_folder": None},
    {"key": "bge-m3", "name": "BAAI/bge-m3", "family": "bge", "cache_folder": None},
]

def main():
    corpus = [json.loads(l) for l in CORPUS.read_text().splitlines()]
    queries = [{"qid": f"{g['entity']}-{lang}", "entity": g["entity"], "src_lang": g["src_lang"],
                "lang": lang, "text": g[f"q_{lang}"]}
               for g in GOLDS for lang in ("en", "zh")]
    print(f"corpus={len(corpus)} queries={len(queries)}")
    out_path = Path(__file__).parent / "selection_results.json"
    results = json.loads(out_path.read_text())["results"] if out_path.exists() else {}
    import sys
    only = set(sys.argv[1:]) or None
    for spec in MODELS:
        if spec["key"] in results and (only is None or spec["key"] not in only):
            print(f"skip cached: {spec['key']}")
            continue
        t0 = time.time()
        kw = {"device": "cpu"}
        if spec["cache_folder"]:
            kw["cache_folder"] = spec["cache_folder"]
        m = SentenceTransformer(spec["name"], **kw)
        load_s = time.time() - t0
        dim = m.get_embedding_dimension()
        prefix_p = "passage: " if spec["family"] == "e5" else ""
        prefix_q = "query: " if spec["family"] == "e5" else None  # bge-m3: no prefixes
        texts = [prefix_p + c["text"] for c in corpus]
        t0 = time.time()
        P = m.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        enc_s = time.time() - t0
        from transformers import AutoTokenizer
        tok_kw = {"cache_dir": spec["cache_folder"]} if spec["cache_folder"] else {}
        tok = AutoTokenizer.from_pretrained(spec["name"], **tok_kw)
        ntok = sum(len(tok(t).input_ids) for t in texts)
        tps = ntok / enc_s
        per_q, hits = [], []
        for q in queries:
            if prefix_q is not None:
                qtext = prefix_q + q["text"]
                qv = m.encode([qtext], normalize_embeddings=True)[0]
            else:
                try:
                    qv = m.encode([q["text"]], prompt_name="query", normalize_embeddings=True)[0]
                except Exception:
                    qv = m.encode([q["text"]], normalize_embeddings=True)[0]
            sims = P @ qv
            top5 = np.argsort(-sims)[:5]
            hit_rank = next((r + 1 for r, i in enumerate(top5)
                             if q["entity"] in corpus[int(i)]["text"]), None)
            per_q.append({"qid": q["qid"], "entity": q["entity"], "q_lang": q["lang"],
                          "src_lang": q["src_lang"],
                          "hit@5": hit_rank is not None,
                          "mrr": 1.0 / hit_rank if hit_rank else 0.0,
                          "top1_id": corpus[int(top5[0])]["id"]})
            if hit_rank is not None:
                hits.append(hit_rank)
        by_qlang = {l: {"n": sum(1 for r in per_q if r["q_lang"] == l),
                        "hit@5": sum(r["hit@5"] for r in per_q if r["q_lang"] == l) / max(1, sum(1 for r in per_q if r["q_lang"] == l)),
                        "mrr@5": sum(r["mrr"] for r in per_q if r["q_lang"] == l) / max(1, sum(1 for r in per_q if r["q_lang"] == l))}
                    for l in ("en", "zh")}
        results[spec["key"]] = {
            "model": spec["name"], "dim": dim, "load_s": round(load_s, 1),
            "corpus_encode_s": round(enc_s, 1), "corpus_tokens": ntok,
            "tokens_per_s_cpu": round(tps), "items_per_s_cpu": round(len(corpus) / enc_s, 1),
            "hit@5": sum(r["hit@5"] for r in per_q) / len(per_q),
            "mrr@5": sum(r["mrr"] for r in per_q) / len(per_q),
            "by_query_lang": by_qlang, "per_query": per_q,
        }
        print(f"{spec['key']}: dim={dim} {tps} tok/s | hit@5={results[spec['key']]['hit@5']:.2f} "
              f"mrr@5={results[spec['key']]['mrr@5']:.3f} | zh-q hit={by_qlang['zh']['hit@5']:.2f} "
              f"mrr={by_qlang['zh']['mrr@5']:.3f}")
        del m, P
    out_path.write_text(json.dumps({"n_corpus": len(corpus), "n_queries": len(queries),
                               "golds": GOLDS, "results": results}, ensure_ascii=False, indent=2))
    print("saved:", out_path)

if __name__ == "__main__":
    main()
