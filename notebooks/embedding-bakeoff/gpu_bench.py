"""A380/OpenVINO lane benchmark: e5-base and bge-m3 on the same trilingual mini-corpus.

Per-lane throughput only — NOT a cross-model comparison (models are compared on
quality via the CPU bake-off). Also verifies GPU/CPU vector parity per model:
min cosine over all passages and identical top-5 ranking on all 16 queries.
Pooling: e5 = mean-pool + normalize; bge-m3 dense = CLS + normalize.
"""
import importlib.util, json, os, sys, time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "eval_embeds",
    Path.home() / "LocalDev/cityu/projects/2026-2027_SemA/cs5481-topic-pilot/work/embed_selection/eval_embeds.py")
ee = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ee)  # needs HF env set BEFORE transformers import inside

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import openvino as ov

CORPUS = [json.loads(l) for l in ee.CORPUS.read_text().splitlines()]
QUERIES = [{"qid": f"{g['entity']}-{lang}", "entity": g["entity"], "text": g[f"q_{lang}"]}
           for g in ee.GOLDS for lang in ("en", "zh")]

MODELS = [
    {"key": "e5-base", "name": "intfloat/multilingual-e5-base", "pool": "mean", "prefix_p": "passage: ", "prefix_q": "query: "},
    {"key": "bge-m3", "name": "BAAI/bge-m3", "pool": "cls", "prefix_p": "", "prefix_q": ""},
]

def encode_torch(model, tok, texts, pool, batch=16, device="cpu"):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i+batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            if pool == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                vec = hidden[:, 0]
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out.append(vec.cpu().numpy())
    return np.vstack(out)

def encode_ov(compiled, tok, texts, pool, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], padding=True, truncation=True, max_length=512, return_tensors="pt")
        res = compiled({k: v.numpy() for k, v in enc.items()})
        hidden = res[compiled.output(0)]
        if pool == "mean":
            mask = enc["attention_mask"].unsqueeze(-1).numpy().astype(np.float32)
            vec = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        else:
            vec = hidden[:, 0]
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        out.append(vec)
    return np.vstack(out)

def hit_mrr(P, queries, corpus):
    per = []
    for q in queries:
        sims = P @ encode_query(q["text"]) if False else None
    return per

results = {}
for m in MODELS:
    tok = AutoTokenizer.from_pretrained(m["name"])
    hf = AutoModel.from_pretrained(m["name"])
    # CPU reference vectors
    p_texts = [m["prefix_p"] + c["text"] for c in CORPUS]
    q_texts = [m["prefix_q"] + q["text"] for q in QUERIES]
    t0 = time.time()
    P_cpu = encode_torch(hf, tok, p_texts, m["pool"])
    cpu_s = time.time() - t0
    Q_cpu = encode_torch(hf, tok, q_texts, m["pool"])

    # OpenVINO convert + compile
    core = ov.Core()
    ex = tok(["example"], return_tensors="pt")
    ov_model = ov.convert_model(hf, example_input=(ex["input_ids"], ex["attention_mask"]))
    ov_model.inputs[0].get_tensor().set_names({"input_ids"})
    ov_model.inputs[1].get_tensor().set_names({"attention_mask"})
    ov_model.reshape({"input_ids": [-1, -1], "attention_mask": [-1, -1]})
    compiled = core.compile_model(ov_model, "GPU", {"INFERENCE_PRECISION_HINT": ov.Type.f32,
                                                    "PERFORMANCE_HINT": "THROUGHPUT"})
    # warmup
    encode_ov(compiled, tok, p_texts[:16], m["pool"])
    t0 = time.time()
    P_gpu = encode_ov(compiled, tok, p_texts, m["pool"])
    gpu_s = time.time() - t0
    Q_gpu = encode_ov(compiled, tok, q_texts, m["pool"])

    ntok = sum(len(tok(t).input_ids) for t in p_texts)
    cos = (P_cpu * P_gpu).sum(1)
    def mrr_hit(P, Q):
        S = P @ Q.T
        hits, mrr = 0, 0.0
        for qi, q in enumerate(QUERIES):
            top5 = np.argsort(-S[:, qi])[:5]
            rank = next((r+1 for r, i in enumerate(top5) if q["entity"] in CORPUS[int(i)]["text"]), None)
            hits += rank is not None
            mrr += 1.0/rank if rank else 0.0
        return hits/len(QUERIES), mrr/len(QUERIES)
    h_cpu, m_cpu = mrr_hit(P_cpu, Q_cpu)
    h_gpu, m_gpu = mrr_hit(P_gpu, Q_gpu)
    results[m["key"]] = {
        "corpus_tokens": ntok, "cpu_encode_s": round(cpu_s,1), "gpu_encode_s": round(gpu_s,1),
        "cpu_tok_s": round(ntok/cpu_s), "a380_tok_s": round(ntok/gpu_s),
        "speedup": round((ntok/gpu_s)/(ntok/cpu_s), 2),
        "min_cosine_cpu_gpu": float(cos.min()),
        "hit@5_cpu": h_cpu, "mrr@5_cpu": round(m_cpu,3),
        "hit@5_gpu": h_gpu, "mrr@5_gpu": round(m_gpu,3),
    }
    print(m["key"], json.dumps(results[m["key"]]))
    del hf
Path(__file__).parent.joinpath("a380_results.json").write_text(json.dumps(results, indent=2))
print("saved a380_results.json")
