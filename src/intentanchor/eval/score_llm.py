"""大模型裁判（设计文档 §5.5.3）：同一份守卫提示词交给百炼上的大模型，风险分取第一个 token 上 BLOCK / (ALLOW + BLOCK) 的概率。

    set -a && . ./.env && set +a
    # 基线：跳过待审核样本，只取第一个 token，算指标
    .venv-serve/bin/python -m intentanchor.eval.score_llm data/x.jsonl --model qwen3.8-max --out runs/2026-10-13-judge-x
    # 给待审核样本打标签：只跑 label 为空的样本，并生成第二行理由
    .venv-serve/bin/python -m intentanchor.eval.score_llm data/x.jsonl --pending --out runs/2026-10-13-pending-judge

关思考、temperature 0。结果逐条追加到 scores.jsonl，已有的 id 跳过，可断点续跑，重跑不重复花钱。
"""
import argparse
import json
import math
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import openai

from intentanchor.eval.score_hf import print_metrics, summarize
from intentanchor.prompt import LABELS, build_messages

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PRICE_CNY_PER_M = {"qwen3.8-max": (12, 36), "qwen3.7-plus": (2, 8), "qwen3.7-flash": (0.2, 0.8)}  # 原价，2026-10 官方文档


def judge(client: openai.OpenAI, model: str, inp: dict, reason: bool) -> dict:
    r = client.chat.completions.create(model=model, messages=build_messages(inp), temperature=0,
                                       max_tokens=80 if reason else 1, logprobs=True, top_logprobs=5,
                                       extra_body={"enable_thinking": False})
    ch = r.choices[0]
    p = Counter()
    for t in ch.logprobs.content[0].top_logprobs:
        p[t.token.strip()] += math.exp(t.logprob)
    allow, block = (p[w] for w in LABELS)
    lines = (ch.message.content or "").strip().split("\n", 1)
    return {"score": block / (allow + block) if allow + block else None, "label_token_mass": allow + block,
            "judge_label": lines[0].strip(), "reason": lines[1].strip() if len(lines) > 1 else None,
            "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+", type=Path)
    p.add_argument("--model", default="qwen3.8-max")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pending", action="store_true", help="只跑待审核样本（label 为空），并生成理由")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    samples = [s for path in a.data for line in path.read_text().splitlines()
               if ((s := json.loads(line))["label"] is None) == a.pending]
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "scores.jsonl"
    done = {r["id"]: r for r in map(json.loads, path.read_text().splitlines() if path.exists() else [])}
    todo = [s for s in samples if s["id"] not in done]
    print(f"{len(samples)} samples, {len(done)} already scored, running {len(todo)}", flush=True)

    client = openai.OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL, max_retries=3, timeout=60)
    lock, t0 = threading.Lock(), time.time()
    with ThreadPoolExecutor(a.workers) as pool, path.open("a") as f:
        futures = {pool.submit(judge, client, a.model, s["input"], a.pending): s for s in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            try:
                row = {"id": s["id"], "source": s["source"], "label": s["label"], "category": s["category"],
                       "risk_source": (s.get("meta") or {}).get("risk_source"), "model": a.model, **fut.result()}
            except openai.APIError as e:  # 单条失败（如内容审核拦截）不影响整批，下次重跑会补
                print(f"  {s['id']}: {type(e).__name__} {str(e)[:150]}", flush=True)
                continue
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            done[s["id"]] = row
            if n % 100 == 0:
                print(f"  [{n}/{len(todo)}]", flush=True)

    rows = [done[s["id"]] for s in samples if s["id"] in done]
    tok = Counter()
    for r in rows:
        tok.update(prompt=r["prompt_tokens"], completion=r["completion_tokens"])
    price = PRICE_CNY_PER_M.get(a.model)
    metrics = {"model": a.model, "data": [str(x) for x in a.data], "threshold": a.threshold, "n": len(rows),
               "missing": len(samples) - len(rows), "seconds_this_run": round(time.time() - t0, 1), **tok,
               "cost_cny": round((tok["prompt"] * price[0] + tok["completion"] * price[1]) / 1e6, 3) if price else None,
               "no_score": sum(r["score"] is None for r in rows),
               "label_token_mass": {"mean": float(np.mean([r["label_token_mass"] for r in rows]))} if rows else None,
               "judge_labels": dict(Counter(r["judge_label"] for r in rows))}
    if not a.pending:
        scored = [(s, done[s["id"]]["score"]) for s in samples if done.get(s["id"], {}).get("score") is not None]
        metrics["groups"] = summarize([s for s, _ in scored], [sc for _, sc in scored], a.threshold)
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    if a.pending:
        print(json.dumps(metrics, ensure_ascii=False))
    else:
        print_metrics(metrics)


if __name__ == "__main__":
    main()
