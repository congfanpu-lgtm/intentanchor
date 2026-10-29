"""和 score_hf 一样的打分，换成 vLLM：风险分 = 第一个生成位置上 P(BLOCK) / (P(ALLOW) + P(BLOCK))。

    VLLM_USE_FLASHINFER_SAMPLER=0 .venv-serve/bin/python -m intentanchor.eval.score_vllm data/v2-en/dev.jsonl \
        --adapter runs/<训练>/best --out runs/<训练>/dev

score_hf 一条条做前向，测 5 个静态测试集（约 1,300 万 token）要 1.2 小时；vLLM 批量预填充快得多，输出格式和 score_hf 相同。
LoRA 权重加载到 bf16 基座上（和 score_hf、上线部署一致）。两个标签 token 都不在前 20 个里的，记 0.5 分（按 score_hf 的经验，
标签 token 合计概率几乎都接近 1，这种情况极少），条数写进 metrics 的 label_tokens_missing。
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from intentanchor.eval.score_hf import print_metrics, summarize
from intentanchor.prompt import LABELS, build_messages

TOP_LOGPROBS = 20


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+", type=Path)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--adapter", type=Path, help="LoRA 权重目录")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--variant", choices=["A", "B"], default="B")
    a = p.parse_args()

    samples = [s for path in a.data for line in path.read_text().splitlines()
               if (s := json.loads(line))["label"] is not None]
    llm = LLM(a.model, max_model_len=6144, gpu_memory_utilization=0.85, enable_prefix_caching=True,
              enable_lora=bool(a.adapter), max_lora_rank=16)
    tok = llm.get_tokenizer()
    label_ids = []
    for w in LABELS:
        t = tok.encode(w, add_special_tokens=False)
        assert len(t) == 1, f"{w} is not a single token: {t}"
        label_ids.append(t[0])
    texts = [tok.apply_chat_template(build_messages(s["input"], a.variant), tokenize=False, add_generation_prompt=True)
             for s in samples]
    lora = LoRARequest("guard", 1, str(a.adapter)) if a.adapter else None
    t0 = time.time()
    outs = llm.generate(texts, SamplingParams(temperature=0, max_tokens=1, logprobs=TOP_LOGPROBS), lora_request=lora)
    elapsed = time.time() - t0

    scores, mass, top1, lens, missing = [], [], [], [], 0
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        pa, pb = (math.exp(lp[i].logprob) if i in lp else 0.0 for i in label_ids)
        if pa + pb == 0:
            missing += 1
        scores.append(pb / (pa + pb) if pa + pb else 0.5)
        mass.append(pa + pb)
        top1.append(tok.decode(max(lp, key=lambda i: lp[i].logprob)))
        lens.append(len(o.prompt_token_ids))

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "scores.jsonl").open("w") as f:
        for s, sc, m, t1, ln in zip(samples, scores, mass, top1, lens):
            f.write(json.dumps({"id": s["id"], "source": s["source"], "label": s["label"], "category": s["category"],
                                "risk_source": (s.get("meta") or {}).get("risk_source"), "score": sc,
                                "label_token_mass": m, "top1": t1, "n_tokens": ln}, ensure_ascii=False) + "\n")
    metrics = {"model": a.model, "adapter": str(a.adapter) if a.adapter else None, "engine": "vllm",
               "data": [str(x) for x in a.data], "threshold": a.threshold, "n": len(samples),
               "seconds": round(elapsed, 1), "label_tokens_missing": missing,
               "n_tokens": {"mean": float(np.mean(lens)), "max": int(max(lens))},
               "label_token_mass": {"mean": float(np.mean(mass)), "share_below_0.5": float(np.mean(np.array(mass) < 0.5))},
               "top1_tokens": dict(Counter(top1).most_common(5)),
               "groups": summarize(samples, scores, a.threshold)}
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print_metrics(metrics)


if __name__ == "__main__":
    main()
