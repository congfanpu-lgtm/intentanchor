"""守卫服务压测（设计文档 §5.6）：不同并发下的 P50、P99 延迟和吞吐。

    # 先按 intentanchor/serve.py 开头的说明起 vLLM 和服务
    .venv-serve/bin/python scripts/bench_serve.py data/v2-en/dev.jsonl --out runs/<日期>-serve-bench/intentanchor.json
    # StepGuard：直接打 18000 端口上的 vLLM（同一张卡、同样的设置），它要先写分析再判定，所以少测几条
    .venv-serve/bin/python scripts/bench_serve.py data/v2-en/dev.jsonl --stepguard --n 40 --out runs/<日期>-serve-bench/stepguard.json

每个并发档位用不重叠的样本，避免 vLLM 前缀缓存把整条输入都缓存住、测出偏快的数字（共用的系统提示词照常缓存，和线上一样）。
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

from intentanchor.client import check
from intentanchor.guard import stepguard_scorer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--n", type=int, default=150, help="每个并发档位的请求数")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--url", default="http://localhost:18080/v1/check")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--stepguard", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    inputs = [json.loads(l)["input"] for l in a.data.read_text().splitlines()]
    inputs = random.Random(0).sample(inputs, a.n * len(a.concurrency) + 1)
    if a.stepguard:
        call = stepguard_scorer(OpenAI(base_url="http://localhost:18000/v1", api_key="EMPTY"))
    else:
        def call(inp):
            r = check(inp, a.url, explain=a.explain, timeout=120)
            assert "error" not in r, r

    def timed(inp):
        t = time.perf_counter()
        call(inp)
        return (time.perf_counter() - t) * 1000

    timed(inputs.pop())  # 预热
    rows = []
    for i, c in enumerate(a.concurrency):
        batch = inputs[i * a.n:(i + 1) * a.n]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            lat = np.array(list(ex.map(timed, batch)))
        wall = time.perf_counter() - t0
        rows.append({"concurrency": c, "n": len(batch), "p50_ms": round(float(np.percentile(lat, 50)), 1),
                     "p99_ms": round(float(np.percentile(lat, 99)), 1), "mean_ms": round(float(lat.mean()), 1),
                     "throughput_rps": round(len(batch) / wall, 2)})
        print(rows[-1], flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"target": "stepguard" if a.stepguard else a.url, "explain": a.explain,
                                     "data": str(a.data), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
