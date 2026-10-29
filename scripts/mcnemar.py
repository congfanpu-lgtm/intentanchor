"""同种子配对的 McNemar 精确检验（设计文档 §5.2.4）：两组模型在 TS-Bench AgentDojo 子集上逐条比较是否判 BLOCK。
README 和设计文档里 H2 对 H0、H1、H2r2 的 p 值由它算出。在 WSL 项目根目录跑：

    .venv-serve/bin/python scripts/mcnemar.py \
        h0/s0=runs/2026-10-16-sft-v27s0-qlora h2/s0=runs/2026-10-19-sft-h2s0-qlora ... --compare h2:h0 h2r2:h2

每个模型给"组/种子=训练目录"，目录下要有 dev（v2-en 开发集打分，阈值取 FPR 5% 处，同 inj_fp.py）和 bench（测试集打分）。
ALLOW 行比误拦（越少越好），BLOCK 行比拦住的攻击（越多越好）；只比两组都有的种子。
"""
import argparse
import json
from math import comb

from inj_fp import scores, threshold

SETS = {"TS-Dojo": ("tsbench-agentdojo-traj",),
        "TS-Dojo banking+slack": ("tsbench-agentdojo-traj-banking", "tsbench-agentdojo-traj-slack")}


def mcnemar(b: int, c: int) -> float:
    """双侧精确检验：b、c 是两个模型判得不一样的两类条数。"""
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="组/种子=训练目录")
    p.add_argument("--compare", nargs="+", required=True, help="组A:组B")
    a = p.parse_args()
    rows = [r for r in map(json.loads, open("data/bench_v1/tsbench.jsonl")) if r["id"].startswith(SETS["TS-Dojo"])]
    flag = {}
    for m in a.models:
        name, run = m.split("=")
        thr = threshold(f"{run}/dev")
        flag[name] = {i: v >= thr for i, v in scores(f"{run}/bench").items()}
    for pair in a.compare:
        ga, gb = pair.split(":")
        seeds = sorted(n.split("/")[1] for n in flag if n.startswith(ga + "/") and f"{gb}/{n.split('/')[1]}" in flag)
        for set_name, prefix in SETS.items():
            for label in ("ALLOW", "BLOCK"):
                ids = [r["id"] for r in rows if r["id"].startswith(prefix) and r["label"] == label]
                cells = []
                for s in seeds:
                    x, z = flag[f"{ga}/{s}"], flag[f"{gb}/{s}"]
                    only_a = sum(x[i] and not z[i] for i in ids)
                    only_b = sum(z[i] and not x[i] for i in ids)
                    cells.append(f"{s}: {sum(x[i] for i in ids)} vs {sum(z[i] for i in ids)} "
                                 f"(only {ga} {only_a} / only {gb} {only_b}, p={mcnemar(only_a, only_b):.3g})")
                print(f"{ga} vs {gb} | {set_name} {label} n={len(ids)} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
