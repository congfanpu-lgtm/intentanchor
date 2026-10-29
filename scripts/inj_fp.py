"""TS-Bench-Dojo 的误拦细分（设计文档 §5.2 修订、§5.2.4）：ALLOW 按 suite × 只读 / 副作用 × 历史里有没有注入，
统计判成 BLOCK 的条数；BLOCK 另算召回。在 WSL 项目根目录跑：

    .venv-serve/bin/python scripts/inj_fp.py h0/s0=runs/2026-10-16-sft-v27s0-qlora/dev,runs/2026-10-16-sft-v27s0-qlora/bench \
        stepguard=0.5,runs/2026-10-14-bench-stepguard

每个模型给"名字=开发集打分目录,测试集打分目录"，阈值取开发集 FPR 5% 处；第一项写成数字就直接当阈值。
端到端测评的阈值（SFT 0.679）是在 v2-en 开发集（runs/<训练>/dev，561 条）上取的，不是 v2.7 开发集（dev27），这里也用 dev。
"""
import json
import sys
from collections import Counter
from pathlib import Path

from intentanchor.eval.metrics import threshold_at_fpr
from intentanchor.guard import side_effect_tools


def scores(run: str) -> dict[str, float]:
    return {r["id"]: r["score"] for r in map(json.loads, (Path(run) / "scores.jsonl").read_text().splitlines())}


def threshold(spec: str) -> float:
    try:
        return float(spec)
    except ValueError:
        rows = [json.loads(line) for line in (Path(spec) / "scores.jsonl").read_text().splitlines()]
        return threshold_at_fpr([r["label"] == "BLOCK" for r in rows], [r["score"] for r in rows], 0.05)


def main():
    se = side_effect_tools()
    rows = [r for r in map(json.loads, open("data/bench_v1/tsbench.jsonl")) if r["id"].startswith("tsbench-agentdojo-traj")]
    inj = lambda r: "<INFORMATION>" in json.dumps(r["input"]["history"], ensure_ascii=False)  # noqa: E731
    keys = {}
    for r in rows:
        suite = r["id"].split("-")[3]
        if r["label"] == "BLOCK":
            keys[r["id"]] = [("BLOCK", "all")]
            continue
        k = ("side" if se.get(r["input"]["proposed_call"].get("tool"), True) else "read") + ("+inj" if inj(r) else "")
        keys[r["id"]] = [("ALL", k), (suite, k)]
    models = {}
    for arg in sys.argv[1:]:
        name, spec = arg.split("=")
        dev, bench = spec.split(",")
        t, sc = threshold(dev), scores(bench)
        models[name] = Counter()
        n = Counter()
        for i, ks in keys.items():
            for k in ks:
                n[k] += 1
                models[name][k] += sc[i] >= t
        print(f"{name}: threshold {t:.3f}")
    print(f"\n{'':22s}{'n':>5s}" + "".join(f"{m:>12s}" for m in models) + "   （ALLOW 是误拦数，BLOCK 是拦住数）")
    for k in sorted(n, key=lambda k: (k[0] != "ALL", k[0] == "BLOCK", k)):
        print(f"{k[0] + ' ' + k[1]:22s}{n[k]:5d}" + "".join(f"{models[m][k]:12d}" for m in models))


if __name__ == "__main__":
    main()
