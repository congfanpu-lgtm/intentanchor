"""按测试集 × 动作类型（回复 / 工具调用）统计 ALLOW 样本里的误拦数（score>=0.5，所有步）。在 WSL 项目根目录跑：

    .venv-serve/bin/python scripts/fp_kind.py stepguard=runs/2026-10-14-bench-stepguard v22=runs/2026-10-15-sft-v22-qlora/bench
"""
import json, sys, glob, collections
kind, bench = {}, {}
for f in glob.glob("data/bench_v1/*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if d["label"] != "ALLOW":
            continue
        pc = d["input"]["proposed_call"]
        kind[d["id"]] = "reply" if "response" in pc and "name" not in pc else "call"
        b = d["id"].split("-")[0]
        bench[d["id"]] = "-".join(d["id"].split("-")[:2]) if b == "tsbench" else b
rows = collections.defaultdict(dict)
for arg in sys.argv[1:]:
    name, run = arg.split("=")
    c = collections.Counter()
    for l in open(f"{run}/scores.jsonl"):
        s = json.loads(l)
        if s["id"] in kind:
            key = (bench[s["id"]], kind[s["id"]])
            c[key + ("n",)] += 1
            c[key + ("fp",)] += s["score"] >= 0.5
    for (b, k, t), v in c.items():
        rows[(b, k)][(name, t)] = v
names = [a.split("=")[0] for a in sys.argv[1:]]
print(f"{'set':22}{'kind':6}{'n':>6}  " + "".join(f"{n:>12}" for n in names) + "   (FP 数)")
for (b, k) in sorted(rows):
    r = rows[(b, k)]
    print(f"{b:22}{k:6}{r[(names[0], 'n')]:>6}  " + "".join(f"{r.get((n, 'fp'), 0):>12}" for n in names))
