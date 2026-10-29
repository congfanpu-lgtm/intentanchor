"""标签审计（设计文档 §5.3.6）：从数据集抽样，人工审一部分、Claude 审全部，算标签准确率。

    .venv-serve/bin/python -I -m intentanchor.data.audit sample data/v1/train.jsonl data/v1/dev.jsonl \
        --out data/v1/audit.jsonl
    .venv-serve/bin/python -I -m intentanchor.data.audit score data/v1/audit.jsonl runs/<审计>/human.jsonl \
        runs/<审计>/claude.jsonl

抽样：--n-human 条在全部带标签样本里简单随机抽（不分层，准确率直接是整体的无偏估计），人工和 Claude 都审；
另抽 --n-extra-random 条（同样简单随机）和 --n-extra-stepguard 条 StepGuard 样本（§5.3.1 要求复核转换后的标签），只给 Claude 审。
v2（§5.3.7）：--n-human 50 --n-extra-random 150 --n-extra-stepguard 0，Claude 共审 200 条，人工复核其中 50 条。
两边都审的部分用来看 Claude 和人工的一致率，决定 Claude 的结论能信几分。
审核结果每行 {"id", "reviewer", "verdict": correct|wrong|unsure, "note"}；
Claude 盲审（看不到标签）给的是 {"id", "reviewer", "label": ALLOW|BLOCK|UNSURE, "reason"}，和样本标签比对后换成 verdict。
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def sample(rows: list[dict], n_human: int, n_extra_stepguard: int, seed: int = 0, n_extra_random: int = 0) -> list[dict]:
    rng = random.Random(seed)
    human = rng.sample(rows, n_human)
    taken = {s["id"] for s in human}
    sg = [s for s in rows if s["label_source"] == "stepguard" and s["id"] not in taken]
    extra = rng.sample(sg, min(n_extra_stepguard, len(sg)))
    taken |= {s["id"] for s in extra}
    extra += rng.sample([s for s in rows if s["id"] not in taken], n_extra_random)
    return [{**s, "audit_role": "human"} for s in human] + [{**s, "audit_role": "claude"} for s in extra]


def to_verdict(v: dict, label: str) -> str:
    if "verdict" in v:
        return v["verdict"]
    return "unsure" if v["label"] == "UNSURE" else ("correct" if v["label"] == label else "wrong")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间，样本少、比例接近 1 时比正态近似可靠。"""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def accuracy(verdicts: list[str]) -> dict:
    c = Counter(verdicts)
    n = c["correct"] + c["wrong"]
    lo, hi = wilson(c["correct"], n)
    return {"n": len(verdicts), "correct": c["correct"], "wrong": c["wrong"], "unsure": c["unsure"],
            "accuracy": round(c["correct"] / n, 3) if n else None, "ci95": [round(lo, 3), round(hi, 3)]}


def score(samples: dict[str, dict], verdicts: list[dict]) -> dict:
    by_rev = defaultdict(dict)
    for v in verdicts:
        by_rev[v["reviewer"]][v["id"]] = {**v, "verdict": to_verdict(v, samples[v["id"]]["label"])}
    out = {}
    for rev, vs in by_rev.items():
        groups = defaultdict(list)
        for i, v in vs.items():
            s = samples[i]
            for k in ("all", f"role={s['audit_role']}", f"label_source={s['label_source']}", f"lang={s['lang']}",
                      f"label={s['label']}"):
                groups[k].append(v["verdict"])
        out[rev] = {k: accuracy(g) for k, g in sorted(groups.items())}
        out[rev]["wrong"] = [{"id": i, "label": samples[i]["label"], "category": samples[i]["category"],
                              "note": v.get("note") or v.get("reason")} for i, v in vs.items() if v["verdict"] == "wrong"]
    revs = sorted(by_rev)
    for i, a in enumerate(revs):  # 两两一致率：都给了明确结论（不是 unsure）的样本里，结论相同的比例
        for b in revs[i + 1:]:
            both = [(by_rev[a][x]["verdict"], by_rev[b][x]["verdict"]) for x in by_rev[a].keys() & by_rev[b].keys()]
            both = [p for p in both if "unsure" not in p]
            agree = sum(x == y for x, y in both)
            out[f"agreement:{a}~{b}"] = {"n": len(both), "agree": agree,
                                         "rate": round(agree / len(both), 3) if both else None}
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sample")
    ps.add_argument("data", nargs="+", type=Path)
    ps.add_argument("--n-human", type=int, default=100)
    ps.add_argument("--n-extra-stepguard", type=int, default=100)
    ps.add_argument("--n-extra-random", type=int, default=0)
    ps.add_argument("--out", type=Path, required=True)
    pc = sub.add_parser("score")
    pc.add_argument("audit", type=Path)
    pc.add_argument("verdicts", nargs="+", type=Path)
    a = p.parse_args()

    if a.cmd == "sample":
        rows = [s for path in a.data for s in map(json.loads, path.read_text().splitlines()) if s["label"]]
        out = sample(rows, a.n_human, a.n_extra_stepguard, n_extra_random=a.n_extra_random)
        a.out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in out))
        print(f"{len(out)} -> {a.out}", dict(Counter((s["audit_role"], s["label_source"]) for s in out)))
    else:
        samples = {s["id"]: s for s in map(json.loads, a.audit.read_text().splitlines())}
        verdicts = [json.loads(line) for path in a.verdicts for line in path.read_text().splitlines()]
        print(json.dumps(score(samples, verdicts), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
