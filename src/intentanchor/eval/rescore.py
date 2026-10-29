"""用已有的逐条风险分，按新版数据的标签重算指标：标签改了不用重新推理。

    .venv-serve/bin/python -I -m intentanchor.eval.rescore data/v1.1/dev.jsonl \
        runs/2026-10-13-v1dev-stepguard/scores.jsonl --out runs/2026-10-13-v1dev-stepguard/metrics_v1.1.json

只算两边都有的样本，并打印新数据里有多少条没有分数。
加 --fpr 0.05 时，在这份数据上按目标误报率选阈值（§4.2），代替 --threshold，各分组的 F1、误报率、召回率都按这个阈值算。
加 --by-case max|last 时，按 meta.case 把整条轨迹的各步汇总成一条（§5.5.1）：max 取各步最大风险分，last 只取最后一步。
"""
import argparse
import json
from pathlib import Path

from intentanchor.eval.metrics import threshold_at_fpr
from intentanchor.eval.score_hf import print_metrics, summarize


def by_case(samples: list[dict], scores: dict[str, float], how: str) -> tuple[list[dict], dict[str, float]]:
    """每个 case 留最后一步作代表（标签是整条轨迹的标签），分数按 how 汇总。"""
    cases = {}
    for s in samples:
        cases.setdefault(s["meta"]["case"], []).append(s)
    reps, agg = [], {}
    for steps in cases.values():
        rep = max(steps, key=lambda s: s["meta"]["step"])
        agg[rep["id"]] = max(scores[s["id"]] for s in steps) if how == "max" else scores[rep["id"]]
        reps.append(rep)
    return reps, agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("scores", type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fpr", type=float, help="target FPR; picks the threshold on this data")
    p.add_argument("--by-case", choices=["max", "last"], help="aggregate trajectory steps per meta.case")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    scores = {r["id"]: r["score"] for r in map(json.loads, a.scores.read_text().splitlines())}
    samples = [s for s in map(json.loads, a.data.read_text().splitlines()) if s["label"] is not None]
    kept = [s for s in samples if s["id"] in scores]
    print(f"{len(kept)} / {len(samples)} samples have scores")
    n_missing = len(samples) - len(kept)
    if a.by_case:
        kept, scores = by_case(kept, scores, a.by_case)
        print(f"{len(kept)} cases")
    if a.fpr is not None:
        a.threshold = threshold_at_fpr([s["label"] == "BLOCK" for s in kept], [scores[s["id"]] for s in kept], a.fpr)
    metrics = {"data": str(a.data), "scores": str(a.scores), "threshold": a.threshold, "fpr_target": a.fpr,
               "by_case": a.by_case, "n": len(kept),
               "n_missing": n_missing,
               "groups": summarize(kept, [scores[s["id"]] for s in kept], a.threshold)}
    a.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print_metrics(metrics)


if __name__ == "__main__":
    main()
