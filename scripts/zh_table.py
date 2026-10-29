"""中文测试集测评表（设计文档 §5.3.4，RQ4）。在 WSL 项目根目录跑：

    .venv-serve/bin/python scripts/zh_table.py \
        h0/s0=runs/2026-10-16-sft-v27s0-qlora/dev,runs/2026-10-25-zhtest-score/h0s0 ... \
        stepguard=runs/2026-10-14-v2endev-stepguard,runs/2026-10-25-zhtest-score/stepguard --compare h2:h0

每个模型给"名字=开发集打分目录,中文测试集打分目录"，阈值取开发集 FPR 5% 处（同 inj_fp.py）。
打分是在草稿（430 条）上做的，风险分和标签无关；这里只取冻结集里的样本，用冻结后的标签。
ALLOW 行是误拦率，BLOCK 行是召回（%）；名字写成"组/种子"的，再加一列组均值。
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from inj_fp import scores, threshold
from intentanchor.eval.metrics import auroc
from mcnemar import mcnemar

LANG = {"zh": "中文", "en": "英文", "mix": "中英混写"}


def group(r: dict) -> str:
    if r["label"] == "BLOCK":
        return f"BLOCK 注入攻击（{LANG[r['id'].split('-')[-2]]}）" if r["category"] == "misaligned_injection" else "BLOCK 越权"
    if r["category"] == "aligned_after_compromise":
        return "ALLOW 被带偏后回到任务"
    return "ALLOW 有注入但没理" if "-inj-" in r["id"] else "ALLOW 干净历史"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="名字=开发集打分目录,中文测试集打分目录")
    p.add_argument("--data", default="data/zh_test/v1.jsonl")
    p.add_argument("--compare", nargs="*", default=[], help="组A:组B，同种子配对 McNemar；组B 也可以是不带种子的模型名")
    a = p.parse_args()
    rows = [json.loads(line) for line in open(a.data)]
    y = [r["label"] == "BLOCK" for r in rows]
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['label']} 全部"].append(r["id"])
        groups[group(r)].append(r["id"])

    flag, auc = {}, {}
    for m in a.models:
        name, spec = m.split("=")
        dev, run = spec.split(",")
        t, s = threshold(dev), scores(run)
        flag[name] = {r["id"]: s[r["id"]] >= t for r in rows}
        auc[name] = auroc(y, [s[r["id"]] for r in rows])
        print(f"{name}: 阈值 {t:.3f}")

    seeds = defaultdict(list)
    for name in flag:
        if "/" in name:
            seeds[name.split("/")[0]].append(name)
    cols = list(flag) + [f"{g}均值" for g, ns in seeds.items() if len(ns) > 1]

    def cell(col, ids):
        names = seeds[col[:-2]] if col.endswith("均值") else [col]
        return np.mean([100 * np.mean([flag[n][i] for i in ids]) for n in names])

    print(f"\n{a.data}（{len(rows)} 条）\n{'':24s}{'n':>4s}" + "".join(f"{c:>11s}" for c in cols))
    print(f"{'AUROC':24s}{len(rows):4d}" + "".join(
        f"{np.mean([auc[n] for n in seeds[c[:-2]]]) if c.endswith('均值') else auc[c]:11.3f}" for c in cols))
    for k in sorted(groups, key=lambda k: (k[:5], "全部" not in k, k)):
        print(f"{k:24s}{len(groups[k]):4d}" + "".join(f"{cell(c, groups[k]):11.1f}" for c in cols))

    for pair in a.compare:
        ga, gb = pair.split(":")
        for label in ("ALLOW", "BLOCK"):
            ids = groups[f"{label} 全部"]
            cells = []
            for na in seeds[ga]:
                nb = gb if gb in flag else f"{gb}/{na.split('/')[1]}"  # 组B没有种子（如 stepguard）就每个种子都和它比
                if nb not in flag:
                    continue
                only_a = sum(flag[na][i] and not flag[nb][i] for i in ids)
                only_b = sum(flag[nb][i] and not flag[na][i] for i in ids)
                cells.append(f"{na.split('/')[1]}: {sum(flag[na][i] for i in ids)} vs {sum(flag[nb][i] for i in ids)} "
                             f"(p={mcnemar(only_a, only_b):.3g})")
            print(f"{ga} vs {gb} | {label} n={len(ids)} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
