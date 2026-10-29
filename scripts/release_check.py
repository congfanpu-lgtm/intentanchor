"""发布前检查（设计文档 §5.7 "发布前检查"）：测试集和训练集查重；测试集、训练集查个人信息。在 WSL 项目根目录跑（约 5 分钟）：

    .venv-serve/bin/python -W ignore scripts/release_check.py h2/s0=runs/2026-10-19-sft-h2s0-qlora/bench \
        h2/s1=runs/2026-10-19-sft-h2s1-qlora/bench h2/s2=runs/2026-10-21-sft-h2s2-qlora/bench \
        h0/s0=runs/2026-10-16-sft-v27s0-qlora/bench stepguard=runs/2026-10-14-bench-stepguard \
        > runs/2026-10-27-release-check.txt

查重：训练集是最终模型的 data/h2/train.jsonl。每条样本取"用户请求 + 待审查调用"：
- 完全相同：整条输入（含历史）相同；用户请求和调用相同；只有用户请求相同；
- 近似：字符 3-gram TF-IDF 余弦（哈希到 2^18 维），每条测试样本取训练集里最像的那条。
给了"名字=测试集打分目录"，整条轨迹的测试集再按"有没有一步的用户请求和训练集相同"分两半，各算按最大值汇总的 AUROC。
个人信息：身份证号（校验位对得上的）、手机号、邮箱、银行卡号（Luhn 校验对得上的），外加真实品牌名。
"""
import json
import math
import re
import statistics
import sys
import zlib
from collections import Counter
from pathlib import Path

import torch

from intentanchor.eval.metrics import auroc
from intentanchor.eval.rescore import by_case

TRAIN = Path("data/h2/train.jsonl")
TESTS = [Path("data/zh_test/v1.jsonl"), *sorted(Path("data/bench_v1").glob("*.jsonl"))]
DIM = 1 << 18
PII = {
    "身份证": re.compile(r"(?<![\dA-Za-z])\d{17}[\dXx](?![\dA-Za-z])"),
    "手机号": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "邮箱": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "银行卡": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
}
BRANDS = ["支付宝", "微信", "淘宝", "天猫", "京东", "美团", "饿了么", "滴滴", "高德", "百度", "抖音", "拼多多",
          "携程", "12306", "钉钉", "飞书", "招商银行", "工商银行", "建设银行", "农业银行", "中国银行", "顺丰", "小红书", "腾讯"]


def id_ok(s: str) -> bool:
    w = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    return "10X98765432"[sum(int(c) * k for c, k in zip(s, w)) % 11] == s[-1].upper()


def luhn(s: str) -> bool:
    d = [int(c) for c in reversed(s)]
    return sum(d[0::2] + [sum(divmod(2 * x, 10)) for x in d[1::2]]) % 10 == 0


VALID = {"身份证": id_ok, "银行卡": luhn}


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def task(s: dict) -> str:
    return (s["input"]["user_task"] or "").strip()


def text(s: dict) -> str:
    return task(s) + "\n" + json.dumps(s["input"]["proposed_call"], ensure_ascii=False, sort_keys=True)


def key(x) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def grams(t: str) -> Counter:
    t = re.sub(r"\s+", " ", t.lower())
    return Counter(zlib.crc32(t[k:k + 3].encode()) % DIM for k in range(len(t) - 2))


def tfidf(docs: list[Counter], idf: dict) -> torch.Tensor:
    rows, cols, vals = [], [], []
    for r, g in enumerate(docs):
        w = {c: n * idf[c] for c, n in g.items()}
        norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
        for c, v in w.items():
            rows.append(r), cols.append(c), vals.append(v / norm)
    return torch.sparse_coo_tensor([rows, cols], vals, (len(docs), DIM), dtype=torch.float32).coalesce()


def group(p: Path, s: dict) -> str:
    m = s.get("meta") or {}
    return p.stem + (f"/{m['subset']}" + (f"/{m['suite']}" if m.get("suite") else "") if p.stem == "tsbench" else "")


def pii(rows: list[dict]) -> dict[str, Counter]:
    found = {k: Counter() for k in [*PII, "品牌"]}
    for s in rows:
        t = key({"input": s["input"], "reason": s.get("reason")})
        for k, rx in PII.items():
            found[k].update(v for v in rx.findall(t) if VALID.get(k, bool)(v))
        found["品牌"].update(b for b in BRANDS if b in t)
    return found


def main():
    train = load(TRAIN)
    tests = {p: load(p) for p in TESTS}
    full = {key(s["input"]) for s in train}
    tc = {text(s) for s in train}
    tasks = {task(s) for s in train} - {""}
    docs = {id(s): grams(text(s)) for rows in [train, *tests.values()] for s in rows}
    df = Counter(c for g in docs.values() for c in g)
    idf = {c: math.log((1 + len(docs)) / (1 + n)) + 1 for c, n in df.items()}
    tm = tfidf([docs[id(s)] for s in train], idf)

    print(f"训练集 {TRAIN}：{len(train)} 条\n")
    print(f"{'测试集':34}{'条数':>6}{'整条相同':>8}{'请求+调用相同':>12}{'请求相同':>8}{'≥0.95':>7}{'≥0.9':>7}{'≥0.8':>7}{'中位数':>8}")
    for p, rows in tests.items():
        sims, near = [], []
        for i in range(0, len(rows), 512):
            chunk = rows[i:i + 512]
            v, a = torch.sparse.mm(tm, tfidf([docs[id(s)] for s in chunk], idf).to_dense().T).max(0)
            sims += v.tolist()
            near += [train[j] for j in a.tolist()]
        by = {}
        for s, v in zip(rows, sims):
            by.setdefault(group(p, s), []).append((s, v))
        for g, items in sorted(by.items()):
            vs = [v for _, v in items]
            print(f"{g:34}{len(items):>6}{sum(key(s['input']) in full for s, _ in items):>8}"
                  f"{sum(text(s) in tc for s, _ in items):>12}{sum(task(s) in tasks for s, _ in items):>8}" + "".join(f"{sum(v >= t for v in vs):>7}" for t in (0.95, 0.9, 0.8))
                  + f"{statistics.median(vs):>8.3f}")
        if p.stem != "tsbench":  # tsbench 的 workspace、travel 是训练环境，像是预期内的
            print("  和训练集最像的 3 条：")
            for v, s, n in sorted(zip(sims, rows, near), key=lambda x: -x[0])[:3]:
                print(f"  {v:.3f}  {s['id']}  ↔  {n['id']}\n    测试：{text(s)[:100]!r}\n    训练：{text(n)[:100]!r}")

    models = dict(m.split("=", 1) for m in sys.argv[1:])
    if models:
        print("\n整条轨迹的测试集按用户请求是否在训练集里出现分两半，按最大值汇总的 AUROC：")
        sc = {name: {r["id"]: r["score"] for r in load(Path(run) / "scores.jsonl")} for name, run in models.items()}
        for p, rows in tests.items():
            if "meta" not in rows[0] or "case" not in rows[0]["meta"] or p.stem == "tsbench":
                continue
            seen = {}
            for s in rows:
                seen[s["meta"]["case"]] = seen.get(s["meta"]["case"], False) or task(s) in tasks
            if not any(seen.values()):
                continue
            for part in (True, False):
                d = [s for s in rows if seen[s["meta"]["case"]] == part]
                cells = []
                for name, s in sc.items():
                    reps, agg = by_case(d, s, "max")
                    cells.append(f"{name} {auroc([x['label'] == 'BLOCK' for x in reps], [agg[x['id']] for x in reps]):.3f}")
                nb = sum(x["label"] == "BLOCK" for x in reps)
                print(f"  {p.stem:12} {'请求在训练集里' if part else '其余':8} {len(reps):>5} 个案例（BLOCK {nb}）  " + " | ".join(cells))

    for name, rows in [("中文测试集", tests[TESTS[0]]), ("训练集", train)]:
        print(f"\n{name}的个人信息扫描（{len(rows)} 条）：")
        for k, c in pii(rows).items():
            print(f"  {k}：{sum(c.values())} 处，{len(c)} 个不同值" + (f"；例：{', '.join(v for v, _ in c.most_common(8))}" if c else ""))


if __name__ == "__main__":
    main()
