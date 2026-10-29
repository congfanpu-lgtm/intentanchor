"""合并各来源，出数据集的训练集和开发集（设计文档 §5.3.2、§5.3.6、§5.3.7）。

    v2-en（v0.5 新定义，带工具返回）：
    .venv-serve/bin/python -I -m intentanchor.data.build data/stepguard_v1.jsonl data/adojo_v1.jsonl \
        data/injecagent_v1.jsonl --out data/v2-en

    .venv-serve/bin/python -I -m intentanchor.data.build data/adojo_v0.2.jsonl data/stepguard_v0.jsonl \
        data/injecagent_v0.jsonl data/synth_v0.jsonl data/zh_trans_v0.jsonl \
        --judge runs/2026-10-13-synth-v0-judge/scores.jsonl --out data/v1
    v1.1 再加 --relabel runs/2026-10-14-label-audit/claude.jsonl runs/2026-10-14-stepguard-review/claude.jsonl \
        runs/2026-10-14-label-audit/human.jsonl

1. configs/pending_labels.yaml 里的待审核样本套用最终标签；
2. --relabel 的复核结论（人工或 Claude，§5.3.6）：和原标签不同的改标签，拿不准的放进待审核队列，
   复核过的都标 audited；译文跟原样本走。多个文件按顺序套用，后面的覆盖前面的（人工放最后）；
3. 合成样本要经大模型裁判复核：裁判和构造标签不一致、或者没有裁判分数的，放进待审核队列（review.jsonl），不进训练；
4. InjecAgent 每个攻击者用例只留 --cap-injecagent 个用户上下文（原来 17 个，太重复）；
5. 输入完全相同的去重：标签相同留一条，标签冲突全部丢掉；
6. 按组切开发集（约 --dev-pct%），同一组的样本只在一边：AgentDojo 按用户任务（拼接样本也一样），StepGuard 按用户请求，
   InjecAgent 按攻击者用例，合成场景按场景，译文跟着原样本走。测试集（留出的 suite、领域、攻击类型）以后单独建。
7. 原样本没加载的译文丢掉（原样本在转换时被剔除，译文也跟着剔除）。
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def apply_pending(samples: list[dict], overrides: dict) -> int:
    n = 0
    for s in samples:
        o = overrides.get(s["id"])
        if o:
            s.update(label=o["label"], category=o["category"], label_source=o["source"], reason=o.get("note"))
            if o["category"] == "misaligned_overreach":
                s["risk_source"] = "inherent_agent_failure"
            n += 1
    return n


RELABEL_CATEGORY = {"ALLOW": "aligned_unspecified", "BLOCK": "misaligned_overreach"}


def load_verdicts(rows: list[dict], labels: dict[str, str]) -> dict[str, str]:
    """复核结果转成 id -> ALLOW / BLOCK / UNSURE。Claude 盲审直接给标签；人工给的是对 / 错 / 拿不准，按原标签换算。
    复核过、但这一版数据里已经没有的样本跳过。"""
    flip = {"ALLOW": "BLOCK", "BLOCK": "ALLOW"}
    out = {}
    for r in rows:
        if r["id"] not in labels:
            continue
        if "label" in r:
            out[r["id"]] = r["label"]
        else:
            out[r["id"]] = {"correct": labels[r["id"]], "wrong": flip[labels[r["id"]]], "unsure": "UNSURE"}[r["verdict"]]
    return out


def apply_relabels(samples: list[dict], verdicts: dict[str, str]) -> Counter:
    stats = Counter()
    for s in samples:
        v = verdicts.get(s["id"]) or verdicts.get(s["meta"].get("translated_from"))
        if v is None:
            continue
        s["audited"] = True
        if v == s["label"]:
            stats["confirmed"] += 1
            continue
        s["meta"]["original_label"] = s["label"]
        if v == "UNSURE":
            s.update(label=None, label_source="pending")
        else:
            s.update(label=v, category=RELABEL_CATEGORY[v], label_source="audit")
        stats[f"to_{v}"] += 1
    return stats


def gate_synthetic(samples: list[dict], judge: dict[str, float]) -> int:
    """裁判不同意或没复核的合成样本改成待审核，原标签留在 meta.constructed_label。返回改了多少条。"""
    n = 0
    for s in samples:
        if s["label_source"] != "synthetic":
            continue
        score = judge.get(s["id"])
        s["meta"]["judge_score"] = score
        if score is None or (score >= 0.5) != (s["label"] == "BLOCK"):
            s["meta"]["constructed_label"] = s["label"]
            s.update(label=None, label_source="pending")
            n += 1
    return n


def cap_injecagent(samples: list[dict], k: int) -> list[dict]:
    by_case = defaultdict(list)
    for s in samples:
        if s["label_source"] == "injecagent" and s["lang"] == "en":
            by_case[s["meta"]["attacker_case"]].append(s["id"])
    rng = random.Random(0)
    keep = {i for ids in by_case.values() for i in rng.sample(sorted(ids), min(k, len(ids)))}
    return [s for s in samples if not (s["label_source"] == "injecagent" and s["lang"] == "en") or s["id"] in keep]


def _input_key(s: dict) -> str:
    i = s["input"]
    return json.dumps([s["lang"], i["user_task"], i["history"], i["proposed_call"]], sort_keys=True, ensure_ascii=False)


def dedup(samples: list[dict]) -> tuple[list[dict], Counter]:
    by_key, stats = defaultdict(list), Counter()
    for s in samples:
        by_key[_input_key(s)].append(s)
    out = []
    for group in by_key.values():
        if len({s["label"] for s in group}) > 1:
            stats["dropped_conflicting"] += len(group)
            continue
        stats["dropped_duplicate"] += len(group) - 1
        out.append(group[0])
    return out, stats


def group_key(s: dict) -> str:
    m = s.get("meta") or {}
    if s["source"] == "agentdojo":
        return f"adojo:{s['suite']}:{s['input']['user_task']}"
    if s["label_source"] == "injecagent":
        return f"injecagent:{m['attacker_case']}"
    if s["source"] == "synthetic":
        return f"synth:{m['scenario']}"
    return f"{s['source']}:{s['input']['user_task']}"


def is_dev(group: str, dev_pct: int) -> bool:
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 100 < dev_pct


def drop_orphan_translations(samples: list[dict]) -> list[dict]:
    """原样本这次没加载（比如转换时被丢掉了）的译文也不要。"""
    ids = {s["id"] for s in samples}
    return [s for s in samples if s["meta"].get("translated_from", s["id"]) in ids]


def assign_groups(samples: list[dict]) -> None:
    """在截断、去重之前算好每条的组，存进 meta.group；译文（meta.translated_from）用原样本的组。"""
    groups = {s["id"]: group_key(s) for s in samples if "translated_from" not in s["meta"]}
    for s in samples:
        orig = s["meta"].get("translated_from")
        s["meta"]["group"] = groups.get(orig) or group_key(s) if orig else groups[s["id"]]


def main():
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+", type=Path)
    p.add_argument("--judge", type=Path, help="合成样本的裁判分数（score_llm 的 scores.jsonl）；没有合成样本时不用给")
    p.add_argument("--pending-labels", type=Path, default=Path("configs/pending_labels.yaml"))
    p.add_argument("--relabel", nargs="*", type=Path, default=[], help="复核结果 jsonl（audit 的格式），后面的覆盖前面的")
    p.add_argument("--cap-injecagent", type=int, default=5)
    p.add_argument("--dev-pct", type=int, default=15)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    samples = [s for path in a.data for s in map(json.loads, path.read_text().splitlines())]
    stats = Counter(loaded=len(samples))
    samples = drop_orphan_translations(samples)
    stats["orphan_translations_dropped"] = stats["loaded"] - len(samples)
    assign_groups(samples)
    stats["pending_labels_applied"] = apply_pending(samples, yaml.safe_load(a.pending_labels.read_text()))
    labels = {s["id"]: s["label"] for s in samples}
    verdicts = {}
    for path in a.relabel:
        verdicts.update(load_verdicts([json.loads(line) for line in path.read_text().splitlines()], labels))
    stats.update({f"relabel_{k}": v for k, v in apply_relabels(samples, verdicts).items()})
    judge = {r["id"]: r["score"] for r in map(json.loads, a.judge.read_text().splitlines())} if a.judge else {}
    stats["synthetic_to_review"] = gate_synthetic(samples, judge)
    n = len(samples)
    samples = cap_injecagent(samples, a.cap_injecagent)
    stats["injecagent_capped"] = n - len(samples)
    samples, d = dedup(samples)
    stats.update(d)
    review = [s for s in samples if s["label"] is None]
    labeled = [s for s in samples if s["label"] is not None]
    for s in labeled:
        s["split"] = "dev" if is_dev(s["meta"]["group"], a.dev_pct) else "train"

    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    split_counts = {}
    for name, rows in [("train", [s for s in labeled if s["split"] == "train"]),
                       ("dev", [s for s in labeled if s["split"] == "dev"]), ("review", review)]:
        rng.shuffle(rows)
        (a.out / f"{name}.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in rows))
        split_counts[name] = {
            "n": len(rows), "by_label": dict(Counter(str(s["label"]) for s in rows)),
            "by_lang_label": {f"{k[0]}/{k[1]}": v for k, v in sorted(Counter((s["lang"], str(s["label"])) for s in rows).items())},
            "by_source_label": {f"{k[0]}/{k[1]}": v for k, v in
                                sorted(Counter((s["label_source"], str(s["label"])) for s in rows).items())},
            "by_category": dict(sorted(Counter(s["category"] for s in rows).items())),
            "by_risk_label": {f"{k[0]}/{k[1]}": v for k, v in
                              sorted(Counter((str(s.get("risk_source")), str(s["label"])) for s in rows).items())}}
    report = {"inputs": [str(x) for x in a.data], "args": {k: str(v) for k, v in vars(a).items()},
              "stats": dict(stats), "splits": split_counts}
    (a.out / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["stats"], ensure_ascii=False))
    for name, c in split_counts.items():
        print(name, c["n"], c["by_label"], c["by_lang_label"])


if __name__ == "__main__":
    main()
