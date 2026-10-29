"""合并中文测试集 a、b 两组盲审（设计文档 §5.3.4）。两组都同意构造标签的直接定；两组一致推翻构造标签的删掉
（生成没达到构造意图）；两组不一致或有 UNSURE 的，加上同意的里面随机抽 50 条（抽查），
写进待裁决的清单 for_user.jsonl。裁决最后由另一个 Claude 会话完成，没有人工核验（见 scripts/zh_freeze.py）。
在 WSL 项目根目录跑：.venv-serve/bin/python scripts/zh_merge.py（2026-10-27 从 runs/2026-10-21-zhtest-audit/ 移进仓库；输出仍写在那个目录）"""
import json, random
from collections import Counter
from pathlib import Path

here = Path("runs/2026-10-21-zhtest-audit")
mapping = json.load(open(here / "mapping.json"))
outs = [json.loads(l) for g in "ab" for f in sorted((here / g).glob("out*.jsonl")) for l in open(f) if l.strip()]
assert sorted(o["qid"] for o in outs) == sorted(mapping), "missing or duplicate qids"
draft = {s["id"]: s for s in map(json.loads, open("data/zh_test/draft.jsonl"))}
v = {i: {} for i in draft}
for o in outs:
    v[mapping[o["qid"]]][o["qid"][0]] = (o["label"], o["reason"])
rows = []
for i, s in draft.items():
    (a, ar), (b, br) = v[i]["a"], v[i]["b"]
    status = "agree" if a == b == s["label"] else "overturned" if a == b != "UNSURE" else "split"
    rows.append({"id": i, "label": s["label"], "category": s["category"], "a": a, "a_reason": ar, "b": b, "b_reason": br,
                 "status": status})
(here / "verdicts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
agreed = [r for r in rows if r["status"] == "agree"]
spot = {r["id"] for r in random.Random(0).sample(agreed, min(50, len(agreed)))}
user = [{**r, "why": "spot_check" if r["status"] == "agree" else "disagree"} for r in rows
        if r["status"] == "split" or r["id"] in spot]
(here / "for_user.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in user))
print(f"{len(rows)} samples: {dict(Counter(r['status'] for r in rows))}; to user: {len(user)} "
      f"({sum(r['why'] == 'disagree' for r in user)} split + {len(spot)} spot check)")
print("a vs b agreement:", sum(r["a"] == r["b"] for r in rows), "/", len(rows))
c = Counter((r["category"], r["label"], r["status"] if r["status"] != "split" else f"{r['a']}/{r['b']}") for r in rows)
for k in sorted(c):
    print(k, c[k])
