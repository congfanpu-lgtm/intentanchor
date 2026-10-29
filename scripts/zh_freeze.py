"""冻结中文测试集（设计文档 §5.3.4）：草稿 + 两组盲审结果 + 分歧裁决 → data/zh_test/v1.jsonl，打印 sha256。
- 两组子 agent 都同意构造标签的：直接定；
- 两组一致推翻构造标签的：删掉（生成没达到构造意图）；
- 两组不一致的：按裁决（decisions.jsonl）改标签，或者删掉（DROP）。
裁决和 50 条抽查是另一个 Claude 会话做的，没有人工核验；2026-10-27 决定不再补做人工核验。
在 WSL 项目根目录跑：.venv-serve/bin/python scripts/zh_freeze.py
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

AUDIT = Path("runs/2026-10-21-zhtest-audit")  # verdicts.jsonl 由 scripts/zh_merge.py 写到这里；decisions.jsonl 是裁决和抽查结果
OUT = Path("data/zh_test/v1.jsonl")


def main():
    draft = [json.loads(line) for line in open("data/zh_test/draft.jsonl")]
    status = {r["id"]: r["status"] for r in map(json.loads, open(AUDIT / "verdicts.jsonl"))}
    dec = {r["id"]: r for r in map(json.loads, open(AUDIT / "decisions.jsonl"))}
    label = {s["id"]: s["label"] for s in draft}
    spot = [i for i in dec if status[i] == "agree"]
    print(f"抽查和构造标签一致：{sum(dec[i]['label'] == label[i] for i in spot)} / {len(spot)}")
    out, dropped = [], Counter()
    for s in draft:
        st = status[s["id"]]
        if st == "overturned" or st == "split" and dec[s["id"]]["label"] == "DROP":
            dropped[st] += 1
            continue
        s = {**s, "audited": True, "label_source": "review" if st == "agree" else "adjudicated"}
        d = dec.get(s["id"]) if st == "split" else None
        if d and d["label"] != s["label"]:
            assert d["label"] == "BLOCK", d  # 这批裁决只有 ALLOW 改 BLOCK；反方向要另定类别
            s["meta"] = {**s["meta"], "constructed": {"label": s["label"], "category": s["category"]}}
            s |= {"label": "BLOCK", "category": "misaligned_overreach", "risk_source": "inherent_agent_failure",
                  "reason": d["note"].split("：", 1)[1]}
        out.append(s)
    OUT.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in out))
    print(f"{len(draft)} → {len(out)}（删掉 {dict(dropped)}），改标签 {sum('constructed' in s['meta'] for s in out)} 条")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest(), OUT)
    for k, n in sorted(Counter((s["label"], s["category"]) for s in out).items()):
        print(k, n)


if __name__ == "__main__":
    main()
