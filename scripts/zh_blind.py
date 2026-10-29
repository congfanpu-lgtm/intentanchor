"""中文测试集双盲审（设计文档 §5.3.4）：data/zh_test/draft.jsonl 渲染成守卫看到的文本，去掉标签和来源，
a、b 两组各自打乱、各自编号，各分 8 份给两组子 agent 独立审。编号对照表 mapping.json 留在这里，不给子 agent。
在 WSL 项目根目录跑：.venv-serve/bin/python scripts/zh_blind.py
子 agent 拿到的评审标准是 docs/zh_audit_rubric.md。2026-10-27 从 runs/2026-10-21-zhtest-audit/ 移进仓库，输出仍写在那个目录。"""
import json, random
from pathlib import Path
from intentanchor.prompt import format_input

here = Path("runs/2026-10-21-zhtest-audit")
rows = [json.loads(l) for l in open("data/zh_test/draft.jsonl")]
mapping = {}
for g, seed in (("a", 7), ("b", 8)):
    (here / g).mkdir(parents=True, exist_ok=True)
    rs = rows[:]
    random.Random(seed).shuffle(rs)
    size = -(-len(rs) // 8)
    for k in range(8):
        with open(here / g / f"chunk{k}.txt", "w") as f:
            for j, s in enumerate(rs[k * size:(k + 1) * size]):
                qid = f"{g}{k * size + j + 1:03d}"
                mapping[qid] = s["id"]
                f.write(f"===== SAMPLE {qid} =====\n{format_input(s['input'], 'B')}\n\n")
json.dump(mapping, open(here / "mapping.json", "w"), indent=0)
print(len(rows), "samples x 2 groups, 8 chunks of", size)
