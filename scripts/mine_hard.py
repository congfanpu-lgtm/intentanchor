"""从样本池里挑出 SFT 模型判错的样本，DPO 和"困难样本继续 SFT"对照共用这一份（设计文档 §5.4.3）。在 WSL 项目根目录跑：

    .venv-serve/bin/python scripts/mine_hard.py data/stepguard_rl4k_v1.jsonl runs/<SFT>/rl4k/scores.jsonl \
        --out data/hard/<SFT 名>.jsonl

判错 = ALLOW 的风险分 ≥ 阈值（误拦），或 BLOCK 的风险分 < 阈值（漏拦）。样本原样写出，格式和训练集相同。
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pool", type=Path)
    p.add_argument("scores", type=Path, help="score_vllm 在样本池上的 scores.jsonl")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    pool = [json.loads(line) for line in a.pool.read_text().splitlines()]
    score = {r["id"]: r["score"] for r in map(json.loads, a.scores.read_text().splitlines())}
    assert set(score) == {s["id"] for s in pool} and len(score) == len(pool), "scores do not match the pool"
    hard = [s for s in pool if (score[s["id"]] >= a.threshold) != (s["label"] == "BLOCK")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in hard))
    kind = lambda s: (s["label"], "reply" if "response" in s["input"]["proposed_call"] else "call")
    total = Counter(map(kind, pool))
    for k, n in sorted(Counter(map(kind, hard)).items()):
        print(f"{k[0]:5} {k[1]:5} wrong {n:4} / {total[k]}")
    print(f"hard {len(hard)} / {len(pool)} -> {a.out}")


if __name__ == "__main__":
    main()
