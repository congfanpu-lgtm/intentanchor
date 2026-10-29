#!/usr/bin/env bash
# 从冻结数据复现 H0 / H2 的训练和离线测评（设计文档 §5.2.4、§5.5.1）。只用 GPU，不调 API。
# 需要：data/v2.7、data/h2、data/v2-en/dev.jsonl（冻结的训练集和开发集）；data/bench_v1（scripts/fetch_benchmarks.sh 后转换）。
#
#   bash scripts/reproduce.sh 0      # 一个种子：H0、H2 各训一次（每个约 4.5 小时，RTX 5080 笔记本）再打分出表
#
# 训练在 GPU 上不是逐位确定的，而且种子之间波动大（同一份数据，开发集阈值能从 0.56 变到 0.85），
# 所以对照的是 README 里每个种子的数字和三个种子的范围，不是某一个数。
set -euo pipefail
S=${1:-0}
export VLLM_USE_FLASHINFER_SAMPLER=0
B="data/bench_v1/tsbench.jsonl data/bench_v1/atbench.jsonl data/bench_v1/rjudge.jsonl data/bench_v1/asse-safety.jsonl data/bench_v1/asse-security.jsonl"
score() { .venv-serve/bin/python -m intentanchor.eval.score_vllm "$@"; }
M=""
for v in h0:v2.7 h2:h2; do
  name=${v%%:*} data=data/${v##*:} out=runs/repro-${v%%:*}s$S
  [ -f $out/best/adapter_config.json ] || .venv/bin/python -m intentanchor.train.sft --train $data/train.jsonl \
    --dev $data/dev.jsonl --mode qlora --epochs 2 --bs 2 --grad-accum 8 --seed $S --out $out
  score data/v2-en/dev.jsonl --adapter $out/best --out $out/dev   # 阈值取在这个开发集 FPR 5% 处
  score $B --adapter $out/best --out $out/bench
  M="$M $name/s$S=$out/dev,$out/bench"
done
.venv-serve/bin/python scripts/bench_table.py $M | tee runs/repro-s$S-table.txt
.venv-serve/bin/python scripts/inj_fp.py $M | tee runs/repro-s$S-injfp.txt
.venv-serve/bin/python scripts/mcnemar.py h0/s$S=runs/repro-h0s$S h2/s$S=runs/repro-h2s$S --compare h2:h0 | tee runs/repro-s$S-mcnemar.txt
