#!/usr/bin/env bash
# 下载 StepGuard 的 5 个静态测试集，固定到 2026-10-14 用的版本并校验 sha256（设计文档 §5.5.1）。
# 这些数据不随本仓库再分发（R-Judge、TS-Bench 没有数据许可证；TS-Bench、ASSEBench 含 AgentHarm，只能用于 AI 安全研究）。
#
#   bash scripts/fetch_benchmarks.sh [data/external/benchmarks]
#   .venv-serve/bin/python -I -m intentanchor.data.benchmarks data/external/benchmarks \
#       data/external/benchmarks/assebench/ASSEBench/dataset --out data/bench_v1
set -euo pipefail
R=${1:-data/external/benchmarks}
TS=https://media.githubusercontent.com/media/MurrayTom/ToolSafe/46358fa424a927a895c6c8322f99032c4eb5155e/TS-Bench
ATB=https://huggingface.co/datasets/AI45Research/ATBench/resolve/4476ef92ed8f85c8d58d8a5b9dfdf55aa7893138/ATBench/test.json

for f in agentdojo-traj/{banking,slack,travel,workspace} agentharm-traj/{benign_steps,harmful_steps} \
         asb-traj/test/{DPI_attack_success,OPI_attack_success,atttack_failure}; do
  mkdir -p "$R/ts_bench/$(dirname $f)"
  curl -sfL "$TS/$f.json" -o "$R/ts_bench/$f.json"
done
mkdir -p "$R/atbench_pro" && curl -sfL "$ATB" -o "$R/atbench_pro/ATbench-Pro.json"

clone() { [ -d "$2" ] || git clone -q "$1" "$2"; git -C "$2" checkout -q "$3"; }
clone https://github.com/Lordog/R-Judge "$R/rjudge/upstream" 83ce301da3ad50dd8b397e772863f5411c3d3dc2
clone https://github.com/Astarojth/AgentAuditor-ASSEBench "$R/assebench" b1204ee
# R-Judge 没有现成的单文件：按路径排序拼接 data/<类别>/*.json 的记录（StepGuard 的输入约定），记录本身不改
python3 -I - "$R/rjudge/upstream" "$R/rjudge/test.json" <<'EOF'
import json, pathlib, sys
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
records = [r for f in sorted((root / "data").glob("*/*.json")) for r in json.loads(f.read_text(encoding="utf-8"))]
out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
EOF

cd "$R" && sha256sum -c --quiet <<'EOF'
eb4a314845c8236ba1bbcd9855973f59279a70368b6eb31d447a4cd243086029  ts_bench/agentdojo-traj/banking.json
08152de8cf9c39d01d58eb708f3a820a892f493841b975e09654ab904cf8a0d5  ts_bench/agentdojo-traj/slack.json
1935121585fb1a02662e8121fe80e78c6d89b77e0c6035ea2b75b593e3728a22  ts_bench/agentdojo-traj/travel.json
74b20a4fcd7df1c3e39a231e64860aeac8cd57ce3531728b329080abbcc862d0  ts_bench/agentdojo-traj/workspace.json
f44fce25b0ed795e3c11fadafb8d8c959d2ae4513e6e9f15e154fa4aba9bf020  ts_bench/agentharm-traj/benign_steps.json
6046b4f4b2e5086c98780433bddf74be542eeec910e3260b4727bf22a9a3365b  ts_bench/agentharm-traj/harmful_steps.json
a900a0775addf35588154d28438127f9c830a08845a105e084b319b1e984d4af  ts_bench/asb-traj/test/DPI_attack_success.json
e491c96e43c8352e65ca948a72cb3e6f366d60f14fd7d3d6e374865eb092ae47  ts_bench/asb-traj/test/OPI_attack_success.json
6cf422389a41416e05d236252ecba8757c1f7d7a978ae270e419ff846ae98f51  ts_bench/asb-traj/test/atttack_failure.json
80c534b5f3517c872b528a2e6b64e34495206968edd184f4aed563d39f0cca09  atbench_pro/ATbench-Pro.json
933eaf4cbb5431ed92c093ba5812ab10a5b29cc78b0f4d62ef6bc1c03ca0b37e  rjudge/test.json
376f078871f18c32f66eb9c94a221db9a2223318cf1b9728c89c85af2563e907  assebench/ASSEBench/dataset/AgentJudge-safety.json
4db9e5c976d0ed35f11b97afcd697259b4954e776777c150f03b3cab5ba31a57  assebench/ASSEBench/dataset/AgentJudge-security.json
EOF
echo "benchmarks ready in $R"
