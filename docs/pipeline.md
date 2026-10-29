# 从原始数据到最终模型：完整命令链

2026-10-25 根据开发记录和代码整理。

**能不能直接重跑**：
- **确定的**：测试集下载和转换（`scripts/fetch_benchmarks.sh`）、从冻结数据训练和测评（`scripts/reproduce.sh`，要有冻结数据，见本节最后一条）。
- **不能一条命令重建的**：训练数据。原因有三个：
  1. 要调 API（合计约 60 元），重跑输出会不同；
  2. 要 Claude 和人工审核；
  3. 部分步骤用的是旧版代码，本仓库只有现在的版本（见最后一节）。
- 冻结的数据文件（`data/v2.7`、`data/h2`、`data/v2-en/dev.jsonl` 等）和模型权重都不公开。多数训练数据是经阿里云百炼生成或处理的，不分发是为了遵守百炼的服务条款。所以从冻结数据训练只有作者本地能跑；这份文档说明数据是怎么来的。

**记号**：

| 记号 | 含义 |
|---|---|
| `S` | `.venv-serve/bin/python` |
| `T` | `.venv/bin/python` |
| `VF` | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| `ENV` | `set -a && . ./.env && set +a`（加载百炼 key；`.env` 里写 `DASHSCOPE_API_KEY=<key>`） |

都在 WSL 项目根目录运行。

## A. 外部数据

| 数据 | 来源和版本 | 放置路径 |
|---|---|---|
| StepGuard 数据 | HF `ninty-seven/stepguard_data` @ aadeae0（2026-10-13 下载时的最新版） | `data/external/stepguard_data/{sft3k,rl4k}/` |
| StepGuard 模型、代码 | HF `ninty-seven/StepGuard` @ c7e4a2c（同上）；GitHub `zheng977/StepGuard` @ 93b580b | 模型在 HF 缓存；代码只用来对照（拦截协议、反馈原文），命令链不依赖它 |
| InjecAgent | GitHub `uiuc-kang-lab/InjecAgent` @ f19c9f2 | `data/external/InjecAgent/data/` |
| AgentDojo | `git+https://github.com/ethz-spylab/agentdojo@089ed468`，benchmark v1.2.2 | 装在 `.venv-serve` 里 |
| 5 个静态测试集 | `bash scripts/fetch_benchmarks.sh`（版本和 sha256 写在脚本里） | `data/external/benchmarks/` |

StepGuard 数据：`.venv-serve/bin/hf download ninty-seven/stepguard_data --repo-type dataset --revision aadeae009800d8fbcd145575f1d1a6b37abb0950 --local-dir data/external/stepguard_data`

## B. 转换公开数据（CPU）

1. `S -I -m intentanchor.data.stepguard data/external/stepguard_data --out data/stepguard_v3.jsonl`
   - 再加 `--subset rl4k` 跑一次，输出 `data/stepguard_rl4k_v1.jsonl`，给 DPO 用。
2. `S -I -m intentanchor.data.benchmarks data/external/benchmarks data/external/benchmarks/assebench/ASSEBench/dataset --out data/bench_v1`
3. `ENV; S -I -m intentanchor.data.injecagent data/external/InjecAgent --out data/injecagent_v1.jsonl`
   - 调 qwen3.8-max 填参数，约 1 元；结果缓存在 `<out>.args.json`。

## C. AgentDojo 训练轨迹（API，约 24 元）

被测模型是百炼的 qwen3.6-35b-a3b，开思考，只跑 workspace、travel 两个 suite。

1. `ENV; S scripts/run_attack_batch.py --thinking on --suites workspace,travel --attack none --out runs/2026-10-13-train-benign`
2. 攻击批次：
   - injecagent：`--attack injecagent --n-user 5`；
   - direct、ignore_previous、system_message：先 `--n-user 5 --limit 30` 试跑，再 `--inj-per-user 2` 筛查。
   - 输出到 `runs/2026-10-13-train-<攻击>`。
3. 拆成单步样本：`S -m intentanchor.data.split_label runs/2026-10-13-train-{benign,injecagent,direct,ignore_previous,system_message} --splice --reads --replies --out data/adojo_v3.jsonl`
   - 运行目录要按这个顺序传。

## D. 合成场景（API）

1. 生成：`ENV; S -I -m intentanchor.data.synth --n-per-domain 22 --langs en,zh --out data/synth_v0.jsonl`
   - qwen3.7-plus，8.1 元。
2. 大模型裁判复核：`ENV; S -m intentanchor.eval.score_llm data/synth_v0.jsonl --model qwen3.8-max --out runs/2026-10-13-synth-v0-judge`
   - 11.1 元。当时用的是 v0.4 的提示词，现在的 `prompt.py` 已经改了（见最后一节）。
3. 补工具返回：`ENV; S -I -m intentanchor.data.fill_outputs data/synth_v0.jsonl --out data/synth_v1.jsonl`
   - 约 1.75 元。

## E. 构建数据集（CPU；按 md5 分组划分，结果确定）

1. **v2-en**：`S -I -m intentanchor.data.build data/stepguard_v1.jsonl data/adojo_v1.jsonl data/injecagent_v1.jsonl --out data/v2-en`
   - 所有阈值都取在它的开发集（561 条）FPR 5% 处。
   - 输入是用早期版本的代码生成的，现在的代码重跑会不同，见最后一节。
2. **翻译**：先建 v2.3（v2-en 的输入加上 `data/adojo_v3.jsonl data/synth_v1.jsonl`，再加 `--judge runs/2026-10-13-synth-v0-judge/scores.jsonl --relabel runs/2026-10-15-synth-relabel/{claude,human}.jsonl`）。然后 `ENV; S -I -m intentanchor.data.translate data/v2.3/train.jsonl data/v2.3/dev.jsonl --per-group 60 --out data/zh_trans_v1.jsonl`
   - 13.8 元。
   - `runs/2026-10-15-synth-relabel/` 是 v1.1 标签审计里合成样本那 50 条的复核结论（Claude 盲审、人工复核各一份，见设计文档 §5.3.7 的 v2.2），只在作者本地。
   - 再建 v2.4：v2.3 的输入加上 `data/zh_trans_v1.jsonl`，参数相同，`--out data/v2.4`。
3. **标签审计**：
   - 抽样：`S -I -m intentanchor.data.audit sample data/v2.4/train.jsonl data/v2.4/dev.jsonl --n-human 50 --n-extra-random 150 --n-extra-stepguard 0 --out data/v2.4/audit.jsonl`；
   - Claude 子 agent 盲审 200 条，人工复核 50 条，得到 `runs/2026-10-15-label-audit-v2/{claude,human}.jsonl`。
4. **v2.7**：`build data/stepguard_v3.jsonl data/adojo_v3.jsonl data/injecagent_v1.jsonl data/synth_v1.jsonl data/zh_trans_v1.jsonl --judge runs/2026-10-13-synth-v0-judge/scores.jsonl --relabel runs/2026-10-15-synth-relabel/{claude,human}.jsonl runs/2026-10-15-label-audit-v2/{claude,human}.jsonl --out data/v2.7`
   - 训练集 6,625 条，开发集 1,122 条。

## F. 训练和打分

1. **SFT**（每个种子约 4.5 小时）：`T -m intentanchor.train.sft --train <数据>/train.jsonl --dev <数据>/dev.jsonl --mode qlora --epochs 2 --bs 2 --grad-accum 8 --seed <s> --out <run>`
   - sft.py 的默认值是 3 epoch、bs 4，所以这几个参数必须显式传。
2. **打分**：`VF S -m intentanchor.eval.score_vllm <数据> --adapter <run>/best --out <run>/<子目录>`
   - 开发集 `data/v2-en/dev.jsonl` → `dev`；
   - 5 个测试集 → `bench`，约 20 分钟；
   - DPO 还要 `data/stepguard_rl4k_v1.jsonl` → `rl4k`。
3. **H0**：用 v2.7 训练。

## G. 困难正常样本（第一轮）

当时用的是第一轮的 `hardneg.py`。现在的版本含第二轮的改动，照下面重跑，得到的 `data/h2` 会和当初不同（见最后一节）。

1. 候选池：`S -m intentanchor.data.hardneg pool runs/2026-10-13-train-{direct,ignore_previous,system_message,injecagent} --base data/v2.7 --out data/hardneg/pool.jsonl`
2. 用 H0 seed 0 给候选池打分：`score_vllm`，输出 `runs/2026-10-19-hardneg-pool`。
3. 挖出待审样本：`hardneg mine data/hardneg/pool.jsonl runs/2026-10-19-hardneg-pool/scores.jsonl --base data/v2.7 --review-out data/hardneg/review.jsonl`
4. 盲审：8 个 Claude 子 agent，得到 `runs/2026-10-19-hardneg-audit/verdicts.jsonl`。
5. 出训练集：`hardneg mine ... --verdicts <盲审结果> --out data`，得到 `data/h1/`、`data/h2/`。H2 训练集 6,774 条。
6. 用 `data/h2` 按 F 训练，得到 H2，也就是最终候选。

## H. 基线、DPO、出表

- **基线**：
  - 零样本：不带 `--adapter` 的 `score_vllm`；
  - StepGuard：`VF S -m intentanchor.eval.score_stepguard <数据> --out <目录>`。
- **DPO**（每次约 12 分钟）：
  1. `S scripts/mine_hard.py data/stepguard_rl4k_v1.jsonl <H0>/rl4k/scores.jsonl --out data/hard/rl4k-v27s0.jsonl`
  2. `T -m intentanchor.train.dpo --train data/hard/rl4k-v27s0.jsonl --init-adapter <H0>/best --out <run>`
  - 对照组"继续 SFT"：`sft.py --init-adapter <H0>/best --lr 2e-5`。
- **出表**：
  - `S scripts/bench_table.py <名字/种子>=<dev 目录>,<bench 目录> ...`
  - `S scripts/inj_fp.py ...`
  - `S scripts/fp_kind.py ...`
  - `S scripts/mcnemar.py <组/种子>=<训练目录> ... --compare h2:h0 h2:h1`：TS-Dojo 上同种子配对的 McNemar 检验。

## I. 端到端（API，每种配置约 23 元）

1. 起守卫：`VF .venv-serve/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 18000 --enable-lora --max-lora-rank 16 --lora-modules sft=<run>/best --max-model-len 6144 --gpu-memory-utilization 0.85 --enable-prefix-caching`
2. 跑用例：`ENV; S scripts/run_attack_batch.py --thinking on --suites banking,slack --workers 16 --guard sft --threshold 0.679 --out runs/<日期>-e2e-sft-attack`
   - 再加 `--attack none` 跑一批，输出 `-benign`。
3. 出表：`S scripts/e2e_table.py runs/<日期>-e2e none sft dpo stepguard`；加 `--compare h2:sft h2:stepguard` 做同一用例的配对检验。
4. 最终模型 H2 seed 0（2026-10-28）：LoRA 名 `h2=runs/2026-10-19-sft-h2s0-qlora/best`，阈值 0.755，其余同上；表里其他配置沿用 2026-10-17 那一轮。队列脚本 `runs/2026-10-28-e2e-h2.sh` 只在作者本地。

## J. 推理服务压测

见 `src/intentanchor/serve.py` 和 `scripts/bench_serve.py` 开头的说明。

## K. 中文测试集（DeepSeek 约 3.5 元，其余 CPU / GPU）

1. 起草：`ENV; S -I -m intentanchor.data.zh_testset --n-per-domain 15 --out data/zh_test/draft.jsonl`，60 个场景里 57 个合格，得到 430 条。
2. 盲审：
   - `S scripts/zh_blind.py` 生成 a、b 两组材料；
   - 两组 Claude 子 agent 按 `docs/zh_audit_rubric.md` 各审 8 份；
   - `S scripts/zh_merge.py` 合并，写出 `verdicts.jsonl`，以及交给核验页的 66 条。
3. 裁决：核验页的结果导出为 `runs/2026-10-21-zhtest-audit/decisions.jsonl`（这次是另一个 Claude 会话做的）。
4. 冻结：`S scripts/zh_freeze.py`，得到 `data/zh_test/v1.jsonl`，350 条，sha256 `a4f13dbf…`。
5. 打分：风险分和标签无关，所以冻结前就在 430 条草稿上打好了，冻结后按最终标签出表。每个模型一个目录：
   - 本项目的模型：`VF S -m intentanchor.eval.score_vllm data/zh_test/draft.jsonl --adapter <run>/best --out runs/<日期>-zhtest-score/<名字>`；零样本基座去掉 `--adapter`；
   - StepGuard：`VF S -m intentanchor.eval.score_stepguard data/zh_test/draft.jsonl --out runs/<日期>-zhtest-score/stepguard`；
   - 只有 Hugging Face 上的 `v1.jsonl` 时，直接对它打分：`zh_table.py` 只按 id 取冻结集里的样本；没有开发集时，逗号前直接写阈值，如 `stepguard=0.5,<中文打分目录>`（AUROC 不受阈值影响）。
6. 出表：`S scripts/zh_table.py <名字>=<开发集打分目录>,<中文打分目录> ... --compare h2:h0 h2:stepguard`。

## L. 发布前检查（GPU 上约 5 分钟）

`S -W ignore scripts/release_check.py h2/s0=runs/2026-10-19-sft-h2s0-qlora/bench ... stepguard=runs/2026-10-14-bench-stepguard > runs/2026-10-27-release-check.txt`：测试集和最终训练集查重（完全相同、用户请求相同、字符 3-gram 近似），R-Judge、ASSE-Safety 按重叠分两半算 AUROC，再扫描个人信息。完整命令见脚本开头。

## 用当前代码重建不出来的地方

| 产物 | 原因 |
|---|---|
| `data/v2-en/dev.jsonl` | 它用的 `stepguard_v1`、`adojo_v1` 是早期代码生成的；后来 stepguard.py 会过滤 HiddenCall，`--splice` 会多产出 spliced_cont |
| `data/v2.3`（翻译的抽样底） | 同上。换了底，`--per-group 60` 抽到的样本就变了 |
| `data/h2` | hardneg.py 第二轮改了候选池（加了 AgentDojo 的 ALLOW 作基础样本）和挖掘规则（某个基础样本只要有一个变体被审成非 ALLOW，它的变体就全部不用） |
| 合成场景的裁判筛选 | 用的是 v0.4 提示词，现在的 `prompt.py` 已经改了 |

这些产物的冻结文件，连同 `zh_trans_v1.jsonl` 和裁判的 `scores.jsonl`，都只在作者本地，不公开。

**没有收进本仓库的脚本**（只在作者本地的 `runs/` 下）：
- 审核相关：训练数据标签审计、两轮困难样本审核的 `make_blind.py`、`merge.py`。中文测试集的已经移进 `scripts/zh_{blind,merge}.py`；
- 一次性分析：如 `r2_eval.py`、`r2_pool.py`（设计文档 §5.2.4 第二轮候选池的误拦率）；
- 训练和测评队列：`runs/*.sh`。

README 里的 p 值都由仓库里的脚本算出：TS-Dojo 上同种子的配对检验用 `scripts/mcnemar.py`，端到端用 `scripts/e2e_table.py --compare`，中文测试集用 `scripts/zh_table.py --compare`。设计文档里较早的 p 值最初由一次性脚本 `hn_sig.py`、`r2_eval.py`、`2026-10-18-e2e-sig.py` 算出；它们用的是同样的检验，已被 `mcnemar.py` 和 `e2e_table.py --compare` 取代。
