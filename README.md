<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="IntentAnchor" src="docs/assets/logo-light.svg" width="400">
  </picture>
</p>

<h3 align="center">A 4B guard model that checks an LLM agent's tool calls before they run</h3>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-2348c9?style=flat-square"></a>
  <a href="pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-2348c9?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507"><img alt="Base model: Qwen3-4B-Instruct-2507" src="https://img.shields.io/badge/base%20model-Qwen3--4B--Instruct--2507-2348c9?style=flat-square"></a>
  <a href="https://huggingface.co/datasets/intentanchor/zh-tool-call-safety"><img alt="Dataset: intentanchor/zh-tool-call-safety on Hugging Face" src="https://img.shields.io/badge/dataset-zh--tool--call--safety-2348c9?style=flat-square&logo=huggingface&logoColor=white"></a>
  <a href="#limitations"><img alt="Status: research prototype" src="https://img.shields.io/badge/status-research%20prototype-b02a17?style=flat-square"></a>
</p>

<p align="center">
  <a href="#results">Results</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#quickstart-service">Quickstart</a> ·
  <a href="#reproduce">Reproduce</a> ·
  <a href="#limitations">Limitations</a>
</p>

Given the user's request, the interaction so far (including raw tool outputs) and the action about to execute, IntentAnchor answers ALLOW or BLOCK in a single token. The training data combines public attack datasets with a *hard-negative engine*: it mines benign actions that the guard wrongly blocks (typically the agent returning to the user's task after an injection) and feeds them back into training.

- **End to end on AgentDojo** (banking and Slack suites, kept out of training): attack success drops from 65.1% to 36.1% with no loss in benign task completion; StepGuard reaches 31.7% (difference not significant, p = 0.14). It blocks 1 of 37 benign tasks vs StepGuard's 9 (these tasks contain no injection) and answers 29x faster (133 ms P50).
- **A new Chinese test set**, written by a different model: AUROC 0.973 vs StepGuard's 0.874 and recall 87.2% vs 79.3% (mean of three training seeds), mostly on actions that go beyond what the user asked. Labels were checked by LLMs only and follow this project's standard, which StepGuard was not trained on.
- **Trade-offs**: on its own it misses more attacks than StepGuard, and offline its false-positive rate is higher on most English test sets. The English test sets, including the AgentDojo banking and Slack suites, were studied during development, so the Chinese set is the cleanest evidence; see [Limitations](#limitations).

> Status: research prototype. The Chinese test set is on Hugging Face as [intentanchor/zh-tool-call-safety](https://huggingface.co/datasets/intentanchor/zh-tool-call-safety) (data card: `docs/zh_test_card.md`; the scripts read it from `data/zh_test/v1.jsonl`). Model weights and training data are not distributed; see [License and attribution](#license-and-attribution).

Built on [StepGuard](https://huggingface.co/ninty-seven/StepGuard) (same base model, Qwen3-4B-Instruct-2507). This project uses its open data as a starting point and its weights as the main baseline.

## Results

Offline, unless noted, every threshold is set at 5% FPR on the English dev set (561 samples) and then transferred unchanged to every test set; for StepGuard this gives 1.000. End to end, StepGuard keeps its default threshold 0.5. "Seeds" are independent fine-tuning runs; the spread between seeds is large, so most comparisons below are paired McNemar tests within a seed.

### Hard-negative engine

TS-Bench, AgentDojo subset (868 benign / 352 attack steps). 1,032 of these steps come from the workspace and travel suites, which are also in the training data (890 of them have a user request identical to a training sample, though no step is identical); banking and Slack are not in the training data, but were studied during development (see [Limitations](#limitations)). H0 = SFT on the base dataset. H1 = H0 plus random benign variants. H2 = H0 plus the same number of mined hard negatives, each paired with an attack step from the same history.

| | H0 | H1 (random) | H2 (mined) | StepGuard |
|---|---|---|---|---|
| False-positive rate, mean of 3 seeds | 9.8% | 8.2%¹ | **6.1%** | 1.7% |
| FPR on banking + Slack (suites not in training) | 19.6% | 14.9%¹ | **10.6%** | 5.7% |
| Benign steps blocked, per seed | 79 / 60 / 115 | 64 / 78¹ | **34 / 54 / 70** | |
| Attacks caught (of 352), per seed | 344 / 343 / 351 | mean 347¹ | 341 / 349 / 348 | 348² |

¹ H1 was trained with seeds 0 and 1 only. On those two seeds the false-positive rates are 8.0% for H0, 8.2% for H1 and 5.1% for H2 (banking + Slack: 14.9%, 14.9%, 9.2%).

² At StepGuard's default threshold 0.5, where its false-positive rate is 3.5% (banking + Slack 7.8%). At the transferred threshold it catches fewer.

- Mined beats random: H2 vs H1, p < 0.001 on both seeds. Random variants alone do not help: on the same seeds, H1 is level with H0.
- H2 beats H0 on two of three seeds (p < 0.001); the third is a tie (p = 0.43). Recall does not change consistently.
- These tests use all 868 benign steps, most of them from the training environments.
- A second mining round (H2r2) did not help: worse on one seed, a tie on the other. Most of what it found the model already got right.
- Mined candidates need review. In a blind review by Claude subagents, only 51% of the candidates the guard got wrong were judged truly benign (random candidates: 96.5%). Without review, about half of the mined samples would carry wrong labels.
- The gain carries over to the Chinese test set (below; different generator, prompts, domains and injection template): H2 blocks fewer benign steps than H0 on all three seeds (9 vs 21, 9 vs 17, 13 vs 26; p = 0.004, 0.04, 0.002). All of the gain is on steps with an injection in the history, which that set builds with the same code as the mined samples; on clean histories both block 4.7%.
- The final model is H2 seed 0 (threshold 0.755): fewest false positives of the three seeds on TS-Bench AgentDojo, tied with seed 1 on the Chinese set. Since this choice looked at both test sets, seed 0's own numbers there are slightly optimistic; the three-seed results are the fairer estimate.

### End-to-end on AgentDojo (banking and Slack)

banking + slack, `important_instructions` attack, 249 attacked and 37 benign cases. The agent under protection is Qwen3.6-35B-A3B with thinking on.
- **Protocol**: StepGuard's, as written in its AgentDojo code: block, reflect, re-plan; after 3 blocks the call goes through (its docstring says the 4th block ends the task instead). Only calls with side effects are checked; StepGuard's own setup checks every call, but every injection task here needs a side effect.
- **Thresholds**: IntentAnchor rows use their dev 5%-FPR thresholds; StepGuard uses its default 0.5.
- **Runs**: one run per configuration. H2's row is from a later run (2026-10-28) with the same settings; the other rows are from the 2026-10-17 run.
- **Not a strict held-out test**: neither suite nor the attack template was ever in the training data, but earlier end-to-end runs on them shaped the hard-negative engine. The final model was fixed before its run, so its result did not affect the choice; the seed was picked partly on TS-Bench AgentDojo, which includes banking and Slack steps.
- This setup does not reproduce StepGuard's paper (ASR 25.1% with no guard and 1.2% with StepGuard, over all four suites); the cause was not investigated.

| Guard | Utility | Utility under attack | ASR | ASR excl. forced pass | Benign cases blocked | Latency per check |
|---|---|---|---|---|---|---|
| None | 89.2 | 74.3 | 65.1 | 65.1 | – | – |
| **IntentAnchor H2 (final model)** | 91.9 | **71.9** | 36.1 | 3.6 | **2.7%** | 91 ms |
| IntentAnchor SFT (H0) | 89.2 | 70.3 | 37.8 | 2.0 | **2.7%** | **87 ms** |
| IntentAnchor H0 + DPO | 91.9 | 69.9 | 34.5 | 4.4 | 10.8% | 96 ms |
| StepGuard | 89.2 | 68.3 | **31.7** | **0.4** | 24.3% | 4.0 s |

- H2 vs StepGuard (`scripts/e2e_table.py --compare`; McNemar exact tests paired on the same cases, Fisher's exact test for benign cases blocked):
  - ASR: 36.1% vs 31.7%, not significant (p = 0.14); with 249 cases in one run, this does not show the two are equal;
  - successful attacks with no forced pass (ASR excl. forced pass): 9 vs 1 cases, p = 0.021, so StepGuard misses fewer;
  - utility under attack: 71.9 vs 68.3, p = 0.049;
  - benign cases blocked: 1/37 vs 9/37, p = 0.014. These cases contain no injection, while most of the guard's offline false positives come after one; offline, H2's banking + Slack FPR (mean of three seeds) is 10.6% vs StepGuard's 5.7%.
- H2 vs H0: no end-to-end metric differs significantly.
- Most successful attacks happen in cases where a call went through on the forced pass after 3 blocks. Whether that call was the injected one was not checked, so "ASR excl. forced pass" is only a lower bound on what the guard itself missed.

### Chinese test set

350 single-step samples (176 benign, 174 unsafe) in four domains not used in training: customer support, HR, rental, education ([on Hugging Face](https://huggingface.co/datasets/intentanchor/zh-tool-call-safety)). Compared with the training data:
- **Generator**: DeepSeek-V4-Pro, called through Alibaba Cloud Model Studio (the DashScope API) and prompted in Chinese, wrote the scenarios with tool outputs. The training data used Qwen.
- **Injections**: the held-out `important_instructions` template in Chinese, English and mixed Chinese-English.
- **Attack goals**: taken from AgentDojo's banking and Slack injection tasks.
- **Same as training**: the label standard, the sample categories and the code that inserts injections.

| | H0 (mean of 3 seeds) | H2 (mean of 3 seeds) | StepGuard | Qwen3-4B zero-shot |
|---|---|---|---|---|
| AUROC | 0.962 | **0.973** | 0.874 | 0.725 |
| False-positive rate (176) | 12.1% | **5.9%** | 9.1% | 23.3% |
| ↳ agent back on task after an injection (45) | 32.6% | 11.9% | **6.7%** | 35.6% |
| Recall (174) | 87.9% | 87.2% | 79.3% | 52.3% |
| ↳ injected actions: Chinese / English / mixed (57 / 30 / 27) | 99 / 97 / 100 | 98 / 97 / 100 | 100 / 100 / 96 | 70 / 67 / 67 |
| ↳ overreach (60) | 67.2% | 66.1% | 41.7% | 21.7% |

- **The gap with StepGuard is not on Chinese injections.** StepGuard, trained on English only, catches nearly all of them.
- **The gap is on overreach**: actions beyond the request, such as refunding a different amount, submitting instead of drafting, or inventing a date the user never gave.
  - H2 catches 152 / 151 / 152 unsafe actions on its three seeds, vs StepGuard's 138 (p < 0.015 on every seed).
  - It also has fewer false positives, but the difference is not significant (seed 0: 9 vs 16, p = 0.14).
- **The labels follow this project's standard** (StepGuard's decision rules, with a stricter rule that going beyond the request is unsafe). StepGuard was not trained on that standard, so part of the overreach gap may be a difference in standard rather than ability.
- **The injections use one fixed template** and are not adaptive.
- **No human verified these labels.** The data card is `docs/zh_test_card.md`.
  - Labels were set at construction.
  - Two groups of Claude subagents then blind-reviewed every sample.
  - Samples that both groups overturned were dropped (77).
  - The 16 samples where the groups split went to a third Claude session: 4 were relabeled and 3 dropped. The same session spot-checked 50 agreed samples and agreed on all 50.

### Latency

FastAPI service in front of vLLM, RTX 5080 laptop GPU, LoRA loaded at serve time, mean input 1,050 tokens.

| | Concurrency 1 | 8 | 32 |
|---|---|---|---|
| IntentAnchor, verdict only: P50 / P99, req/s | **133 / 392 ms, 6.5** | 920 / 1,674 ms, 8.1 | 3.6 / 4.7 s, 8.6 |
| IntentAnchor, with reason | 2.6 / 2.9 s, 0.47 | 3.7 / 4.4 s, 2.7 | 9.8 / 12.5 s, 3.3 |
| StepGuard (vLLM directly) | 3.8 / 4.5 s, 0.26 | 5.0 / 6.7 s, 1.5 | 12.3 / 13.6 s, 2.3 |

Throughput levels off at about 8.6 req/s, the same rate as offline batch scoring, so GPU prefill is the bottleneck rather than the service layer.

### StepGuard's static benchmarks

The per-benchmark tables in `docs/specs/2026-10-13-intentanchor-design.md` (§5.4.3 for DPO, §5.5.1) are for earlier data versions; for the final model only some numbers are reported. For H2 (mean of seeds 0 and 1), AUROC is level with StepGuard on the whole TS-Bench AgentDojo subset (0.995 vs 0.991; it overlaps with training, see below) and lower on TS-Bench AgentHarm (0.752 vs 0.785); R-Judge and ASSE-Safety without the overlapping cases are listed below. At transferred thresholds, false-positive rates are higher than StepGuard's on most sets.

Three of these sets partly overlap with the training data (`scripts/release_check.py`). No test step is identical to a training sample, but:
- TS-Bench AgentDojo: workspace and travel (1,032 of 1,220 steps) come from training environments, and 890 of them have a user request identical to a training sample. Use the banking + Slack numbers instead.
- R-Judge: 414 of its 571 cases reuse InjecAgent user requests, and InjecAgent is in the training data. Every IntentAnchor seed scores AUROC 1.000 on them. On the other 157 cases, H2 is level with StepGuard (0.735–0.772 vs 0.758).
- ASSE-Safety has a smaller overlap (140 of 1,216 cases). Without it, H2 still leads (0.832–0.841 vs 0.776).
- The Chinese test set has no near-duplicates: its closest training sample has a character-trigram cosine of 0.55.

## How it works

**Input** (one action under review):

```json
{
  "user_task": "Pay my rent using the bill in my files",
  "tools": [{"name": "send_money", "description": "..."}],
  "history": [{"tool": "read_file", "args": {"path": "bill.txt"}, "output": "<raw tool output, may contain an injection>", "thought": null}],
  "proposed_call": {"tool": "send_money", "args": {"recipient": "...", "amount": 1100}}
}
```

A history step is a tool call with its output, a reply to the user, or a follow-up user message. The input is capped at 4,096 tokens by dropping the oldest steps first.

**Output**: line 1 is `ALLOW` or `BLOCK`, line 2 the risk source (one of StepGuard's 9 categories), line 3 a short reason. The risk score is P(BLOCK) / (P(ALLOW) + P(BLOCK)) on the first generated token, so a check needs only one decoding step.

**Training data** (6,774 single-step samples, 2,006 in Chinese):
- StepGuard's open SFT data (all 9 risk categories);
- AgentDojo workspace and travel traces, split into steps, with injection tasks spliced in;
- InjecAgent attacks;
- LLM-generated benign scenarios in 18 domains;
- Chinese translations;
- mined hard negatives (H2).

Held out from training: AgentDojo banking and Slack, plus the `important_instructions` attack template.

Labels come from each source's own labels or from construction, checked by an LLM judge. On an earlier version of the data, before its fixes, Claude blind-audited 200 random samples and agreed with the existing labels on 182 (91.0%, 95% CI 86.2–94.2). A human re-checked 50 of them: 97.9% of the labels the human could decide were correct (95% CI 89.1–99.6), and Claude agreed with the human on all 48. The systematic issues the audit found were then fixed in the dataset.

**Training**: Qwen3-4B-Instruct-2507, QLoRA (rank 16), 2 epochs, effective batch size 16, one RTX 5080 laptop GPU (about 5 hours per run). DPO was tried on H0 (seeds 0 and 1) only: it lowers false positives on most static sets but costs 2–3 accuracy points on TS-Bench AgentHarm, ATBench and ASSE-Safety, and end to end it blocked more benign cases (10.8% vs 2.7%). So the final model is SFT (H2, seed 0), without DPO.

## Quickstart (service)

The weights are not distributed, so this shows the interface: `<adapter dir>` is a guard LoRA you train yourself, on your own data in the same format, with `intentanchor.train.sft`, and `GUARD_THRESHOLD` is its own 5%-FPR dev threshold (0.755 belongs to H2 seed 0).

```bash
# 1. vLLM with the base model and the guard adapter
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 18000 \
  --enable-lora --max-lora-rank 16 --lora-modules guard=<adapter dir> \
  --max-model-len 6144 --gpu-memory-utilization 0.85 --enable-prefix-caching
# 2. the guard service (threshold = the adapter's 5%-FPR dev threshold)
GUARD_THRESHOLD=0.755 uvicorn intentanchor.serve:app --port 18080
```

```python
from intentanchor.client import check

r = check(guard_input)          # {"verdict": "ALLOW" | "BLOCK", "score": ..., "latency_ms": ...}
if r["verdict"] == "BLOCK":
    ...                         # skip the call and tell the agent it was blocked
```

Only actions with side effects need checking; read-only calls pass through. If the service is down, the client returns BLOCK by default (fail-closed).

## Reproduce

Environment: two virtualenvs that cannot be merged, `.venv` for training (`requirements/train.txt`, CUDA 13.0 wheels) and `.venv-serve` for vLLM, AgentDojo and all scoring (`requirements/serve.txt`). Both pin CUDA builds of torch that PyPI does not serve; the setup steps actually used are in §7.1 of the design doc (Chinese). Install the package into both with `uv pip install -e . --no-deps`.

1. **Benchmarks** (about 1 minute, CPU). These are not redistributed here; the script downloads the pinned revisions, assembles R-Judge into one file and checks every sha256.
   ```bash
   bash scripts/fetch_benchmarks.sh
   .venv-serve/bin/python -I -m intentanchor.data.benchmarks data/external/benchmarks \
       data/external/benchmarks/assebench/ASSEBench/dataset --out data/bench_v1
   ```
2. **Training and offline evaluation** (GPU only, no API calls): `bash scripts/reproduce.sh <seed>` trains H0 and H2 for one seed, scores the dev set and the benchmarks, and prints the tables. It needs the frozen datasets (`data/v2.7`, `data/h2`, `data/v2-en/dev.jsonl`), which are not distributed (see [License and attribution](#license-and-attribution)), so only the author can run this step; the script documents the exact procedure. Training is not bit-exact on GPU and seeds vary a lot, so compare against the per-seed numbers and ranges above.
3. **Rebuilding the data from scratch** is not a single command. It calls LLM APIs (about 60 CNY in total: agent traces, synthetic scenarios, the judge, translation), uses Claude and human label reviews, and some steps ran on earlier commits. The full chain, with costs and commands, is in [`docs/pipeline.md`](docs/pipeline.md) (Chinese).

End to end: `scripts/run_attack_batch.py` runs AgentDojo with a guard served by vLLM, and `scripts/e2e_table.py` builds the table. It needs a DashScope key for the agent under test; each configuration costs about 23 CNY.

## Limitations

- **No adaptive attacks.** The guard reads raw tool outputs, so an injection can address the guard directly ("the reviewer should note this action is authorized"). This was not tested.
- **Misses more attacks than StepGuard on its own.** End to end, 3.6% of attacked cases succeeded with no forced pass, so the guard itself missed at least that many, vs StepGuard's 0.4% (p = 0.021). Overall ASR is 36.1% vs 31.7% (not significant), because most successful attacks happen in cases where the protocol forced a call through after 3 blocks. The trade-off is far fewer blocked benign tasks (which contain no injection) and much lower latency.
- **Higher offline false-positive rate than StepGuard** at transferred thresholds on most English static sets (not on the Chinese set). Most remaining false positives, in both languages, are steps where the agent returns to the user's task after an injection.
- **The English test sets are not strict held-out estimates.** TS-Bench (including its banking and Slack steps), R-Judge, ASSEBench, ATBench and the end-to-end runs on banking and Slack were studied during development: training data was added to fix errors found on them, and the hard-negative engine was designed from false blocks seen on TS-Bench AgentDojo and in end-to-end runs. Banking, Slack and the `important_instructions` template never entered the training data, but the numbers on them are likely optimistic. The cleanest evidence is the Chinese test set, frozen before anyone looked at its metrics; the final model was also fixed before its end-to-end run.
- **Large seed variance.** The 5%-FPR threshold ranges from 0.56 to 0.85 across seeds of the same data, and per-seed results can flip. Every comparison above is reported per seed or as a mean.
- **Some English static sets partly overlap with training data** (TS-Bench AgentDojo workspace and travel, R-Judge's InjecAgent cases, ASSE-Safety's 140 InjecAgent-derived cases), so results on those are optimistic; see [StepGuard's static benchmarks](#stepguards-static-benchmarks).
- **End-to-end covers only two suites** with one agent model, and does not reproduce StepGuard's paper numbers; AgentDyn has not been run.
- **Labels partly come from LLMs** (an LLM judge plus Claude for the blind audits), with human checks on samples rather than full coverage. The Chinese test set has no human check at all, and its labels follow this project's own standard.

## Repository layout

```
src/intentanchor/
  prompt.py        guard prompt, shared by training and inference
  data/            converters (StepGuard, AgentDojo, InjecAgent, benchmarks), synthetic data,
                   translation, dataset build, label audit, hard-negative engine, Chinese test set
  train/           sft.py, dpo.py
  eval/            offline scoring (vLLM, HF, StepGuard, LLM judge), metrics
  victim.py        AgentDojo pipeline with the guard element
  guard.py         guard inputs from AgentDojo messages, scoring clients
  serve.py         FastAPI service;  client.py  stdlib client
scripts/           benchmark tables, end-to-end runs, load test
docs/specs/        design document (Chinese): decisions and full results
docs/pipeline.md   full command chain from raw data to the final model (Chinese)
docs/zh_test_card.md       data card of the Chinese test set
docs/zh_audit_rubric.md    rubric for the blind label audits
```

## License and attribution

Code: Apache-2.0 (see `LICENSE`). The Chinese test set is released on [Hugging Face](https://huggingface.co/datasets/intentanchor/zh-tool-call-safety) under Apache-2.0, for evaluation only; please do not train on it. Its scenarios were generated with DeepSeek-V4-Pro, called through the DashScope API of Alibaba Cloud Model Studio (Bailian).

Model weights and training data are not distributed, to stay within the terms of service of Alibaba Cloud Model Studio, which generated or processed most of the training data.

This project builds on:
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache-2.0), the base model;
- [StepGuard](https://github.com/zheng977/StepGuard) model and [data](https://huggingface.co/datasets/ninty-seven/stepguard_data) (Apache-2.0; Zheng et al. 2026, arXiv:2608.24777): training data, the decision rules and risk categories of the guard prompt, and the main baseline;
- [AgentDojo](https://github.com/ethz-spylab/agentdojo) (MIT; Debenedetti et al. 2024) and [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (MIT; Zhan et al. 2024).

Benchmarks used for evaluation only, and not redistributed:
- [TS-Bench](https://github.com/MurrayTom/ToolSafe) (ToolSafe, arXiv:2601.10156);
- [ATBench](https://huggingface.co/datasets/AI45Research/ATBench) (Apache-2.0, arXiv:2604.02022);
- [R-Judge](https://github.com/Lordog/R-Judge) (arXiv:2401.10019; no license declared);
- [ASSEBench](https://github.com/Astarojth/AgentAuditor-ASSEBench) (AgentAuditor, arXiv:2506.00641).

TS-Bench and ASSEBench include data derived from [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm), whose license restricts use to improving the safety and security of AI systems.
