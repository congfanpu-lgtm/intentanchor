"""守卫推理客户端（设计文档 §5.5.2、§5.6）：把 AgentDojo 的消息拼成 §4.1 的输入，请求本地 vLLM 的 OpenAI 兼容服务打分。

    # 本项目守卫：基座 + 多个 LoRA，请求里的 model 填 LoRA 名
    VLLM_USE_FLASHINFER_SAMPLER=0 .venv-serve/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 18000 \
        --enable-lora --max-lora-rank 16 --lora-modules sft=runs/<SFT>/best dpo=runs/<DPO>/final \
        --max-model-len 6144 --gpu-memory-utilization 0.85 --enable-prefix-caching
    # StepGuard
    VLLM_USE_FLASHINFER_SAMPLER=0 .venv-serve/bin/vllm serve ninty-seven/StepGuard --port 18000 \
        --max-model-len 6144 --gpu-memory-utilization 0.85

风险分和离线打分相同：本项目守卫取第一个 token 上 P(BLOCK) / (P(ALLOW) + P(BLOCK))（score_vllm），
StepGuard 取 <Judgment> 后第一个 token 上 unsafe / (safe + unsafe)，取不到按硬判定给 0/1（score_stepguard）。
"""
import math
from pathlib import Path

import yaml

from intentanchor.data.split_label import split_trace
from intentanchor.eval.score_stepguard import build_prompt, hard_judgment, judgment_score
from intentanchor.prompt import LABELS, build_messages

TOP_LOGPROBS = 20


def side_effect_tools(path: Path = Path("configs/side_effect.yaml")) -> dict[str, bool]:
    """把各 suite 的副作用标注合成一张表；同名工具在不同 suite 里标得不一样就报错。"""
    merged = {}
    for tools in yaml.safe_load(path.read_text()).values():
        for name, se in tools.items():
            assert merged.setdefault(name, se) == se, f"{name} is labeled differently across suites"
    return merged


def guard_inputs(query: str, docs: dict[str, str], messages: list[dict], side_effect: dict[str, bool]) -> list[dict]:
    """messages 是可 JSON 序列化的 AgentDojo 消息，最后一条是带 tool_calls 的 assistant 消息。
    返回其中每个副作用调用的守卫输入（§4.1），顺序同 tool_calls；历史的拼法和训练数据相同（split_trace）。"""
    k = sum(side_effect.get(c["function"], True) for c in messages[-1]["tool_calls"])
    steps = list(split_trace({"messages": messages}, side_effect=side_effect))[-k:] if k else []
    return [{"user_task": query, "tools": [{"name": c["tool"], "description": docs.get(c["tool"], "")}],
             "history": history, "proposed_call": c} for _, history, c in steps]


def label_score(choice) -> float:
    """第一个生成 token 上的 P(BLOCK) / (P(ALLOW) + P(BLOCK))；两个都不在 top-k 里记 0.5（同 score_vllm）。"""
    top = {t.token: math.exp(t.logprob) for t in choice.logprobs.content[0].top_logprobs}
    pa, pb = (top.get(w, 0.0) for w in LABELS)
    return pb / (pa + pb) if pa + pb else 0.5


def intentanchor_scorer(client, model: str):
    def score(inp: dict) -> float:
        r = client.chat.completions.create(model=model, messages=build_messages(inp), max_tokens=1, temperature=0,
                                           logprobs=True, top_logprobs=TOP_LOGPROBS)
        return label_score(r.choices[0])
    return score


def stepguard_scorer(client, model: str = "ninty-seven/StepGuard"):
    def score(inp: dict) -> float:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": build_prompt(inp)}],
                                           max_tokens=768, temperature=0, logprobs=True, top_logprobs=TOP_LOGPROBS)
        c = r.choices[0]
        s = judgment_score([t.token for t in c.logprobs.content],
                           [[(x.token, x.logprob) for x in t.top_logprobs] for t in c.logprobs.content])
        return s if s is not None else float(hard_judgment(c.message.content or "") == "unsafe")
    return score
