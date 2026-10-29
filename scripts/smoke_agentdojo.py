"""冒烟测试 5（设计文档 §7.1）：被测模型跑 1 个 AgentDojo 用户任务（无防御、无攻击），记录 token 用量。

    set -a && . ./.env && set +a
    python scripts/smoke_agentdojo.py --suite workspace --task user_task_0
"""
import argparse
import os
from collections import Counter

import openai
from agentdojo.agent_pipeline import (AgentPipeline, InitQuery, OpenAILLM, SystemMessage, ToolsExecutionLoop,
                                      ToolsExecutor)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.task_suite.load_suites import get_suite

MODEL = "qwen3.6-35b-a3b"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼，华北2（北京）
PRICE_CNY_PER_M = {"prompt": 1.8, "completion": 10.8}  # 原价，2026-10 查的官方文档
BENCHMARK_VERSION = "v1.2.2"

p = argparse.ArgumentParser()
p.add_argument("--suite", default="workspace")
p.add_argument("--task", default="user_task_0")
a = p.parse_args()

client = openai.OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL)
usage = Counter()
_create = client.chat.completions.create


def create(**kwargs):
    # AgentDojo 把 system 一律写成 OpenAI 的 developer 角色，百炼不认，换回 system
    kwargs["messages"] = [{**m, "role": "system"} if m["role"] == "developer" else m for m in kwargs["messages"]]
    # AgentDojo 不支持传额外参数；百炼非流式调用必须关思考
    r = _create(**kwargs, extra_body={"enable_thinking": False})
    usage.update(calls=1, prompt=r.usage.prompt_tokens, completion=r.usage.completion_tokens)
    return r


client.chat.completions.create = create

llm = OpenAILLM(client, MODEL, temperature=0.0)
pipeline = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm,
                          ToolsExecutionLoop([ToolsExecutor(), llm])])
pipeline.name = MODEL

for name in ["workspace", "travel", "banking", "slack"]:
    s = get_suite(BENCHMARK_VERSION, name)
    print(f"{name}: {len(s.user_tasks)} user tasks, {len(s.injection_tasks)} injection tasks")

suite = get_suite(BENCHMARK_VERSION, a.suite)
task = suite.get_user_task_by_id(a.task)
print(f"\n[{a.suite}/{a.task}] {task.PROMPT}")
utility, _ = suite.run_task_with_pipeline(pipeline, task, injection_task=None, injections={})

cost = sum(usage[k] * PRICE_CNY_PER_M[k] for k in PRICE_CNY_PER_M) / 1e6
print(f"utility={utility}  LLM 调用 {usage['calls']} 次，输入 {usage['prompt']} / 输出 {usage['completion']} token，"
      f"约 {cost:.4f} 元")
