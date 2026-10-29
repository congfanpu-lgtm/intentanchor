"""守卫推理服务（设计文档 §5.6）：FastAPI 做输入校验、拼提示词、打分和阈值判断；模型由本地 vLLM 的 OpenAI 兼容服务提供。

    VLLM_USE_FLASHINFER_SAMPLER=0 .venv-serve/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 18000 \
        --enable-lora --max-lora-rank 16 --lora-modules guard=runs/<H2>/best \
        --max-model-len 6144 --gpu-memory-utilization 0.85 --enable-prefix-caching
    GUARD_THRESHOLD=0.755 .venv-serve/bin/uvicorn intentanchor.serve:app --port 18080

    POST /v1/check  请求：§4.1 的输入 JSON，外加 "explain": true | false
                    响应：{"verdict", "score", "latency_ms"}，explain 时加 "risk_source"、"reason"

explain=false 只生成 1 个 token；explain=true 生成全文（第二行风险来源、第三行理由），分数都取第一个 token。
阈值用开发集 FPR 5% 处的值，每个模型不同（§5.2.4），通过 GUARD_THRESHOLD 传入。
"""
import os
import time
from typing import Any

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, model_validator

from intentanchor.guard import TOP_LOGPROBS, label_score
from intentanchor.prompt import build_messages

MODEL = os.environ.get("GUARD_MODEL", "guard")
THRESHOLD = float(os.environ.get("GUARD_THRESHOLD", "0.5"))
client = OpenAI(base_url=os.environ.get("GUARD_VLLM_URL", "http://localhost:18000/v1"), api_key="EMPTY")
app = FastAPI(title="IntentAnchor guard")


class Step(BaseModel):
    """history 的一步：工具调用、给用户的回复、用户的后续消息，三选一（§4.1、prompt.py）。"""
    tool: str | None = None
    args: dict = {}
    output: Any = None
    thought: str | None = None
    response: str | None = None
    user: str | None = None

    @model_validator(mode="after")
    def one_kind(self):
        if sum(x is not None for x in (self.tool, self.response, self.user)) != 1:
            raise ValueError("a step needs exactly one of tool / response / user")
        return self

    def as_input(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Tool(BaseModel):
    name: str
    description: str = ""


class CheckRequest(BaseModel):
    user_task: str
    tools: list[Tool] = []
    history: list[Step] = []
    proposed_call: Step
    explain: bool = False

    @model_validator(mode="after")
    def reviewable(self):
        if self.proposed_call.user is not None:
            raise ValueError("proposed_call must be a tool call or a reply")
        return self


@app.post("/v1/check")
def check(req: CheckRequest):  # 简化：同步函数，FastAPI 放进线程池（默认 40 线程）；并发超过 40 再改 AsyncOpenAI
    t0 = time.perf_counter()
    inp = {"user_task": req.user_task, "tools": [t.model_dump() for t in req.tools],
           "history": [s.as_input() for s in req.history], "proposed_call": req.proposed_call.as_input()}
    c = client.chat.completions.create(model=MODEL, messages=build_messages(inp), max_tokens=128 if req.explain else 1,
                                       temperature=0, logprobs=True, top_logprobs=TOP_LOGPROBS).choices[0]
    score = label_score(c)
    out = {"verdict": "BLOCK" if score >= THRESHOLD else "ALLOW", "score": round(score, 4)}
    if req.explain:
        lines = (c.message.content or "").splitlines() + ["", ""]
        out["risk_source"] = lines[1].removeprefix("Risk source:").strip() or None
        out["reason"] = lines[2].strip() or None
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
