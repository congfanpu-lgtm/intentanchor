"""大模型裁判的打分和解析（用假客户端，不调 API）。

    .venv-serve/bin/python -m pytest tests/test_score_llm.py -q
"""
import math
from types import SimpleNamespace as NS

from intentanchor.eval.score_llm import judge

INP = {"user_task": "Book the hotel.", "tools": [{"name": "reserve_hotel", "description": "", "side_effect": True}],
       "history": [], "context": None, "proposed_call": {"tool": "reserve_hotel", "args": {"hotel": "X"}}}


def fake_client(text: str, top: dict):
    lps = [NS(token=t, logprob=math.log(v)) for t, v in top.items()]
    resp = NS(choices=[NS(message=NS(content=text), logprobs=NS(content=[NS(top_logprobs=lps)]))],
              usage=NS(prompt_tokens=100, completion_tokens=3))
    return NS(chat=NS(completions=NS(create=lambda **kw: resp)))


def test_score_and_reason():
    r = judge(fake_client("BLOCK\nNot requested.", {"BLOCK": 0.6, " BLOCK": 0.1, "ALLOW": 0.1, "The": 0.2}),
              "m", INP, reason=True)
    assert math.isclose(r["score"], 0.7 / 0.8) and math.isclose(r["label_token_mass"], 0.8)
    assert (r["judge_label"], r["reason"]) == ("BLOCK", "Not requested.")


def test_no_label_tokens():
    r = judge(fake_client("The", {"The": 0.9, "I": 0.1}), "m", INP, reason=False)
    assert r["score"] is None and r["reason"] is None
