"""DPO 偏好对的测试。

    .venv-serve/bin/python -m pytest tests/test_dpo.py -q
"""
from intentanchor.train.dpo import to_pair

INP = {"user_task": "Pay my rent.", "tools": [], "history": [],
       "proposed_call": {"tool": "send_money", "args": {"to": "landlord"}, "thought": None}}


def test_pair_flips_only_the_label():
    for label, wrong in (("ALLOW", "BLOCK"), ("BLOCK", "ALLOW")):
        pair = to_pair({"label": label, "risk_source": "corrupted_tool_feedback", "reason": "ignored", "input": INP})
        assert pair["chosen"] == [{"role": "assistant", "content": f"{label}\nRisk source: corrupted_tool_feedback"}]
        assert pair["rejected"] == [{"role": "assistant", "content": f"{wrong}\nRisk source: corrupted_tool_feedback"}]
        assert pair["prompt"][-1]["role"] == "user" and "Pay my rent." in pair["prompt"][-1]["content"]
