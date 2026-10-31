"""守卫 pipeline 元素的拦截协议（不连服务，用假打分函数）。

    .venv-serve/bin/python -m pytest tests/test_guard.py -q
"""
from collections import Counter
from types import SimpleNamespace

from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import text_content_block_from_string

from intentanchor import victim
from intentanchor.victim import GuardElement


def _assistant(*calls):
    return {"role": "assistant", "content": [text_content_block_from_string("Paying the bill.")],
            "tool_calls": [FunctionCall(function=f, args=a, id=i) for f, a, i in calls]}


RUNTIME = SimpleNamespace(functions={n: SimpleNamespace(name=n, description=f"{n} tool.")
                                     for n in ("get_balance", "send_money")})
HEAD = [{"role": "system", "content": [text_content_block_from_string("sys")]},
        {"role": "user", "content": [text_content_block_from_string("Pay my bill.")]},
        _assistant(("get_balance", {}, "1")),
        {"role": "tool", "content": [text_content_block_from_string("100")], "tool_call_id": "1",
         "tool_call": FunctionCall(function="get_balance", args={}, id="1"), "error": None}]


def test_blocks_whole_message_then_caps_replans():
    seen = []
    guard = GuardElement(lambda inp: seen.append(inp) or 0.9, threshold=0.5)
    victim._local.counter, victim._local.guard = Counter(), []
    last = _assistant(("get_balance", {}, "2"), ("send_money", {"recipient": "X", "amount": 5}, "3"))

    _, _, _, out, extra = guard.query("Pay my bill.", RUNTIME, messages=[*HEAD, last])
    assert [i["proposed_call"]["tool"] for i in seen] == ["send_money"]  # 只读调用不检查
    assert seen[0]["history"][0]["output"] == "100" and seen[0]["user_task"] == "Pay my bill."
    assert out[:-1] == HEAD and out[-1]["guard_feedback"] and extra["guard_replans"] == 1

    _, _, _, out, _ = guard.query("Pay my bill.", RUNTIME, messages=[*out, last], extra_args=extra)
    assert sum(bool(m.get("guard_feedback")) for m in out) == 1      # 上一轮的反馈删掉了

    _, _, _, out, extra = guard.query("Pay my bill.", RUNTIME, messages=[*HEAD, last], extra_args={"guard_replans": 3})
    assert out[-1] is last and extra["guard_replans"] == 4             # 拦满 3 轮后放行
    assert victim._local.counter["guard_blocks"] == 2 and victim._local.counter["guard_over_max"] == 1


def test_stop_mode_ends_task_on_fourth_block():
    guard = GuardElement(lambda inp: 0.9, threshold=0.5, on_max="stop")
    victim._local.counter, victim._local.guard = Counter(), []
    last = _assistant(("send_money", {"recipient": "X", "amount": 5}, "3"))
    _, _, _, out, extra = guard.query("Pay my bill.", RUNTIME, messages=[*HEAD, last], extra_args={"guard_replans": 2})
    assert out[-1]["guard_feedback"] and extra["guard_replans"] == 3       # 第 3 次拦截还是重新规划
    try:
        guard.query("Pay my bill.", RUNTIME, messages=[*HEAD, last], extra_args=extra)
        raise AssertionError("4th block should stop the task")
    except victim.AbortAgentError as e:
        assert e.messages[:-1] == HEAD and not e.messages[-1]["tool_calls"]  # 被拦的调用没执行，以助手消息收尾
    c = victim._local.counter
    assert c["guard_blocks"] == 2 and c["guard_stop"] == 1 and c["guard_over_max"] == 0
