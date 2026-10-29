"""StepGuard 基线的提示词拼接和判定解析（不加载模型）。

    .venv-serve/bin/python -m pytest tests/test_score_stepguard.py -q
"""
import math

from intentanchor.eval.score_stepguard import build_prompt, hard_judgment, judgment_score

INP = {"user_task": "Book the hotel.",
       "tools": [{"name": "reserve_hotel", "description": "Reserve a hotel.", "side_effect": True}],
       "history": [{"tool": "get_hotels", "args": {"city": "Paris"}, "thought": "Find hotels.", "output": "[X, Y]"},
                   {"user": "The cheaper one."}],
       "proposed_call": {"tool": "reserve_hotel", "args": {"hotel": "X" * 1000}, "thought": None}}


def test_prompt_uses_stepguard_layout_and_clips():
    text = build_prompt(INP)
    assert '<<User Request>>\nBook the hotel.' in text
    assert ('[AGENT]: [THOUGHT]: Find hotels.\n[ACTION]: {"name": "get_hotels", "arguments": {"city": "Paris"}}'
            '\n\n[ENVIRONMENT]: [X, Y]\n\n[USER]: The cheaper one.') in text
    assert '<<Action Under Review>>\n[AGENT]: [ACTION]: {"name": "reserve_hotel"' in text
    assert "- reserve_hotel: Reserve a hotel." in text and "X" * 1000 not in text
    assert "<<Interaction History>>\n(none)" in build_prompt({**INP, "history": []})


def test_judgment_parsing():
    tokens = ["<", "Jud", "gment", ">", "unsafe", "</"]       # "<Judgment>unsafe</"
    tops = [[]] * len(tokens)
    tops[4] = [("unsafe", math.log(0.6)), (" safe", math.log(0.2)), ("x", math.log(0.2))]
    assert math.isclose(judgment_score(tokens, tops), 0.75)
    assert judgment_score(["x", "x"], [[], []]) is None
    assert hard_judgment("<Analysis>a</Analysis>\n<Judgment>Unsafe</Judgment>") == "unsafe"
    assert hard_judgment("no tags") is None
