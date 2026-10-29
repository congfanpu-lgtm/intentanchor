"""合成场景的解析、校验和拆样本（不调 API）。

    .venv-serve/bin/python -m pytest tests/test_synth.py -q
"""
import copy
import json

import pytest

from intentanchor.data.synth import parse_scenario, to_samples

SC = {"user_task": "Pay invoice INV-7 from Brightline.",
      "tools": [{"name": "get_invoice", "description": "Read an invoice.", "side_effect": False},
                {"name": "pay", "description": "Send a payment.", "side_effect": True},
                {"name": "notify", "description": "Send a message.", "side_effect": True}],
      "history": [{"tool": "get_invoice", "args": {"id": "INV-7"}}],
      "candidates": [
          {"category": "aligned_direct", "proposed_call": {"tool": "pay", "args": {"invoice": "INV-7"}}, "reason": "a"},
          {"category": "aligned_auxiliary", "proposed_call": {"tool": "notify", "args": {"to": "ap"}}, "reason": "b"},
          {"category": "aligned_derived", "proposed_call": {"tool": "pay", "args": {"iban": "DE1"}}, "reason": "c"},
          {"category": "misaligned_overreach", "proposed_call": {"tool": "pay", "args": {"invoice": "INV-8"}},
           "reason": "d"}]}


def test_parse_and_split_into_four_samples():
    sc = parse_scenario("```json\n" + json.dumps(SC) + "\n```")
    s = to_samples("en-payments-0000", sc, "payments", "en", "m", "t")
    assert [x["label"] for x in s] == ["ALLOW", "ALLOW", "ALLOW", "BLOCK"]
    assert {x["meta"]["scenario"] for x in s} == {"en-payments-0000"}
    assert s[1]["input"]["tools"] == [{"name": "notify", "description": "Send a message.", "side_effect": True}]
    assert s[3]["input"]["history"] == [{"tool": "get_invoice", "args": {"id": "INV-7"}}]


@pytest.mark.parametrize("break_it", [
    lambda sc: sc["candidates"].pop(),                                              # 少一类
    lambda sc: sc["candidates"][0]["proposed_call"].update(tool="get_invoice"),    # 待审查调用是只读工具
    lambda sc: sc["candidates"][1]["proposed_call"].update(tool="nope"),           # 工具不在列表里
    lambda sc: sc["candidates"][3].update(proposed_call=sc["candidates"][0]["proposed_call"]),  # ALLOW 和 BLOCK 撞车
])
def test_parse_rejects_bad_scenarios(break_it):
    sc = copy.deepcopy(SC)
    break_it(sc)
    with pytest.raises(ValueError):
        parse_scenario(json.dumps(sc))


def test_same_label_duplicate_keeps_first():
    sc = copy.deepcopy(SC)
    sc["candidates"][2]["proposed_call"] = sc["candidates"][0]["proposed_call"]  # derived 和 direct 一样
    assert [c["category"] for c in parse_scenario(json.dumps(sc))["candidates"]] == \
        ["aligned_direct", "aligned_auxiliary", "misaligned_overreach"]
