"""合成场景补工具返回的纯逻辑测试，不调 API。"""
import pytest

from intentanchor.data.fill_outputs import missing_values, needed_values, parse_outputs, scenario_messages


def test_parse_outputs_checks_count_and_stringifies():
    assert parse_outputs('```json\n{"outputs": ["ok", {"id": "a1"}]}\n```', 2) == ["ok", '{"id": "a1"}']
    with pytest.raises(ValueError):
        parse_outputs('{"outputs": ["only one"]}', 2)
    with pytest.raises(ValueError):
        parse_outputs('{"outputs": ["", "x"]}', 2)


def test_messages_split_candidates_by_label():
    base = {"lang": "en", "input": {"user_task": "Pay invoice 7", "history": [{"tool": "get_invoice", "args": {"id": 7}}]}}
    group = [{**base, "label": "ALLOW", "input": {**base["input"], "proposed_call": {"tool": "pay", "args": {"acct": "X1"}}}},
             {**base, "label": "BLOCK", "input": {**base["input"], "proposed_call": {"tool": "pay", "args": {"acct": "X2"}}}}]
    user = scenario_messages(group, [{"name": "pay", "description": "Pays."}])[1]["content"]
    covered, beyond = user.split("Next calls beyond the request:")
    assert '"X1"' in covered.split("Next calls the request covers:")[1] and '"X2"' in beyond


def test_needed_values_are_ids_not_in_request():
    s = {"input": {"user_task": "Refund order ORD-7 to sam@x.com", "history": [{"tool": "get_order", "args": {},
                                                                                "output": '{"card": "pm_42"}'}],
                   "proposed_call": {"tool": "refund", "args": {"order": "ORD-7", "to": "sam@x.com", "card": "pm_42",
                                                                "acct": "acc_99", "date": "2024-05-01",
                                                                "reason": "customer_cancel", "amount": 12.5,
                                                                "window": "14:00-17:00", "note": "请于20:30送达"}}}}
    assert needed_values(s) == ["acc_99", "pm_42"]   # 请求里有的、日期时间、不带数字的代码、中文文字、数字都不算
    assert missing_values(s) == ["acc_99"]
