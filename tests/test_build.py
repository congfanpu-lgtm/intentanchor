"""数据集 v1 合并的纯逻辑：裁判把关、InjecAgent 截断、去重、分组（不读真实数据）。

    .venv-serve/bin/python -m pytest tests/test_build.py -q
"""
from intentanchor.data.build import (apply_relabels, assign_groups, cap_injecagent, dedup, drop_orphan_translations,
                                     gate_synthetic, is_dev, load_verdicts)


def mk(i, label="ALLOW", source="synthetic", label_source="synthetic", lang="en", task="t", meta=None, suite=None):
    return {"id": i, "label": label, "source": source, "label_source": label_source, "lang": lang, "suite": suite,
            "meta": meta or {}, "input": {"user_task": task, "history": [], "proposed_call": {"tool": i, "args": {}}}}


def test_gate_synthetic():
    s = [mk("a"), mk("b", "BLOCK"), mk("c"), mk("d", label_source="auto")]
    assert gate_synthetic(s, {"a": 0.1, "b": 0.2}) == 2          # b 裁判不同意，c 没有分数
    assert [x["label"] for x in s] == ["ALLOW", None, None, "ALLOW"]
    assert s[1]["meta"]["constructed_label"] == "BLOCK" and s[1]["label_source"] == "pending"


def test_cap_injecagent_keeps_k_per_attacker_case():
    s = [mk(f"ia-{c}-{u}", "BLOCK", "public_attack", "injecagent", meta={"attacker_case": c})
         for c in ("dh00", "ds01") for u in range(17)]
    kept = cap_injecagent(s + [mk("z")], 5)
    assert len(kept) == 11 and kept[-1]["id"] == "z"


def test_dedup_drops_conflicts():
    a, b, c = mk("a"), mk("b"), mk("c", "BLOCK")
    b["input"] = a["input"]
    c["input"] = {"user_task": "x", "history": [], "proposed_call": {}}
    d = mk("d")
    d["input"] = c["input"]
    out, stats = dedup([a, b, c, d])
    assert [x["id"] for x in out] == ["a"] and stats == {"dropped_duplicate": 1, "dropped_conflicting": 2}


def test_translation_follows_original_group():
    orig = mk("adojo-1", source="agentdojo", label_source="spliced", task="Book a hotel", suite="travel")
    zh = mk("adojo-1-zh", source="agentdojo", label_source="spliced", task="订酒店", suite="travel",
            meta={"translated_from": "adojo-1"}, lang="zh")
    assign_groups([orig, zh])
    assert orig["meta"]["group"] == zh["meta"]["group"] == "adojo:travel:Book a hotel"
    assert is_dev("adojo:travel:Book a hotel", 15) == is_dev(zh["meta"]["group"], 15)


def test_stepguard_orig_id_is_not_a_translation_marker():
    orig = mk("stepguard-sft-1", source="stepguard", label_source="stepguard", task="Send report", meta={"orig_id": "sft-1"})
    zh = mk("stepguard-sft-1-zh", source="stepguard", label_source="stepguard", task="发送报告", lang="zh",
            meta={"orig_id": "sft-1", "translated_from": "stepguard-sft-1"})
    assign_groups([orig, zh])
    assert zh["meta"]["group"] == orig["meta"]["group"] == "stepguard:Send report"


def test_relabel_human_overrides_claude_and_translation_follows():
    a, b, c = mk("a", "BLOCK", "stepguard", "stepguard"), mk("b", "ALLOW"), mk("c", "BLOCK", "stepguard", "stepguard")
    a_zh = mk("a-zh", "BLOCK", "stepguard", "stepguard", lang="zh", meta={"translated_from": "a"})
    labels = {s["id"]: s["label"] for s in (a, b, c, a_zh)}
    verdicts = load_verdicts([{"id": "a", "label": "ALLOW"}, {"id": "b", "label": "BLOCK"}, {"id": "c", "label": "UNSURE"}],
                             labels)
    verdicts.update(load_verdicts([{"id": "b", "verdict": "correct"}], labels))  # 人工覆盖 Claude
    stats = apply_relabels([a, b, c, a_zh], verdicts)
    assert stats == {"to_ALLOW": 2, "confirmed": 1, "to_UNSURE": 1}
    assert (a["label"], a["category"], a["label_source"], a["meta"]["original_label"]) == \
        ("ALLOW", "aligned_unspecified", "audit", "BLOCK")
    assert a_zh["label"] == "ALLOW" and b["label"] == "ALLOW" and b["label_source"] == "synthetic"
    assert c["label"] is None and c["label_source"] == "pending"
    assert all(s["audited"] for s in (a, b, c, a_zh))


def test_orphan_translation_dropped():
    a, b = mk("a"), mk("b")
    a_zh = mk("a-zh", lang="zh", meta={"translated_from": "a"})
    x_zh = mk("x-zh", lang="zh", meta={"translated_from": "x"})
    assert [s["id"] for s in drop_orphan_translations([a, b, a_zh, x_zh])] == ["a", "b", "a-zh"]
