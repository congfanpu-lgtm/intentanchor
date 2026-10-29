"""标签审计的抽样和计分（不读真实数据）。

    .venv-serve/bin/python -m pytest tests/test_audit.py -q
"""
from intentanchor.data.audit import sample, score, wilson


def mk(i, label="ALLOW", label_source="synthetic"):
    return {"id": i, "label": label, "label_source": label_source, "lang": "en", "category": "c"}


def test_sample_roles_disjoint():
    rows = [mk(f"s{i}") for i in range(50)] + [mk(f"g{i}", label_source="stepguard") for i in range(30)]
    out = sample(rows, 20, 100)
    ids = [s["id"] for s in out]
    assert len(ids) == len(set(ids))
    extra = [s for s in out if s["audit_role"] == "claude"]
    assert sum(s["audit_role"] == "human" for s in out) == 20
    assert extra and all(s["label_source"] == "stepguard" for s in extra)
    assert len(extra) == 30 - sum(s["audit_role"] == "human" and s["label_source"] == "stepguard" for s in out)
    # v2 的用法：另抽的是全体随机样本，不和人工那部分重叠
    out = sample(rows, 10, 0, n_extra_random=40)
    assert len({s["id"] for s in out}) == 50 and sum(s["audit_role"] == "claude" for s in out) == 40


def test_score_blind_labels_and_agreement():
    samples = {i: {**mk(i, lab), "audit_role": "human"} for i, lab in [("a", "ALLOW"), ("b", "BLOCK"), ("c", "ALLOW")]}
    verdicts = [{"id": "a", "reviewer": "human", "verdict": "correct"},
                {"id": "b", "reviewer": "human", "verdict": "wrong", "note": "user asked for it"},
                {"id": "c", "reviewer": "human", "verdict": "unsure"},
                {"id": "a", "reviewer": "claude", "label": "ALLOW"},
                {"id": "b", "reviewer": "claude", "label": "BLOCK"},
                {"id": "c", "reviewer": "claude", "label": "BLOCK"}]
    out = score(samples, verdicts)
    assert out["human"]["all"]["accuracy"] == 0.5 and out["human"]["all"]["unsure"] == 1
    assert out["human"]["wrong"] == [{"id": "b", "label": "BLOCK", "category": "c", "note": "user asked for it"}]
    assert out["claude"]["all"]["correct"] == 2 and out["claude"]["all"]["wrong"] == 1
    assert out["agreement:claude~human"] == {"n": 2, "agree": 1, "rate": 0.5}  # c 人工拿不准，不计


def test_wilson():
    lo, hi = wilson(90, 100)
    assert 0.82 < lo < 0.83 and 0.94 < hi < 0.95
