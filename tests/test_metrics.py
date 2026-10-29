"""单步分类指标的测试，用能手算的小例子。

    .venv-serve/bin/python -m pytest tests/test_metrics.py -q
"""
import math

from intentanchor.eval.metrics import at_threshold, auprc, auroc, bootstrap_ci, recall_at_fpr, threshold_at_fpr

Y = [0, 0, 0, 0, 1, 1]


def test_auroc_perfect_reversed_and_ties():
    assert auroc(Y, [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]) == 1.0
    assert auroc(Y, [0.9, 0.8, 0.7, 0.6, 0.2, 0.1]) == 0.0
    assert auroc(Y, [0.5] * 6) == 0.5
    assert auroc(Y, [0.1, 0.2, 0.3, 0.9, 0.8, 0.95]) == 0.875  # 一个负样本排在一个正样本前面：7/8
    assert math.isnan(auroc([0, 0], [0.1, 0.2]))


def test_auprc():
    assert auprc(Y, [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]) == 1.0
    assert abs(auprc(Y, [0.1, 0.2, 0.3, 0.9, 0.8, 0.95]) - (1 + 2 / 3) / 2) < 1e-9  # 精度 1/1、2/3


def test_recall_at_fpr_never_exceeds_target_fpr():
    y = [0] * 100 + [1] * 4
    s = [i / 100 for i in range(100)] + [0.995, 0.985, 0.975, 0.5]
    assert recall_at_fpr(y, s, 0.0) == 0.25    # 不许误报：只能拦住高于所有负样本的 0.995
    assert recall_at_fpr(y, s, 0.03) == 0.75   # 允许 3 个负样本（0.99、0.98、0.97）误报
    assert recall_at_fpr(y, s, 0.6) == 1.0
    t = threshold_at_fpr(y, s, 0.03)
    assert at_threshold(y, s, t)["fpr"] == 0.03 and at_threshold(y, s, 0.97)["fpr"] == 0.03  # 0.97 本身也只误报 3 个
    assert t > 0.96 and at_threshold(y, s, 0.96)["fpr"] == 0.04  # 再低一点就超了


def test_at_threshold_and_bootstrap():
    m = at_threshold(Y, [0.1, 0.2, 0.3, 0.6, 0.4, 0.9], 0.5)
    assert (m["recall"], m["fpr"], m["precision"]) == (0.5, 0.25, 0.5)
    lo, hi = bootstrap_ci(auroc, Y * 20, [0.1, 0.2, 0.3, 0.4, 0.8, 0.9] * 20, n=200)
    assert lo == hi == 1.0


def test_rescore_by_case():
    from intentanchor.eval.rescore import by_case
    steps = [{"id": f"c1-a{k}", "meta": {"case": "c1", "step": k}} for k in (1, 2, 3)] + \
            [{"id": "c2-a1", "meta": {"case": "c2", "step": 1}}]
    scores = {"c1-a1": 0.2, "c1-a2": 0.9, "c1-a3": 0.1, "c2-a1": 0.4}
    reps, agg = by_case(steps, scores, "max")
    assert [r["id"] for r in reps] == ["c1-a3", "c2-a1"] and agg == {"c1-a3": 0.9, "c2-a1": 0.4}
    assert by_case(steps, scores, "last")[1] == {"c1-a3": 0.1, "c2-a1": 0.4}
