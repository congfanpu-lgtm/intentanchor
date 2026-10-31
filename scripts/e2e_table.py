"""端到端测评汇总表（设计文档 §5.5.2）。每种配置要有 <前缀>-<配置>-attack 和 <前缀>-<配置>-benign 两个 run_attack_batch 输出目录：

    .venv-serve/bin/python scripts/e2e_table.py runs/2026-10-17-e2e none sft dpo stepguard
    .venv-serve/bin/python scripts/e2e_table.py runs/2026-10-28-e2e none sft h2 stepguard --compare h2:sft h2:stepguard

列：无攻击 utility、受攻击 utility、ASR（%）；ASR 不含放行 = 攻击成功、且这个用例没有因为拦满 3 轮而放行过调用的比例，
即守卫自己漏掉的（StepGuard 协议拦满 3 轮后放行，§5.5.2）；误拦用例 = 无攻击用例里守卫拦过至少一次的比例，
拦截 = 两批合计拦下的轮数（超过 3 轮后放行的另计），终止 = 拦满 3 轮后被终止的用例数（--on-max stop），每次检查的平均耗时（多线程并发下，含排队），花费（元），报错用例数。
--compare A:B 做配对检验：ASR、不含放行、受攻击 utility 按同一用例配对，用 McNemar 精确检验；误拦用例用 Fisher 精确检验
（和 2026-10-18 的检验口径相同）。
"""
import argparse
import json
from math import comb
from pathlib import Path

from mcnemar import mcnemar

from intentanchor.victim import cost_cny


def load(run: Path) -> list[dict]:
    recs = {}  # 断点续跑会重复写同一个用例，取最后一条
    for line in (run / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        recs[(r["suite"], r["user_task"], r["injection_task"])] = r
    return list(recs.values())


def pct(xs) -> str:
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):5.1f}" if xs else "    -"


def fisher(a: int, n1: int, c: int, n2: int) -> float:
    """双侧 Fisher 精确检验：a / n1 对 c / n2。"""
    tot, k = a + c, n1 + n2
    p = lambda x: comb(n1, x) * comb(n2, tot - x) / comb(k, tot)  # noqa: E731
    return sum(p(x) for x in range(max(0, tot - n2), min(n1, tot) + 1) if p(x) <= p(a) + 1e-12)


def compare(prefix: str, x: str, y: str) -> None:
    key = lambda r: (r["suite"], r["user_task"], r["injection_task"])  # noqa: E731
    att = [{key(r): r for r in load(Path(f"{prefix}-{c}-attack")) if r["status"] == "ok"} for c in (x, y)]
    common = att[0].keys() & att[1].keys()
    for name, f in (("ASR", lambda r: r["security"]), ("不含放行", lambda r: r["security"] and not r.get("guard_over_max")),
                    ("受攻击 utility", lambda r: r["utility"])):
        b = sum(bool(f(att[0][k])) and not f(att[1][k]) for k in common)
        c = sum(not f(att[0][k]) and bool(f(att[1][k])) for k in common)
        print(f"{x} 对 {y} {name}：只有 {x} 为真 {b}，只有 {y} 为真 {c}，McNemar p = {mcnemar(b, c):.3f}")
    ben = [[r for r in load(Path(f"{prefix}-{c}-benign")) if r["status"] == "ok"] for c in (x, y)]
    fx, fy = (sum(r.get("guard_blocks", 0) > 0 for r in bs) for bs in ben)
    print(f"{x} 对 {y} 误拦用例：{fx}/{len(ben[0])} 对 {fy}/{len(ben[1])}，"
          f"Fisher p = {fisher(fx, len(ben[0]), fy, len(ben[1])):.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--compare", nargs="*", default=[], help="配置A:配置B")
    args = ap.parse_args()
    prefix, configs = args.prefix, args.configs
    print(f"{'config':10s} {'suite':8s} utility util@atk   ASR  不含放行  误拦用例  拦截(超限)  终止  ms/检查   花费  报错")
    for c in configs:
        benign, attack = load(Path(f"{prefix}-{c}-benign")), load(Path(f"{prefix}-{c}-attack"))
        for suite in ("banking", "slack", "ALL"):
            b = [r for r in benign if suite in ("ALL", r["suite"])]
            a = [r for r in attack if suite in ("ALL", r["suite"])]
            bo, ao = [r for r in b if r["status"] == "ok"], [r for r in a if r["status"] == "ok"]
            n = lambda k: sum(r.get(k, 0) for r in b + a)  # noqa: E731
            ms = f"{n('guard_ms') / n('guard_checks'):8.0f}" if n("guard_checks") else "       -"
            cost = cost_cny({k: n(k) for k in ("prompt", "completion")})
            print(f"{c:10s} {suite:8s}   {pct(r['utility'] for r in bo)}   {pct(r['utility'] for r in ao)} "
                  f"{pct(r['security'] for r in ao)}    {pct(r['security'] and not r.get('guard_over_max') for r in ao)}"
                  f"    {pct(r.get('guard_blocks', 0) > 0 for r in bo)}"
                  f"  {n('guard_blocks'):5d} ({n('guard_over_max'):2d})  {n('guard_stop'):4d}  {ms} {cost:6.1f}  {len(b + a) - len(bo + ao):4d}")
        print()
    for pair in args.compare:
        compare(prefix, *pair.split(":"))


if __name__ == "__main__":
    main()
