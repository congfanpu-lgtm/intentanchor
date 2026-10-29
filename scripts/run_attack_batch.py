"""AgentDojo 批量运行：被测模型走百炼，逐个用例记录结果、token 用量、错误和完整轨迹，可断点续跑。

    set -a && . ./.env && set +a
    .venv-serve/bin/python scripts/run_attack_batch.py --thinking on --suites banking --out runs/x            # 无防御、带攻击
    .venv-serve/bin/python scripts/run_attack_batch.py --thinking on --suites banking --defense tool_filter --out runs/y
    .venv-serve/bin/python scripts/run_attack_batch.py --thinking on --suites banking --attack none --out runs/z  # 无攻击 utility
    .venv-serve/bin/python scripts/run_attack_batch.py --thinking on --suites banking --guard sft --threshold 0.679 --out runs/w
    （--guard 填本地 vLLM 服务里的模型名，比如 LoRA 名 sft；填 stepguard 就用 StepGuard 的格式，服务的启动命令见 intentanchor.guard）
    加 --limit 10 先小批量估成本；--inj-per-user 2 让每个用户任务只配 2 个注入任务，用来低成本筛查。

输出：results.jsonl（每例结果）、traces.jsonl（每例完整轨迹和注入文本，供 intentanchor.data 拆样本）、summary.json。
security=True 表示注入目标达成（攻击成功），ASR = 状态为 ok 且有注入的用例里 security 的比例。
API 报错（比如被百炼内容审核拦下）的用例单独记为 bad_request / error，不计入 ASR，在汇总里单列。
"""
import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from agentdojo.agent_pipeline.agent_pipeline import DEFENSES
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.task_suite.load_suites import get_suite

from intentanchor.guard import intentanchor_scorer, stepguard_scorer
from intentanchor.victim import BENCHMARK_VERSION, MODEL, GuardElement, cost_cny, make_client, make_pipeline, run_case

TRACE_KEYS = ("injections", "messages", "guard")
GUARD_KEYS = ("guard_checks", "guard_blocks", "guard_over_max", "guard_ms")


def run_one(suite, attack, pipeline, ut_id: str, it_id: str | None) -> dict:
    try:
        injections = attack.attack(suite.get_user_task_by_id(ut_id), suite.get_injection_task_by_id(it_id)) \
            if it_id else {}
    except Exception as e:  # noqa: BLE001
        return {"suite": suite.name, "user_task": ut_id, "injection_task": it_id, "status": "error",
                "error": f"attack: {type(e).__name__}: {e}"[:400], "seconds": 0}
    return run_case(suite, pipeline, ut_id, it_id, injections)


def summarize(records: list[dict]) -> dict:
    out = {}
    for name in sorted({r["suite"] for r in records}) + ["ALL"]:
        rs = [r for r in records if name in ("ALL", r["suite"])]
        ok = [r for r in rs if r["status"] == "ok"]
        attacked = [r for r in ok if r["security"] is not None]
        tok = Counter()
        for r in rs:
            tok.update({k: r.get(k, 0) for k in ("calls", "prompt", "completion", *GUARD_KEYS)})
        cost = cost_cny(tok)
        out[name] = {
            "cases": len(rs), "ok": len(ok), "errors": Counter(r["status"] for r in rs if r["status"] != "ok"),
            "ASR": round(sum(r["security"] for r in attacked) / len(attacked), 4) if attacked else None,
            "utility": round(sum(r["utility"] for r in ok) / len(ok), 4) if ok else None,  # 带攻击时即受攻击 utility
            **tok, "cost_cny": round(cost, 3), "cost_cny_per_case": round(cost / len(rs), 4) if rs else None,
            "avg_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 1) if rs else None,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thinking", choices=["on", "off"], required=True)
    p.add_argument("--suites", default="workspace,travel,banking,slack")
    p.add_argument("--n-user", type=int, default=None, help="每个 suite 取前 N 个用户任务（默认全部）")
    p.add_argument("--limit", type=int, default=None, help="总共最多跑多少个用例（各 suite 轮流取）")
    p.add_argument("--inj-per-user", type=int, default=None,
                   help="每个用户任务只配 K 个注入任务，按序号轮换，覆盖全部注入任务（默认全配）")
    p.add_argument("--attack", default="important_instructions", help="none = 不注入，只测 utility")
    p.add_argument("--defense", default="none", choices=["none", *DEFENSES])
    p.add_argument("--guard", help="本地守卫服务里的模型名（LoRA 名），或 stepguard")
    p.add_argument("--threshold", type=float, help="守卫的拦截阈值（开发集 FPR 5%% 处；StepGuard 用它自己的 0.5）")
    p.add_argument("--guard-url", default="http://localhost:18000/v1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.guard and (a.threshold is None or a.defense != "none"):
        p.error("--guard needs --threshold and no --defense")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(a), "model": MODEL, "benchmark": BENCHMARK_VERSION}, indent=2))
    results_path = out / "results.jsonl"
    done = {}
    if results_path.exists():  # 断点续跑：status=ok 的不再重跑
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            done[(r["suite"], r["user_task"], r["injection_task"])] = r

    guard = None
    if a.guard:
        gc = openai.OpenAI(base_url=a.guard_url, api_key="local", timeout=120, max_retries=2)
        guard = GuardElement(stepguard_scorer(gc) if a.guard == "stepguard" else intentanchor_scorer(gc, a.guard),
                             a.threshold)
    pipeline = make_pipeline(make_client(a.thinking == "on"), None if a.defense == "none" else a.defense, guard)
    suites, attacks, per_suite = {}, {}, []
    for name in a.suites.split(","):
        suites[name] = get_suite(BENCHMARK_VERSION, name)
        uts = list(suites[name].user_tasks)[: a.n_user]
        if a.attack == "none":
            attacks[name] = None
            per_suite.append([(name, u, None) for u in uts])
        else:
            attacks[name] = load_attack(a.attack, suites[name], pipeline)
            its = list(suites[name].injection_tasks)
            k = min(a.inj_per_user or len(its), len(its))
            per_suite.append([(name, u, its[(n * k + j) % len(its)]) for n, u in enumerate(uts) for j in range(k)])
    # 各 suite 轮流取，--limit 截断时也能覆盖到每个 suite
    cases = [g[i] for i in range(max(map(len, per_suite))) for g in per_suite if i < len(g)][: a.limit]
    todo = [c for c in cases if done.get(c, {}).get("status") != "ok"]
    print(f"{len(cases)} cases, {len(cases) - len(todo)} already done, running {len(todo)} with {a.workers} workers",
          flush=True)

    lock = threading.Lock()
    with ThreadPoolExecutor(a.workers) as pool, results_path.open("a") as fr, (out / "traces.jsonl").open("a") as ft:
        futures = [pool.submit(run_one, suites[s], attacks[s], pipeline, u, i) for s, u, i in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            key = {k: rec[k] for k in ("suite", "user_task", "injection_task")}
            with lock:
                fr.write(json.dumps({k: v for k, v in rec.items() if k not in TRACE_KEYS}, ensure_ascii=False) + "\n")
                ft.write(json.dumps({**key, "status": rec["status"], "utility": rec.get("utility"),
                                     "security": rec.get("security"), **{k: rec.get(k) for k in TRACE_KEYS}},
                                    ensure_ascii=False) + "\n")
                fr.flush()
                ft.flush()
            done[tuple(key.values())] = rec
            if rec["status"] != "ok":
                print(f"  [{n}/{len(todo)}] {'/'.join(map(str, key.values()))}: {rec['status']} "
                      f"{rec.get('error', '')[:150]}", flush=True)
            elif n % 20 == 0:
                print(f"  [{n}/{len(todo)}] ...", flush=True)

    summary = summarize([done[c] for c in cases if c in done])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["ALL"], ensure_ascii=False))


if __name__ == "__main__":
    main()
