"""StepGuard 用的静态测试集转成单步样本（设计文档 §5.5.1）。

    .venv-serve/bin/python -I -m intentanchor.data.benchmarks data/external/benchmarks \
        data/external/benchmarks/assebench/ASSEBench/dataset --out data/bench_v1

每个测试集一个 <out>/<名字>.jsonl，每条是一个待检查动作，meta.case 是原记录的 id，标签是原记录的标签（1 = 不安全 → BLOCK）：
- TS-Bench（单步）：current_action 是待检查动作；score 0.5 的丢掉，和 StepGuard 一致。agentdojo 子集按 suite 记进 meta.suite，
  其中 banking、slack 是本项目留出的测试环境，workspace、travel 是训练环境；
- ATBench、R-Judge、ASSEBench（整条轨迹）：每个 agent 动作各出一条，之前的事件作历史，meta.last 标出最后一个动作。
  打分后用 rescore --by-case 按记录汇总：max = 任意一步的最大风险分（守卫逐步拦截的用法），last = 只看最后一步（和 StepGuard 适配器取的动作相同）。
  ASSEBench 的 ambiguous 记录丢掉，和 StepGuard 一致。AgentSafety 找不到公开版本，没有转。
"""
import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

from intentanchor.data.stepguard import _parse_action

_decoder = json.JSONDecoder()
TS_STEP = re.compile(r"(?m)^\(1\) Thought:")
NAMED_CALL = re.compile(r"^(\w+)\s*(\{.*\})\s*$", re.S)                    # GmailReadEmail{'email_id': '1'}
REACT_CALL = re.compile(r"^(\w+)\s*\n\s*Action Input:\s*(.*)$", re.S)       # Name\nAction Input: {...}


def _args(text: str):
    text = text.strip()
    try:
        obj, _ = _decoder.raw_decode(text)
    except ValueError:
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return {"input": text}
    return obj if isinstance(obj, dict) else {"input": obj}


def parse_agent(action: str | None, thought: str | None) -> dict:
    """轨迹测试集里 agent 的一步 → 工具调用或回复。action 为空时内容在 thought 里（ASSEBench 常见）。"""
    action, thought = (action or "").strip(), (thought or "").strip() or None
    if not action:
        return {"response": thought or "", "thought": None}
    if action.startswith("Final Answer:"):
        return {"response": action.removeprefix("Final Answer:").strip(), "thought": thought}
    if not action.startswith(("{", "Complete")):
        if m := REACT_CALL.match(action) or NAMED_CALL.match(action):
            return {"tool": m.group(1), "args": _args(m.group(2)), "thought": thought}
    try:
        step = _parse_action(action)
    except ValueError:
        step = {"response": action}
    if "tool" in step and isinstance(step["args"], str):
        step["args"] = _args(step["args"])
    return {**step, "thought": thought}


def trajectory_steps(contents: list[list[dict]]) -> tuple[str, list[dict]]:
    """返回 (用户请求, 步骤)：第一条用户消息是请求，之后的用户消息、agent 动作按顺序，环境反馈挂到前一个 agent 动作上。"""
    turns = [t for group in contents for t in group]
    if not turns or turns[0]["role"] != "user":
        raise ValueError("trajectory does not start with a user turn")
    steps = []
    for t in turns[1:]:
        if t["role"] == "user":
            steps.append({"user": t["content"] or ""})
        elif t["role"] == "agent":
            steps.append({**parse_agent(t.get("action"), t.get("thought")), "output": None})
        elif t.get("content"):
            last = next((s for s in reversed(steps) if "user" not in s), None)
            if last is not None:
                last["output"] = t["content"] if last["output"] is None else last["output"] + "\n" + t["content"]
    return turns[0]["content"], steps


def ts_steps(text: str) -> list[dict]:
    """TS-Bench 的 "(1) Thought / (2) Action / (3) Action Input / Observation" 文本 → 步骤。"""
    steps = []
    for chunk in TS_STEP.split(text)[1:]:
        thought, sep, rest = chunk.partition("(2) Action:")
        if not sep:
            steps.append({"response": thought.strip(), "thought": None, "output": None})
            continue
        name, _, rest = rest.partition("(3) Action Input:")
        inp, sep, obs = rest.partition("Observation:")
        steps.append({"tool": name.strip(), "args": _args(inp), "thought": thought.strip() or None,
                      "output": obs.strip() if sep else None})
    return steps


def _tool_doc(env_info: str, name: str) -> str:
    m = re.search(rf"(?m)^{re.escape(name)}: (.*)$", env_info or "")
    return m.group(1).strip() if m else ""


def _sample(bench: str, case: str, k: int, n: int, user_task: str, history: list, action: dict, label: int,
            docs: dict, meta: dict) -> dict:
    proposed = {key: v for key, v in action.items() if key != "output"}
    tools = [{"name": proposed["tool"], "description": docs.get(proposed["tool"], ""), "side_effect": None}] \
        if "tool" in proposed else []
    return {"id": f"{bench}-{case}-a{k}", "lang": "en", "source": f"bench:{bench}", "split": "test", "suite": None,
            "domain": None, "attack_family": None, "category": "unsafe" if label else "safe",
            "risk_source": meta.pop("risk_source", None), "label": "BLOCK" if label else "ALLOW",
            "label_source": bench, "audited": False, "reason": None,
            "input": {"user_task": user_task, "tools": tools, "history": history, "proposed_call": proposed},
            "meta": {"bench": bench, "case": f"{bench}-{case}", "step": k, "n_steps": n, "last": k == n, **meta}}


def convert_ts(root: Path) -> tuple[list[dict], Counter]:
    out, stats = [], Counter()
    files = sorted(root.glob("ts_bench/agentdojo-traj/*.json")) + sorted(root.glob("ts_bench/agentharm-traj/*.json")) \
        + sorted(root.glob("ts_bench/asb-traj/test/*.json"))
    for f in files:
        subset = f.parent.name if f.parent.name != "test" else "asb-traj"
        for i, r in enumerate(json.loads(f.read_text())):
            if r["score"] == 0.5:
                stats["skip_score_0.5"] += 1
                continue
            action = ts_steps(r["current_action"])
            if len(action) != 1:
                stats["skip_parse"] += 1
                continue
            name = action[0].get("tool")
            meta = {"subset": subset, "file": f.stem}
            if subset == "agentdojo-traj":
                meta["suite"] = f.stem
            out.append(_sample("tsbench", f"{subset}-{f.stem}-{i}", 1, 1, r["instruction"], ts_steps(r["history"]),
                               action[0], int(r["score"] == 1), {name: _tool_doc(r.get("env_info"), name)} if name else {}, meta))
    return out, stats


def convert_trajectories(bench: str, records: list[dict]) -> tuple[list[dict], Counter]:
    out, stats = [], Counter()
    for r in records:
        if r.get("ambiguous"):
            stats["skip_ambiguous"] += 1
            continue
        try:
            user_task, steps = trajectory_steps(r["contents"])
        except (ValueError, KeyError, TypeError):
            stats["skip_parse"] += 1
            continue
        actions = [i for i, s in enumerate(steps) if "user" not in s]
        if not actions:
            stats["skip_no_agent_action"] += 1
            continue
        docs = {t["name"]: t.get("description", "") for t in r.get("tool_used") or [] if isinstance(t, dict)}
        meta = {k: r[k] for k in ("risk_source", "attack_type", "risk_type", "scenario") if r.get(k) is not None}
        for k, i in enumerate(actions, 1):
            out.append(_sample(bench, str(r["id"]), k, len(actions), user_task, steps[:i], steps[i], int(r["label"]),
                               docs, dict(meta)))
    return out, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="data/external/benchmarks")
    p.add_argument("asse", type=Path, help="ASSEBench 的 dataset 目录")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sets = {"tsbench": convert_ts(a.root)}
    for bench, path in [("atbench", a.root / "atbench_pro/ATbench-Pro.json"), ("rjudge", a.root / "rjudge/test.json"),
                        ("asse-safety", a.asse / "AgentJudge-safety.json"),
                        ("asse-security", a.asse / "AgentJudge-security.json")]:
        sets[bench] = convert_trajectories(bench, json.loads(path.read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    for bench, (samples, stats) in sets.items():
        (a.out / f"{bench}.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
        cases = {s["meta"]["case"]: s["label"] for s in samples}
        print(f"{bench}: {len(samples)} steps, {len(cases)} cases {dict(Counter(cases.values()))}, "
              f"steps by action {dict(Counter('tool' in s['input']['proposed_call'] for s in samples))}, {dict(stats)}")


if __name__ == "__main__":
    main()
