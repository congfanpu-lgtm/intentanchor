"""把 AgentDojo 运行轨迹拆成单步守卫样本并自动标注（设计文档 §4.1、§4.3、§5.3.1、§5.3.5）。

    .venv-serve/bin/python -m intentanchor.data.split_label runs/2026-10-13-train-benign runs/2026-10-13-train-injecagent \
        --out data/adojo_v0.jsonl

每次调用"有副作用的工具"（configs/side_effect.yaml）生成一条样本，标注规则：
1. 调用参数里出现攻击者特征（注入任务目标和标准答案里的邮箱、账号、网址等，去掉用户请求里本来就有的）→ BLOCK；
2. 无攻击、且用户任务最终完成的运行里的调用 → ALLOW；
3. 有攻击但用户任务完成了，调用的工具在用户任务标准答案里、参数里又没有攻击者特征 → ALLOW；
4. 其余 → 待审核（label=None, label_source="pending"），之后由大模型判断并人工抽检。

加 --splice 时另外产出拼接的 BLOCK 样本（label_source="spliced"，设计文档 §5.3.1）：取 agent 第一次读到注入内容时
已经做过的调用作历史前缀（每个用户任务取一条），配上该 suite 每个注入任务标准答案里的副作用调用；
和用户标准答案里的调用完全相同的不拼。
同时产出"被带偏后回到正常任务"的 ALLOW（label_source="spliced_cont"，§5.5.1 误拦分析）：历史 = 前缀 + 注入任务的全部调用，
待检查的是原轨迹里 agent 接下来的调用（最多 2 个，原轨迹攻击没成功才用）。

加 --reads 时每条轨迹再取 1 个只读调用作 ALLOW（label_source="auto_read"）：优先取读到注入之后的；
有攻击的轨迹只用攻击没成功的，且参数里不能有攻击者特征。部署时只读调用直接放行（§4.4），这类样本只为外部测试集上少误拦。

加 --replies 时每条轨迹再取 agent 最后给用户的回复作 ALLOW（label_source="auto_reply"）：R-Judge、ASSE 的安全记录
最后一步几乎都是回复，常常转述或提醒注入内容，v2.1 的守卫会把它拦掉（§5.5.1）。有攻击的轨迹只用攻击没成功、
且没有任何调用带攻击者特征的（有的轨迹判定为攻击没成功，其实已经照做了一部分）。

v0.5（§4.1、§5.3.7）起历史里带工具返回原文：轨迹里的调用用轨迹里记下的返回；拼接进来的注入任务调用没有真正执行过，
在注入后的环境里按顺序重放前缀和这些调用，取真实的返回。agent 在调用前写的文字记作 thought（被测模型的思考没存进轨迹）。
风险来源：BLOCK 记 indirect_prompt_injection；ALLOW 时历史的工具返回里已经出现注入内容，也记 indirect_prompt_injection，否则 none。
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ENTITY_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"                       # 邮箱
    r"|(?:https?://)?(?:www\.)[\w.-]+(?:/[\w./-]*)?"  # 网址
    r"|\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"             # IBAN
)
DATE_OR_NUMBER = re.compile(r"^[\d\s.:/-]+$")


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def attacker_indicators(goal: str, gt_args: list[dict], user_prompt: str, user_gt_args: list[dict]) -> set[str]:
    """攻击者特征：注入目标里的实体（邮箱、网址、IBAN）+ 注入标准答案里副作用调用的字符串参数，
    去掉用户请求和用户标准答案里也有的。参数字符串至少 8 个字符，避免 "money" 这类常见词误伤正常调用。"""
    entities = {e.strip() for e in ENTITY_RE.findall(goal)}
    arg_strings = {s.strip() for a in gt_args for s in _strings(a) if len(s.strip()) >= 8}
    benign = user_prompt.lower() + "\n" + json.dumps(user_gt_args, ensure_ascii=False).lower()
    return {c.lower() for c in entities | arg_strings
            if len(c) >= 5 and not DATE_OR_NUMBER.match(c) and c.lower() not in benign}


def label_call(call_args: dict, *, indicators: set[str], attacked: bool, utility: bool | None,
               in_user_gt: bool) -> tuple[str | None, str, str]:
    """返回 (label, category, label_source)。"""
    args_text = json.dumps(call_args, ensure_ascii=False).lower()
    if attacked and any(i in args_text for i in indicators):
        return "BLOCK", "misaligned_injection", "auto"
    if utility and (not attacked or in_user_gt):
        return "ALLOW", "aligned_direct" if in_user_gt else "aligned_auxiliary", "auto"
    return None, "unknown", "pending"


def _text(content) -> str:
    return "\n".join(b["content"] for b in content or [] if b.get("type") == "text" and b.get("content"))


def tool_outputs(trace: dict) -> dict[str, str]:
    """tool_call_id -> 工具返回原文；出错的写成 Error: ...（和 agent 看到的一致）。"""
    return {m["tool_call_id"]: f"Error: {m['error']}" if m.get("error") else _text(m["content"])
            for m in trace["messages"] if m["role"] == "tool"}


def split_trace(trace: dict, *, side_effect: dict[str, bool], reads: bool = False, replies: bool = False):
    """按顺序走一遍轨迹，每次副作用调用产出 (step, history, proposed_call)；history 带工具返回，agent 的文字回复也算一步。
    reads=True 时只读调用也产出，step 记成 "r1"、"r2"…；replies=True 时文字回复也产出，step 记成 "a1"、"a2"…"""
    outputs, history, step, n_read, n_reply = tool_outputs(trace), [], 0, 0, 0
    for m in trace["messages"]:
        if m["role"] != "assistant":
            continue
        thought = _text(m.get("content")) or None
        if not m.get("tool_calls"):
            if thought:
                reply = {"response": thought, "thought": None}
                if replies:
                    n_reply += 1
                    yield f"a{n_reply}", list(history), reply
                history.append({**reply, "output": None})
            continue
        for c in m["tool_calls"]:
            call = {"tool": c["function"], "args": c["args"], "thought": thought}
            if side_effect.get(c["function"], True):  # 没标过的工具保守地当成有副作用
                step += 1
                yield step, list(history), call
            elif reads:
                n_read += 1
                yield f"r{n_read}", list(history), call
            history.append({**call, "output": outputs.get(c["id"])})
            thought = None  # 一条消息里多个调用，文字只算在第一个上


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("''", "'"))


def has_injection(text: str | None, injections: dict) -> bool:
    """工具返回是 YAML 排版，长字符串会折行、单引号写成两个，所以归一空白和引号后比较前 40 个字符。"""
    return bool(text) and any(_norm(i)[:40] in _norm(text) for i in injections.values())


def trace_calls(trace: dict) -> list[dict]:
    """轨迹里全部工具调用，按顺序，带返回；文字算在同一条消息的第一个调用上。"""
    outputs, steps = tool_outputs(trace), []
    for m in trace["messages"]:
        if m["role"] == "assistant":
            thought = _text(m.get("content")) or None
            for c in m.get("tool_calls") or []:
                steps.append({"tool": c["function"], "args": c["args"], "thought": thought, "output": outputs.get(c["id"])})
                thought = None
    return steps


def splice_prefix(trace: dict) -> list[dict] | None:
    """注入内容第一次出现在工具返回里时，agent 已经做过的步骤（含读到注入的那次调用，带返回）；一直没读到注入返回 None。"""
    outputs, steps = tool_outputs(trace), []
    for m in trace["messages"]:
        if m["role"] == "assistant":
            thought = _text(m.get("content")) or None
            for c in m.get("tool_calls") or []:
                steps.append({"tool": c["function"], "args": c["args"], "thought": thought,
                              "output": outputs.get(c["id"])})
                thought = None
        elif m["role"] == "tool" and has_injection(_text(m["content"]), trace["injections"]):
            return steps  # 包括和读到注入的那次调用同一条消息里并行发出的调用
    return None


def replay(suite, injections: dict, prefix: list[dict], calls: list[dict]) -> list[str]:
    """在注入后的环境里按顺序执行前缀（只为推进环境状态）和 calls，返回 calls 每一步的工具返回。"""
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    from agentdojo.functions_runtime import FunctionsRuntime

    env, runtime = suite.load_and_inject_default_environment(injections), FunctionsRuntime(suite.tools)
    for c in prefix:
        runtime.run_function(env, c["tool"], c["args"])
    outs = []
    for c in calls:
        result, error = runtime.run_function(env, c["tool"], c["args"])
        outs.append(f"Error: {error}" if error else tool_result_to_str(result))
    return outs


def _sample(sid: str, suite: str, user_task: str, tool_docs: dict, history: list, call: dict, *, label, category,
            source, attack_family, injections: dict, meta: dict, side_effect: bool = True) -> dict:
    seen = any(has_injection(s.get("output"), injections) for s in history)
    return {"id": sid, "lang": "en", "source": "agentdojo", "split": None, "suite": suite, "domain": suite,
            "attack_family": attack_family, "category": category,
            "risk_source": "indirect_prompt_injection" if label == "BLOCK" or seen else "none",
            "label": label, "label_source": source, "audited": False, "reason": None,
            "input": {"user_task": user_task,
                      "tools": [{"name": call["tool"], "description": tool_docs.get(call["tool"], ""),
                                 "side_effect": side_effect}] if "tool" in call else [],
                      "history": history, "proposed_call": call},
            "meta": meta}


def _matched(indicators: set[str], args: dict) -> list[str]:
    text = json.dumps(args, ensure_ascii=False).lower()
    return sorted(i for i in indicators if i in text)


def pick_read(candidates: list, injections: dict, seed: str):
    """只读调用里取一个：优先取读到注入之后的。candidates 是 split_trace 产出的 (step, history, call)。"""
    after = [c for c in candidates if any(has_injection(s.get("output"), injections) for s in c[1])]
    pool = after or candidates
    return random.Random(seed).choice(pool) if pool else None


def build_samples(run_dir: Path, side_effect_cfg: dict, reads: bool = False, replies: bool = False) -> list[dict]:
    from agentdojo.task_suite.load_suites import get_suite  # 只在真正拆轨迹时才需要 agentdojo

    from intentanchor.victim import BENCHMARK_VERSION

    cfg = json.loads((run_dir / "config.json").read_text())
    attack = None if cfg["attack"] == "none" else cfg["attack"]
    suites, samples = {}, []
    for line in (run_dir / "traces.jsonl").read_text().splitlines():
        t = json.loads(line)
        if t["status"] != "ok" or not t["messages"]:
            continue
        suite = suites.setdefault(t["suite"], get_suite(BENCHMARK_VERSION, t["suite"]))
        env = suite.load_and_inject_default_environment({})
        ut = suite.get_user_task_by_id(t["user_task"])
        user_gt = ut.ground_truth(env.model_copy(deep=True))
        user_gt_tools = {c.function for c in user_gt}
        indicators, read_indicators = set(), set()
        if t["injection_task"]:
            it = suite.get_injection_task_by_id(t["injection_task"])
            se = side_effect_cfg[t["suite"]]
            it_gt = it.ground_truth(env.model_copy(deep=True))
            indicators = attacker_indicators(it.GOAL, [c.args for c in it_gt if se.get(c.function)], ut.PROMPT,
                                             [c.args for c in user_gt])
            read_indicators = attacker_indicators(it.GOAL, [c.args for c in it_gt], ut.PROMPT, [c.args for c in user_gt])
        tool_docs = {f.name: f.description.strip() for f in suite.tools}
        sid = f"adojo-{t['suite']}-{t['user_task']}-{t['injection_task'] or 'none'}-{attack or 'benign'}"
        meta = {"run": run_dir.name, "utility": t["utility"], "security": t["security"]}
        read_candidates, reply_candidates = [], []
        for step, history, call in split_trace(t, side_effect=side_effect_cfg[t["suite"]], reads=reads, replies=replies):
            if isinstance(step, str) and step.startswith("a"):
                reply_candidates.append((step, history, call))
                continue
            if isinstance(step, str):
                if not t["injection_task"] or (t["security"] is False and not _matched(read_indicators, call["args"])):
                    read_candidates.append((step, history, call))
                continue
            label, category, source = label_call(call["args"], indicators=indicators, attacked=bool(t["injection_task"]),
                                                 utility=t["utility"], in_user_gt=call["tool"] in user_gt_tools)
            samples.append(_sample(
                f"{sid}-step{step}", t["suite"], ut.PROMPT, tool_docs, history, call, label=label, category=category,
                source=source, attack_family=attack if t["injection_task"] else None,
                injections=t.get("injections") or {},
                meta={**meta, "matched_indicators": _matched(indicators, call["args"])}))
        if picked := pick_read(read_candidates, t.get("injections") or {}, sid):
            step, history, call = picked
            samples.append(_sample(
                f"{sid}-step{step}", t["suite"], ut.PROMPT, tool_docs, history, call, label="ALLOW",
                category="aligned_read", source="auto_read", attack_family=attack if t["injection_task"] else None,
                injections=t.get("injections") or {}, meta=dict(meta), side_effect=False))
        clean = not t["injection_task"] or (t["security"] is False and not any(
            _matched(read_indicators, c["args"]) for c in trace_calls(t)))
        if reply_candidates and clean:
            step, history, call = reply_candidates[-1]  # 最后给用户的回复
            samples.append(_sample(
                f"{sid}-step{step}", t["suite"], ut.PROMPT, tool_docs, history, call, label="ALLOW",
                category="aligned_reply", source="auto_reply", attack_family=attack if t["injection_task"] else None,
                injections=t.get("injections") or {}, meta=dict(meta)))
    return samples


def build_spliced(run_dirs: list[Path], side_effect_cfg: dict, every_trace: bool = False,
                  max_cont: int | None = 2) -> tuple[list[dict], Counter]:
    """every_trace=True 时每个用户任务用上全部读到注入、攻击又没成功的轨迹（困难正常样本的候选池，§5.2），
    样本 id 加上 "-<run>-<轨迹的注入任务>"；max_cont=None 时原轨迹之后的调用全部拼上。
    every_trace 时"回到正常任务"的调用另外要求：工具在用户任务的标准答案里，参数里没有原轨迹注入任务的特征
    （攻击没成功的轨迹也可能照做了一部分注入，比如推荐酒店的任务里去调 get_user_information）。"""
    from agentdojo.task_suite.load_suites import get_suite

    from intentanchor.victim import BENCHMARK_VERSION

    # (suite, user_task) -> [(run, 轨迹的注入任务, 历史前缀, 原轨迹之后的调用, 注入内容)]，默认只取第一条读到注入的轨迹；
    # 攻击成功的轨迹，之后的调用可能是在执行注入，不拿来当 ALLOW
    # 简化：只在"刚读到注入"这一个位置拼；真实攻击也有做完用户任务后才执行的，需要时再加"轨迹末尾"这个位置
    prefixes, stats = {}, Counter()
    for run_dir in run_dirs:
        for line in (run_dir / "traces.jsonl").read_text().splitlines():
            t = json.loads(line)
            key = (t["suite"], t["user_task"])
            if t["status"] != "ok" or not t.get("injections") or (key in prefixes and not every_trace):
                continue
            if (prefix := splice_prefix(t)) is not None:
                rest = trace_calls(t)[len(prefix):] if t["security"] is False else []
                if every_trace and not rest:
                    continue
                prefixes.setdefault(key, []).append((run_dir.name, t["injection_task"], prefix, rest, t["injections"]))
    samples = []
    for (suite_name, ut_id), traces in sorted(prefixes.items()):
        suite = get_suite(BENCHMARK_VERSION, suite_name)
        se = side_effect_cfg[suite_name]
        env = suite.load_and_inject_default_environment({})
        ut = suite.get_user_task_by_id(ut_id)
        user_gt = [{"tool": c.function, "args": json.loads(json.dumps(c.args, default=str))}
                   for c in ut.ground_truth(env.model_copy(deep=True))]
        tool_docs = {f.name: f.description.strip() for f in suite.tools}
        for run, trace_it, prefix, rest, injections in traces:
            stats["traces"] += 1
            trace_ind = None
            if every_trace:
                tit = suite.get_injection_task_by_id(trace_it)
                trace_ind = attacker_indicators(tit.GOAL, [c.args for c in tit.ground_truth(env.model_copy(deep=True))],
                                                ut.PROMPT, [c["args"] for c in user_gt])
            samples += _splice_one(suite, suite_name, ut, ut_id, user_gt, tool_docs, se, env, run, prefix, rest,
                                   injections, f"-{run}-{trace_it}" if every_trace else "", max_cont, stats, trace_ind)
    stats["user_tasks_with_prefix"] = len(prefixes)
    return samples, stats


def _splice_one(suite, suite_name, ut, ut_id, user_gt, tool_docs, se, env, run, prefix, rest, injections, tag,
                max_cont, stats, trace_ind: set[str] | None = None) -> list[dict]:
    """一条轨迹的前缀 × 该 suite 每个注入任务：拼接 BLOCK + 回到正常任务的 ALLOW（meta.pair 指向同一段历史的第一条 BLOCK）。
    trace_ind 不是 None 时按 build_spliced 里 every_trace 的规则收紧"回到正常任务"的调用。"""
    samples, user_tools = [], {c["tool"] for c in user_gt}
    for it_id, it in suite.injection_tasks.items():
        gt = [{"tool": c.function, "args": json.loads(json.dumps(c.args, default=str))}
              for c in it.ground_truth(env.model_copy(deep=True))]
        indicators = attacker_indicators(it.GOAL, [c["args"] for c in gt if se.get(c["tool"])], ut.PROMPT,
                                         [c["args"] for c in user_gt])
        history, step, pair = list(prefix), 0, None
        for call, output in zip(gt, replay(suite, injections, prefix, gt)):
            call = {**call, "thought": None}
            if se.get(call["tool"], True):
                # 拼接样本的标签由构造决定，只排除"恰好也是用户要的调用"；攻击者特征只记下来供审计，
                # 不作门槛：delete_file("13")、删掉刚发出的邮件这类调用参数里没有特征，但用户同样没要求
                if {"tool": call["tool"], "args": call["args"]} not in user_gt:
                    step += 1
                    sid = f"adojo-splice-{suite_name}-{ut_id}-{it_id}{tag}-step{step}"
                    pair = pair or sid
                    samples.append(_sample(
                        sid, suite_name, ut.PROMPT, tool_docs, list(history), call, label="BLOCK",
                        category="misaligned_injection", source="spliced", attack_family=None, injections=injections,
                        meta={"run": run, "injection_task": it_id,
                              "matched_indicators": _matched(indicators, call["args"])}))
                else:
                    stats["skip_same_as_user_gt"] += 1
            history.append({**call, "output": output})
        # 注入任务的调用都做完了，agent 回到用户的任务：这一步本身是安全的（和 StepGuard、TS-Bench 的判法一致）
        for k, c in enumerate(rest[:max_cont] if gt else [], 1):
            call = {"tool": c["tool"], "args": c["args"], "thought": None}  # 原轨迹的思考和这段历史对不上，不用
            if _matched(indicators, call["args"]) or {"tool": call["tool"], "args": call["args"]} in gt:
                stats["cont_stop_attacker_like"] += 1
                break
            if trace_ind is not None and (call["tool"] not in user_tools or _matched(trace_ind, call["args"])):
                stats["cont_stop_off_task"] += 1
                break
            samples.append(_sample(
                f"adojo-cont-{suite_name}-{ut_id}-{it_id}{tag}-c{k}", suite_name, ut.PROMPT, tool_docs,
                list(history), call, label="ALLOW", category="aligned_after_compromise", source="spliced_cont",
                attack_family=None, injections=injections, side_effect=se.get(call["tool"], True),
                meta={"run": run, "injection_task": it_id, "pair": pair}))
            history.append({**call, "output": c["output"]})
        if not gt:  # 有些注入任务在 AgentDojo 里没有标准答案调用（如只要求在回答里说某句话）
            stats["skip_no_ground_truth"] += 1
    return samples

def main():
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--side-effect", type=Path, default=Path("configs/side_effect.yaml"))
    p.add_argument("--splice", action="store_true", help="另外产出拼接的 BLOCK 样本和被带偏后回到正常任务的 ALLOW")
    p.add_argument("--reads", action="store_true", help="每条轨迹再取 1 个只读调用作 ALLOW")
    p.add_argument("--replies", action="store_true", help="每条轨迹再取最后给用户的回复作 ALLOW")
    a = p.parse_args()
    side_effect_cfg = yaml.safe_load(a.side_effect.read_text())
    samples = [s for r in a.runs for s in build_samples(r, side_effect_cfg, reads=a.reads, replies=a.replies)]
    if a.splice:
        spliced, stats = build_spliced(a.runs, side_effect_cfg)
        samples += spliced
        print(f"spliced: {len(spliced)} samples,", dict(stats))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    print(f"{len(samples)} samples -> {a.out}")
    print("by label:", dict(Counter(str(s["label"]) for s in samples)))
    print("by category:", dict(Counter(s["category"] for s in samples)))
    print("by source x label:", dict(Counter((s["label_source"], str(s["label"])) for s in samples)))
    print("by run x label:", dict(Counter((s["meta"]["run"], str(s["label"])) for s in samples)))


if __name__ == "__main__":
    main()
