"""StepGuard 开源数据（Apache-2.0）转换成本项目的样本格式（设计文档 §5.3.7，v0.5）。

    .venv-serve/bin/python -I -m intentanchor.data.stepguard data/external/stepguard_data --out data/stepguard_v1.jsonl

默认转 sft3k（进训练集）；`--subset rl4k` 转 rl4k 作 DPO / 困难样本挖掘的样本池，它没有分析和不安全步号，
不安全的整条轨迹丢掉，和 sft3k 重复的输入也丢掉。9 类风险来源全收，标签用原判定（safe → ALLOW，unsafe → BLOCK），
理由用它的 <Analysis>。保留工具返回原文和 agent 的思考（§4.1 变体 B）。
- 单步样本（<<Action Under Review>>）：待检查的动作可以是工具调用，也可以是直接回复用户的文字；
- 整条轨迹样本：不安全的取 <UnsafeStep> 那一步作待检查动作、之前的作历史；安全的取最后一个动作。
  步号按 agent 动作（工具调用和文字回复）出现的顺序数，和 StepGuard 序列化时的编号方式一致。
  待检查动作或历史里有一步写的是文字回复、后面却跟着工具返回的丢掉（HiddenCall）：原轨迹实际执行了一个
  没写出来的调用，StepGuard 按那次执行标 unsafe，守卫却只看得到一句回复（多是"请确认参数"），或看到一句
  转述那次执行结果的回复，标签对不上（§5.3.7 v2 审计）。两种格式都查。
用到 AgentDojo 测试环境（banking、slack）工具名的样本丢掉，防止泄漏。
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

_decoder = json.JSONDecoder()
# 行首标记：可选的 [Step k] / [Observation after Step k] 前缀，若干个 [AGENT]:，再跟 USER / THOUGHT / ACTION / RESPONSE / ENVIRONMENT
MARK = re.compile(r"(?m)^(?:\[Step \d+\]\s*|\[Observation after Step \d+\]\s*)?(?P<agent>(?:\[AGENT\]:[ \t]*\n?[ \t]*)*)"
                  r"(?:\[(?P<kind>USER|THOUGHT|ACTION|RESPONSE|ENVIRONMENT)\]:)?")
COMPLETE = re.compile(r"""Complete\s*\{\s*['"]response['"]\s*:\s*['"](.*)['"]\s*\}\s*$""", re.S)


def _between(text: str, start: str, end: str | None) -> str:
    i = text.index(start) + len(start)
    return text[i: text.index(end, i)] if end else text[i:]


def segments(text: str) -> list[tuple[str, str]]:
    """切成 (类型, 内容)：类型是 USER / THOUGHT / ACTION / RESPONSE / ENVIRONMENT，或者 AGENT（只有 [AGENT]: 前缀，后面是回复文字）。
    只认行首标记；工具返回里出现在行中间的 "[ACTION]:" 不算。"""
    marks = [m for m in MARK.finditer(text) if m.group("agent") or m.group("kind")]
    out = []
    for m, nxt in zip(marks, marks[1:] + [None]):
        body = text[m.end(): nxt.start() if nxt else len(text)].strip()
        out.append((m.group("kind") or "AGENT", body))
    return out


def _parse_action(body: str) -> dict:
    """[ACTION]: 后面的内容：{"name", "arguments"} 是工具调用；Complete{"response": ...} 和其他文字是回复。"""
    if body.startswith("{"):
        obj, _ = _decoder.raw_decode(body)
        if isinstance(obj, str):  # 少数样本把调用写成了 JSON 字符串
            obj = json.loads(obj)
        if not isinstance(obj, dict) or "name" not in obj:
            raise ValueError(f"unexpected action payload: {str(obj)[:80]}")
        return {"tool": obj["name"], "args": obj.get("arguments") or {}}
    if body.startswith("Complete"):
        try:
            obj, _ = _decoder.raw_decode(body[len("Complete"):].lstrip())
            return {"response": obj.get("response", "") if isinstance(obj, dict) else str(obj)}
        except ValueError:  # 有的写成 Python 字典、单引号里还有没转义的撇号，只能按正则取
            if m := COMPLETE.match(body):
                return {"response": m.group(1)}
            raise
    return {"response": body}


def parse_events(text: str, user_task: str) -> list[dict]:
    """把一段历史 / 轨迹解析成 §4.1 的步骤列表（工具调用、回复、用户消息），工具返回挂到前一个调用上。"""
    steps, thought = [], None
    for kind, body in segments(text):
        if kind == "THOUGHT":
            thought = body
        elif kind == "USER":
            if not (not steps and body == user_task):  # 轨迹开头重复一遍的用户请求不算
                steps.append({"user": body})
        elif kind == "ENVIRONMENT":  # 挂到前一个 agent 动作上；有些轨迹在文字回复之后也有环境反馈
            last = next((s for s in reversed(steps) if "user" not in s), None)
            if last is None:
                raise ValueError("environment output before any agent action")
            last["output"] = body if last.get("output") is None else last["output"] + "\n" + body
        elif kind == "ACTION" or body:  # RESPONSE 或 AGENT 后面直接跟文字，就是回复
            steps.append({**(_parse_action(body) if kind == "ACTION" else {"response": body}), "thought": thought,
                          "output": None})
            thought = None
    return steps


def _tools(body: str) -> dict[str, str]:
    tools = {}
    for line in body.splitlines():
        m = re.match(r"\s*-\s*([^:]+):\s*(.*)", line)
        if m:
            tools[m.group(1).strip()] = m.group(2).strip()
    return tools


def _input(user_task: str, history: list[dict], action: dict, tools: dict[str, str]) -> dict:
    proposed = {k: v for k, v in action.items() if k != "output"}
    tool_docs = [{"name": proposed["tool"], "description": tools.get(proposed["tool"], ""), "side_effect": None}] \
        if "tool" in proposed else []
    return {"user_task": user_task, "tools": tool_docs, "history": history, "proposed_call": proposed}


class HiddenCall(ValueError):
    pass


def _check_hidden(steps: list[dict]) -> None:
    if any("response" in s and s.get("output") is not None for s in steps):
        raise HiddenCall("a reply is followed by tool output")


def parse_prompt(prompt: str, unsafe_step: int | None = None) -> dict:
    """解析一条 StepGuard 提示的输入部分，返回 §4.1 的输入；格式不对抛异常，由调用方计数跳过。"""
    body = prompt[prompt.index("### Input"):]
    tools = _tools(_between(body, "<<Available Tools>>", None))
    if "<<Action Under Review>>" in body:
        user_task = _between(body, "<<User Request>>", "<<Interaction History>>").strip()
        history = parse_events(_between(body, "<<Interaction History>>", "<<Action Under Review>>"), user_task)
        actions = parse_events(_between(body, "<<Action Under Review>>", "<<Available Tools>>"), user_task)
        if len(actions) != 1 or "user" in actions[0]:
            raise ValueError(f"expected 1 action under review, got {len(actions)}")
        _check_hidden(history + actions)
        return _input(user_task, history, actions[0], tools)
    user_task = _between(body, "<<User Request>>", "<BEGIN TRAJECTORY>").strip()
    steps = parse_events(_between(body, "<BEGIN TRAJECTORY>", "<END TRAJECTORY>"), user_task)
    action_idx = [i for i, s in enumerate(steps) if "user" not in s]
    if not action_idx:
        raise ValueError("trajectory without agent actions")
    if unsafe_step is None:
        k = action_idx[-1]
    elif 1 <= unsafe_step <= len(action_idx):
        k = action_idx[unsafe_step - 1]
    else:
        raise ValueError(f"unsafe step {unsafe_step} out of range ({len(action_idx)} actions)")
    _check_hidden(steps[:k + 1])
    return _input(user_task, steps[:k], steps[k], tools)


def _tag(name: str, text: str) -> str | None:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else None


def _tool_names(inp: dict) -> set[str]:
    return {s["tool"] for s in inp["history"] + [inp["proposed_call"]] if "tool" in s}


def _records(root: Path, subset: str):
    """逐条给出 (原 id, 提示, 判定, 风险来源, 分析, 不安全步号原文)。rl4k 只有标签，没有分析和不安全步号。"""
    if subset == "sft3k":
        for i, line in enumerate((root / "sft3k/agentguard_sft3k_sharegpt.jsonl").read_text().splitlines()):
            conv = json.loads(line)["conversations"]
            resp = conv[1]["value"]
            yield (f"sft-{i:05d}", conv[0]["value"], _tag("Judgment", resp), _tag("RiskSource", resp),
                   _tag("Analysis", resp), _tag("UnsafeStep", resp))
    else:
        for line in (root / "rl4k/agentguard_rl4k_grpo.jsonl").read_text().splitlines():
            d = json.loads(line)
            yield d["id"].removeprefix("stepguard-"), d["instruction"], d["label"], d["risk_source"], None, None


def _key(inp: dict) -> str:
    return json.dumps([inp["user_task"], inp["history"], inp["proposed_call"]], sort_keys=True, ensure_ascii=False)


def convert(root: Path, leak_tools: set[str], subset: str = "sft3k",
            exclude: frozenset[str] = frozenset()) -> tuple[list[dict], Counter]:
    """exclude：要跳过的输入（_key），转 rl4k 时传 sft3k 的，免得和训练数据重复。"""
    stats, samples, seen = Counter(), [], {}
    for orig_id, prompt, judgment, risk, analysis, raw in _records(root, subset):
        level = "step" if "<<Action Under Review>>" in prompt else "trajectory"
        unsafe_step = None
        if level == "trajectory" and judgment == "unsafe":
            if not raw or not raw.isdigit():
                stats["skip_unsafe_trajectory_without_step"] += 1
                continue
            unsafe_step = int(raw)
        try:
            inp = parse_prompt(prompt, unsafe_step)
        except (ValueError, KeyError, TypeError) as e:  # JSONDecodeError 是 ValueError 的子类
            stats[f"skip_parse:{level}:{type(e).__name__}"] += 1
            continue
        if _tool_names(inp) & leak_tools:
            stats["skip_leak_agentdojo_test_tool"] += 1
            continue
        label = "BLOCK" if judgment == "unsafe" else "ALLOW"
        key = _key(inp)
        if key in exclude:
            stats["skip_in_exclude"] += 1
            continue
        if key in seen:  # 重复的：标签一致就去重，不一致两条都不要
            prev = seen[key]
            stats["dedup"] += 1
            if prev is not None and prev["label"] != label:
                samples.remove(prev)
                seen[key] = None
                stats["drop_conflicting_labels"] += 1
            continue
        sample = {
            "id": f"stepguard-{orig_id}", "lang": "en", "source": "stepguard", "split": None, "suite": None,
            "domain": None, "attack_family": risk if label == "BLOCK" else None,
            "category": f"{'unsafe' if label == 'BLOCK' else 'safe'}:{risk}", "risk_source": risk,
            "label": label, "label_source": "stepguard", "audited": False, "reason": analysis,
            "input": inp,
            "meta": {"subset": subset, "orig_id": orig_id, "level": level, "unsafe_step": unsafe_step,
                     "action": "call" if "tool" in inp["proposed_call"] else "reply"},
        }
        seen[key] = sample
        samples.append(sample)
    return samples, stats


def main():
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--side-effect", type=Path, default=Path("configs/side_effect.yaml"))
    p.add_argument("--subset", choices=["sft3k", "rl4k"], default="sft3k")
    a = p.parse_args()
    cfg = yaml.safe_load(a.side_effect.read_text())
    leak_tools = set(cfg["banking"]) | set(cfg["slack"])  # AgentDojo 测试环境的工具名
    exclude = frozenset()
    if a.subset == "rl4k":  # 和 sft3k 重复的不要（sft3k 已进训练集）
        exclude = frozenset(_key(s["input"]) for s in convert(a.root, leak_tools)[0])
    samples, stats = convert(a.root, leak_tools, a.subset, exclude)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    print(f"{len(samples)} samples -> {a.out}")
    print("by label:", dict(Counter(s["label"] for s in samples)))
    print("by level x action x label:",
          dict(sorted(Counter((s["meta"]["level"], s["meta"]["action"], s["label"]) for s in samples).items())))
    print("by risk x label:", dict(sorted(Counter((s["risk_source"], s["label"]) for s in samples).items())))
    print("skipped / dedup:", dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
