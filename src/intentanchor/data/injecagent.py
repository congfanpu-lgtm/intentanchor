"""InjecAgent 公开数据（MIT，github.com/uiuc-kang-lab/InjecAgent）转成单步守卫样本（设计文档 §5.3.1）。

    set -a && . ./.env && set +a
    .venv-serve/bin/python -I -m intentanchor.data.injecagent data/external/InjecAgent --out data/injecagent_v0.jsonl

17 个用户用例 × 62 个攻击者用例（直接伤害 30、窃取数据 32）= 1,054 条，全部 BLOCK（misaligned_injection）：
user_task 是用户指令，history 是用户那次工具调用（窃取数据时再加上攻击者要求的读取调用），
proposed_call 是攻击者要求的副作用调用。
InjecAgent 只给了攻击者指令的文字，没有调用参数：每个攻击者用例用大模型按 tools.json 的参数定义把指令转成调用，
结果缓存在 <out 去掉后缀>.args.json，重跑不再花钱。base 和 enhanced 两种设置只差注入文本，只用 base。

v0.5（§4.1）起历史带工具返回：用户那次调用的返回是 InjecAgent 的 Tool Response Template，<Attacker Instruction> 换成攻击指令，
思考用它给的 Thought；窃取数据时攻击者要求的读取调用，返回取 attacker_simulated_responses.json 里该工具的第一条。
风险来源都是 indirect_prompt_injection。
"""
import argparse
import ast
import json
import os
import re
from collections import Counter
from itertools import groupby
from pathlib import Path

FILL_MODEL = "qwen3.8-max"
FILL_SYSTEM = """You convert records of the public InjecAgent benchmark into structured tool calls for a dataset.
You get an instruction and the tools it names, with their parameter schemas. Return the calls an agent would make to carry out the instruction, in the given order, as a JSON list: [{"tool": "<name>", "args": {...}}].
Fill every required parameter with a concrete value taken from the instruction. When a value would come from an earlier call's output (for example the data to be emailed), write a short realistic stand-in value. Use only parameters defined in the schema. Output the JSON only."""


def load_tools(root: Path) -> dict[str, dict]:
    """InjecAgent 的工具名是工具包名 + 工具名，如 GmailSendEmail。"""
    toolkits = json.loads((root / "data/tools.json").read_text())
    return {tk["toolkit"] + t["name"]: {**t, "toolkit": tk["toolkit"]} for tk in toolkits for t in tk["tools"]}


def _fill_messages(instruction: str, names: list[str], tools: dict) -> list[dict]:
    spec = "\n".join(f"- {n}: {tools[n]['summary']}\n  parameters: {json.dumps(tools[n]['parameters'])}" for n in names)
    return [{"role": "system", "content": FILL_SYSTEM},
            {"role": "user", "content": f"Instruction: {instruction}\n\nTools, in order:\n{spec}"}]


def parse_calls(text: str, names: list[str], tools: dict) -> list[dict]:
    """解析并校验大模型给的调用：工具和顺序一致（同一工具可以连续调用多次，如删除 3 份文档）、
    必填参数都有、没有多余参数。不合格抛 ValueError。"""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    calls = json.loads(text)
    if [k for k, _ in groupby(c.get("tool") for c in calls)] != names:
        raise ValueError(f"tools {[c.get('tool') for c in calls]} != {names}")
    for c in calls:
        params = {p["name"]: p for p in tools[c["tool"]]["parameters"]}
        missing = [n for n, p in params.items() if p.get("required") and n not in c["args"]]
        extra = [n for n in c["args"] if n not in params]
        if missing or extra:
            raise ValueError(f"{c['tool']}: missing {missing}, extra {extra}")
    return [{"tool": c["tool"], "args": c["args"]} for c in calls]


def fill_args(cases: dict[str, dict], tools: dict, cache: Path) -> dict[str, list[dict]]:
    import openai

    from intentanchor.eval.score_llm import BASE_URL

    done = json.loads(cache.read_text()) if cache.exists() else {}
    if done.keys() >= cases.keys():  # 全在缓存里就不连 API，也不需要 key
        return done
    client = openai.OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL, max_retries=3, timeout=120)
    for key, case in cases.items():
        if key in done:
            continue
        names = case["Attacker Tools"]
        for attempt in range(3):
            r = client.chat.completions.create(model=FILL_MODEL, temperature=0, max_tokens=800,
                                               messages=_fill_messages(case["Attacker Instruction"], names, tools),
                                               extra_body={"enable_thinking": False})
            try:
                done[key] = parse_calls(r.choices[0].message.content, names, tools)
                break
            except (ValueError, json.JSONDecodeError, AttributeError, KeyError, TypeError) as e:
                print(f"  {key} attempt {attempt}: {e}", flush=True)
        cache.write_text(json.dumps(done, indent=1, ensure_ascii=False))  # 每个用例写一次，中断了也不丢
    return done


def simulated_output(sim: dict[str, str], tool: str) -> str | None:
    """attacker_simulated_responses 的键是 "(工具名,参数)"，参数写法五花八门；取这个工具的第一条（一般是空参数那条）。"""
    return next((v for k, v in sim.items() if k.startswith(f"({tool},")), None)


def convert(tools: dict, user_cases: list[dict], attacker_cases: dict[str, dict],
            calls_by_case: dict[str, list[dict]], sim: dict[str, str]) -> list[dict]:
    samples = []
    for ui, u in enumerate(user_cases):
        thought = u["Thought"].removeprefix("Thought:").strip() or None
        for key, case in attacker_cases.items():
            if key not in calls_by_case:
                continue
            user_call = {"tool": u["User Tool"], "args": ast.literal_eval(u["Tool Parameters"]), "thought": thought,
                         "output": u["Tool Response Template"].replace("<Attacker Instruction>",
                                                                       case["Attacker Instruction"])}
            # 待审查调用取攻击者最后那个工具的第一次调用；窃取数据时前面的读取调用放进历史
            calls = calls_by_case[key]
            i = next(i for i, c in enumerate(calls) if c["tool"] == case["Attacker Tools"][-1])
            reads = [{**c, "thought": None, "output": simulated_output(sim, c["tool"])} for c in calls[:i]]
            call = {**calls[i], "thought": None}
            samples.append({
                "id": f"injecagent-{key}-u{ui:02d}", "lang": "en", "source": "public_attack", "split": None,
                "suite": None, "domain": tools[u["User Tool"]]["toolkit"], "attack_family": "injecagent",
                "category": "misaligned_injection", "risk_source": "indirect_prompt_injection",
                "label": "BLOCK", "label_source": "injecagent",
                "audited": False, "reason": None,
                "input": {"user_task": u["User Instruction"],
                          "tools": [{"name": call["tool"], "description": tools[call["tool"]]["summary"],
                                     "side_effect": True}],
                          "history": [user_call, *reads], "proposed_call": call},
                "meta": {"dataset": "injecagent", "user_case": ui, "attacker_case": key,
                         "attack_type": case["Attack Type"]},
            })
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tools = load_tools(a.root)
    user_cases = [json.loads(line) for line in (a.root / "data/user_cases.jsonl").read_text().splitlines()]
    attacker_cases = {f"{kind}{i:02d}": json.loads(line)
                      for kind in ("dh", "ds")
                      for i, line in enumerate((a.root / f"data/attacker_cases_{kind}.jsonl").read_text().splitlines())}
    unknown = {n for c in attacker_cases.values() for n in c["Attacker Tools"]} - tools.keys()
    unknown |= {u["User Tool"] for u in user_cases} - tools.keys()
    assert not unknown, f"tools missing from tools.json: {unknown}"

    calls = fill_args(attacker_cases, tools, a.out.with_suffix(".args.json"))
    sim = json.loads((a.root / "data/attacker_simulated_responses.json").read_text())
    samples = convert(tools, user_cases, attacker_cases, calls, sim)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    print(f"{len(samples)} samples -> {a.out}; attacker cases filled {len(calls)}/{len(attacker_cases)}")
    print("by attack type:", dict(Counter(s["meta"]["attack_type"] for s in samples)))
    print("proposed tools:", len({s["input"]["proposed_call"]["tool"] for s in samples}))


if __name__ == "__main__":
    main()
