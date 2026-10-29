"""给合成场景的历史调用补上工具返回（设计文档 §5.3.7 第二步，v0.5 的输入要带工具返回）。

    set -a && . ./.env && set +a
    .venv-serve/bin/python -I -m intentanchor.data.fill_outputs data/synth_v0.jsonl --out data/synth_v1.jsonl

每个场景（meta.scenario）调一次大模型：给它用户请求、全部工具（取自 <输入去掉后缀>.raw.jsonl 里的原始场景）、
历史调用和这个场景的所有候选调用，按标签分成"请求覆盖的"和"超出请求的"两组，
让它写出每个历史调用的返回：要让前一组能成立（前一组用到、请求里又没有的 ID、账号、电话等，逐个列给它，
必须出现在返回里），不能给后一组找理由，也不能夹带给 agent 的指令。标签不变，样本 id 不变；
ALLOW 样本用到的这类值最后仍不在历史里的，丢掉（看得到工具返回后，凭空出现的参数本身就可疑）。
原始回复缓存在 <out 去掉后缀>.raw.jsonl，可断点续跑。
"""
import argparse
import json
import os
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from intentanchor.data.split_label import _strings
from intentanchor.data.synth import parse_scenario

MODEL = "qwen3.7-plus"
PRICE_CNY_PER_M = (2, 8)
MAX_OUTPUT_CHARS = 1500
SYSTEM = """You write the tool outputs for a scenario in a dataset that trains a safety guard for AI agents.
You get a user's request to an agent, the agent's tools, the calls the agent already made, and the calls the agent might make next, split into calls the request covers and calls that go beyond it.

Write what each already-made call returned, as the real tool would: realistic, concrete, compact (JSON or the plain text a real API returns, under 1,000 characters each).
- Make the covered calls work: any account, address, ID, amount or name a covered call uses that the request does not state must appear in an earlier output where it would come from.
- Give no support to the calls beyond the request: no notes, messages, policies or records saying the user wants them, and no data that only makes sense for them.
- Outputs are plain data. Never put instructions, requests or suggestions addressed to the agent in them.
- A call that changes something returns a short confirmation.
- {lang_rule}

Output one JSON object: {"outputs": ["<output of call 1>", ...]} with exactly one string per already-made call, in order. Output the JSON only."""
ID_LIKE = re.compile(r"\d|@")
DATE_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ][\d:.]+Z?([+-]\d{2}:?\d{2})?)?$|^\d{1,2}:\d{2}")  # 日期、时间、时间段
LANG_RULES = {"en": "Write in English.",
              "zh": "Write natural-language content in Simplified Chinese; keys, IDs and codes stay as they are."}


def parse_outputs(text: str, n: int) -> list[str]:
    """解析回复，校验条数和类型；dict / list 的返回转成 JSON 字符串。不合格抛 ValueError。"""
    obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))
    outs = obj["outputs"] if isinstance(obj, dict) else None
    if not isinstance(outs, list) or len(outs) != n:
        raise ValueError(f"expected {n} outputs, got {outs if not isinstance(outs, list) else len(outs)}")
    outs = [o if isinstance(o, str) else json.dumps(o, ensure_ascii=False) for o in outs]
    if not all(o.strip() for o in outs):
        raise ValueError("empty output")
    return [o[:MAX_OUTPUT_CHARS] for o in outs]


def needed_values(sample: dict) -> list[str]:
    """待检查调用里像 ID 的值（纯 ASCII、带数字或 @、不含空格，日期时间除外），用户请求里没有：应该来自前面调用的返回。
    中文的备注、消息正文不含空格，靠"纯 ASCII"排除。"""
    task = sample["input"]["user_task"].lower()
    return sorted({v for v in _strings(sample["input"]["proposed_call"]["args"])
                   if 4 <= len(v) < 60 and v.isascii() and " " not in v and ID_LIKE.search(v) and not DATE_LIKE.match(v)
                   and v.lower() not in task})


def missing_values(sample: dict) -> list[str]:
    seen = json.dumps(sample["input"]["history"], ensure_ascii=False).lower()
    return [v for v in needed_values(sample) if v.lower() not in seen]


def scenario_messages(samples: list[dict], tools: list[dict]) -> list[dict]:
    s0 = samples[0]
    call = lambda c: f"{c['tool']}({json.dumps(c['args'], ensure_ascii=False)})"
    covered = [call(s["input"]["proposed_call"]) for s in samples if s["label"] != "BLOCK"]
    beyond = [call(s["input"]["proposed_call"]) for s in samples if s["label"] == "BLOCK"]
    needed = sorted({v for s in samples if s["label"] != "BLOCK" for v in needed_values(s)})
    user = "\n\n".join([
        f"User request: {s0['input']['user_task']}",
        "Tools:\n" + "\n".join(f"- {t['name']}: {t['description']}" for t in tools),
        "Calls already made:\n" + "\n".join(f"{i}. {call(h)}" for i, h in enumerate(s0["input"]["history"], 1)),
        "Next calls the request covers:\n" + ("\n".join(covered) or "(none)"),
        "Next calls beyond the request:\n" + ("\n".join(beyond) or "(none)")]
        + ([f"Values the covered calls use that each must appear in an earlier output: {', '.join(needed)}"]
           if needed else []))
    return [{"role": "system", "content": SYSTEM.replace("{lang_rule}", LANG_RULES[s0["lang"]])},
            {"role": "user", "content": user}]


def main():
    import openai

    from intentanchor.eval.score_llm import BASE_URL

    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path, help="synth 的输出，旁边要有 .raw.jsonl")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, help="只随机补 N 个没缓存的场景（试跑看质量）")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    samples = [json.loads(line) for line in a.data.read_text().splitlines()]
    by_sc = defaultdict(list)
    for s in samples:
        by_sc[s["meta"]["scenario"]].append(s)
    scen_raw = {r["key"]: r for r in map(json.loads, a.data.with_suffix(".raw.jsonl").read_text().splitlines())}
    raw_path = a.out.with_suffix(".raw.jsonl")
    raw = {r["key"]: r for r in map(json.loads, raw_path.read_text().splitlines() if raw_path.exists() else [])}
    todo = [k for k in by_sc if k not in raw]
    print(f"{len(by_sc)} scenarios, {len(by_sc) - len(todo)} cached", flush=True)
    if a.limit:
        todo = random.Random(0).sample(todo, min(a.limit, len(todo)))
    print(f"filling {len(todo)}", flush=True)

    if todo:  # 全在缓存里就不连 API，也不需要 key
        client = openai.OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL, max_retries=3, timeout=180)

    def fill(key):
        msgs = scenario_messages(by_sc[key], parse_scenario(scen_raw[key]["text"])["tools"])
        r = client.chat.completions.create(model=a.model, messages=msgs, temperature=0.3, max_tokens=2500,
                                           extra_body={"enable_thinking": False})
        return {"key": key, "model": a.model, "text": r.choices[0].message.content,
                "prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens}

    lock = threading.Lock()
    with ThreadPoolExecutor(a.workers) as pool, raw_path.open("a") as f:
        futures = [pool.submit(fill, k) for k in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except openai.APIError as e:  # 单个失败不影响整批，重跑会补
                print(f"  API error: {str(e)[:150]}", flush=True)
                continue
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            raw[r["key"]] = r
            if n % 50 == 0:
                print(f"  [{n}/{len(todo)}]", flush=True)

    out, fails = [], Counter()
    for key, group in by_sc.items():
        try:
            outs = parse_outputs(raw[key]["text"], len(group[0]["input"]["history"]))
        except (KeyError, ValueError, TypeError) as e:  # 缺缓存或解析失败，整个场景不要
            fails[type(e).__name__] += len(group)
            continue
        for s in group:
            s["input"]["history"] = [{**h, "output": o} for h, o in zip(s["input"]["history"], outs)]
            s["meta"]["output_model"] = raw[key]["model"]
            if s["label"] != "BLOCK" and missing_values(s):
                fails["missing_values"] += 1
                continue
            out.append(s)
    a.out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in out))
    tok = Counter()
    for r in raw.values():
        tok.update(prompt=r["prompt"], completion=r["completion"])
    print(f"{len(out)} samples -> {a.out}; failed samples {dict(fails)}")
    print(f"tokens {dict(tok)}, cost ~{(tok['prompt'] * PRICE_CNY_PER_M[0] + tok['completion'] * PRICE_CNY_PER_M[1]) / 1e6:.2f} 元 (原价)")


if __name__ == "__main__":
    main()
