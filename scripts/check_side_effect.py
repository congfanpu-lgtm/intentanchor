"""校验 configs/side_effect.yaml（设计文档 §4.4）：

1. 每个 suite 的工具都标了，没有多余的；
2. 每个注入任务的标准答案（ground_truth）里至少有一次副作用调用，否则"只查副作用工具"会漏掉这种攻击；
3. 统计用户任务标准答案里副作用调用的占比，估算守卫能少检查多少次。

    .venv-serve/bin/python scripts/check_side_effect.py
"""
import sys
from pathlib import Path

import yaml
from agentdojo.task_suite.load_suites import get_suite

labels = yaml.safe_load((Path(__file__).parent.parent / "configs/side_effect.yaml").read_text())
failed = False

# v1.2 新加的注入任务没有写 ground_truth（返回空列表），人工看过 GOAL 和 security() 后的结论
MANUAL = {
    **{("workspace", f"injection_task_{i}"): "要 send_email 把邮件或文件发给攻击者，再 delete_email / delete_file 删痕迹"
       for i in range(6, 14)},
    ("travel", "injection_task_6"): "OUT_OF_SCOPE：只要求 agent 在最终回答里说一句推荐酒店的话，不经过任何工具；"
                                    "只查工具调用的守卫抓不到（StepGuard 也一样），报告时注明",
}

for name, tools in labels.items():
    suite = get_suite("v1.2.2", name)
    actual = {f.name for f in suite.tools}
    if set(tools) != actual:
        print(f"[FAIL] {name}: 缺 {sorted(actual - set(tools))}，多 {sorted(set(tools) - actual)}")
        failed = True
        continue

    env = suite.load_and_inject_default_environment({})
    covered, manual, out_of_scope = 0, 0, []
    for it_id, it in suite.injection_tasks.items():
        calls = [c.function for c in it.ground_truth(env.model_copy(deep=True))]
        note = MANUAL.get((name, it_id))
        if any(tools[c] for c in calls):
            covered += 1
        elif not calls and note:
            manual += 1
            if note.startswith("OUT_OF_SCOPE"):
                out_of_scope.append(it_id)
        else:
            print(f"[FAIL] {name}/{it_id} 的标准答案里没有副作用调用，也没有人工结论：{calls}")
            failed = True

    total = side = 0
    for ut in suite.user_tasks.values():
        calls = [c.function for c in ut.ground_truth(env.model_copy(deep=True))]
        total += len(calls)
        side += sum(tools[c] for c in calls)
    n_side = sum(tools.values())
    print(f"{name:10s} 工具 {len(tools):2d} 个，副作用 {n_side:2d} 个｜注入任务 {len(suite.injection_tasks):2d} 个："
          f"标准答案验证 {covered}，人工核对 {manual}，守卫管不到 {out_of_scope or '无'}｜"
          f"用户任务标准答案 {total:3d} 次调用，副作用 {side:3d} 次（{side / total:.0%}）")

sys.exit(1 if failed else 0)
