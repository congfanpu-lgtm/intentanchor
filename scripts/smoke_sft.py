"""冒烟测试 2 / 3（设计文档 §7.1）：跑几步 SFT，报告 loss、显存峰值、每步耗时。

    python scripts/smoke_sft.py --model Qwen/Qwen3-0.6B --mode qlora --seq 512
    python scripts/smoke_sft.py --model Qwen/Qwen3-4B-Instruct-2507 --mode lora --seq 2048
"""
import argparse
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B")
p.add_argument("--mode", choices=["qlora", "lora"], default="qlora")
p.add_argument("--seq", type=int, default=512)
p.add_argument("--bs", type=int, default=1)
p.add_argument("--steps", type=int, default=10)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
# 每条样本都比 --seq 长、会被截断到 --seq，显存数字才代表真实序列长度
filler = "用户只要求总结邮件，没有要求发送邮件。The user asked for a summary, not to send an email. "
reps = a.seq // len(tok(filler).input_ids) + 2
ds = Dataset.from_dict({"text": [f"样本 {i}：" + filler * reps for i in range(100)]})

quant = None
if a.mode == "qlora":
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                             quantization_config=quant, device_map={"": 0})
weights_gb = torch.cuda.memory_allocated() / 2**30

step_times = []


class StepTimer(TrainerCallback):
    def on_step_begin(self, *args, **kwargs):
        torch.cuda.synchronize()
        self.t = time.perf_counter()

    def on_step_end(self, *args, **kwargs):
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - self.t)


trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    train_dataset=ds,
    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules="all-linear", task_type="CAUSAL_LM"),
    args=SFTConfig(output_dir="/tmp/smoke_sft", max_steps=a.steps, per_device_train_batch_size=a.bs,
                   max_length=a.seq, learning_rate=2e-4, gradient_checkpointing=True, bf16=True,
                   logging_steps=1, report_to="none", save_strategy="no"),
    callbacks=[StepTimer()],
)
torch.cuda.reset_peak_memory_stats()
trainer.train()

losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
steady = step_times[2:] or step_times  # 前两步含 CUDA 预热，不计入
print(f"\nRESULT model={a.model} mode={a.mode} seq={a.seq} bs={a.bs}")
print(f"  loss: {losses[0]:.3f} -> {losses[-1]:.3f} ({'下降' if losses[-1] < losses[0] else '没有下降'})")
print(f"  基座权重显存 {weights_gb:.2f} GB；训练峰值 allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB，"
      f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")
print(f"  每步耗时 {sum(steady) / len(steady):.2f} s（去掉前两步）")
