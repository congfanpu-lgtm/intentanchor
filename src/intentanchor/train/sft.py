"""SFT 守卫模型（设计文档 §5.4.2）：QLoRA 或 bf16 LoRA，只对回答部分算损失。

    .venv/bin/python -m intentanchor.train.sft --train data/h2/train.jsonl --dev data/h2/dev.jsonl \
        --mode qlora --epochs 2 --bs 2 --grad-accum 8 --seed 0 --out runs/<日期>-sft-h2s0-qlora

回答 = 标签、风险来源、理由（§4.2，prompt.answer_text）。StepGuard 样本的理由是它的 <Analysis>，合成样本有一句理由，其余没有。
每个 epoch 末在开发集上算 eval loss，结束时取 eval loss 最低的 checkpoint，LoRA 权重存到 <out>/best。
<out>/run_config.json 记参数和数据文件哈希，<out>/log_history.json 记 loss 曲线。
--init-adapter 在已有的 LoRA 上继续训（DPO 的对照组"困难样本继续 SFT"，§5.4.3），不新建 LoRA。
"""
import argparse
import hashlib
import json
from pathlib import Path

from intentanchor.prompt import answer_text, build_messages


def to_example(s: dict, variant: str = "B") -> dict:
    answer = answer_text(s["label"], s.get("risk_source"), s.get("reason"))
    return {"prompt": build_messages(s["input"], variant), "completion": [{"role": "assistant", "content": answer}]}


def main():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--mode", choices=["qlora", "lora"], default="qlora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--variant", choices=["A", "B"], default="B", help="输入变体（§4.1）")
    p.add_argument("--init-adapter", type=Path, help="在这个 LoRA 上继续训")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    load = lambda path: [to_example(s, a.variant) for s in map(json.loads, path.read_text().splitlines())]
    train, dev = Dataset.from_list(load(a.train)), Dataset.from_list(load(a.dev))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run_config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(a).items()},
         "data_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in (a.train, a.dev)},
         "n_train": len(train), "n_dev": len(dev)}, indent=2, ensure_ascii=False))

    tok = AutoTokenizer.from_pretrained(a.model)
    longest = max(len(tok(tok.apply_chat_template(e["prompt"] + e["completion"], tokenize=False)).input_ids)
                  for e in train)  # transformers 5 里 tokenize=True 返回的是字典，不能直接取 len
    assert longest <= a.max_length, f"longest example {longest} tokens > --max-length, answers would be cut off"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16) if a.mode == "qlora" else None
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, quantization_config=quant,
                                                 device_map={"": 0})
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=train, eval_dataset=dev,
        peft_config=None if a.init_adapter else LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                           target_modules="all-linear", task_type="CAUSAL_LM"),
        args=SFTConfig(output_dir=str(a.out / "ckpt"), num_train_epochs=a.epochs, learning_rate=a.lr,
                       lr_scheduler_type="cosine", warmup_steps=0.03, per_device_train_batch_size=a.bs,
                       per_device_eval_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
                       max_length=a.max_length, completion_only_loss=True, gradient_checkpointing=True, bf16=True,
                       eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                       metric_for_best_model="eval_loss", save_total_limit=1, logging_steps=10, seed=a.seed,
                       report_to="none"),
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.save_model(str(a.out / "best"))
    (a.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    print(json.dumps({"best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
                      "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                      "longest_example_tokens": longest}))


if __name__ == "__main__":
    main()
