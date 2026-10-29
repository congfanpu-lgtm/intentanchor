"""DPO 守卫模型（设计文档 §5.4.3）：在 SFT 的 LoRA 上继续训，参考模型是训练前的 SFT 模型。

    .venv/bin/python -m intentanchor.train.dpo --train data/hard/<SFT 名>.jsonl \
        --init-adapter runs/<SFT>/best --out runs/<日期>-dpo-<名字>

训练样本是 scripts/mine_hard.py 挑出的判错样本。偏好对：chosen = 正确标签 + 风险来源；rejected 只把标签换成错的，
在判错样本上就是模型实际给出的标签。rl4k 没有理由，所以两边都不带理由，两个回答只差标签，信号落在打分用的标签 token 上。
参考模型：传入的是带已训 adapter 的 PeftModel 时，TRL 自动复制一份冻结的 "ref" adapter，不用另载模型。
不在开发集上评估，存最后一步的权重到 <out>/final；选哪个版本，训完用 score_vllm 在开发集上比。
"""
import argparse
import hashlib
import json
from pathlib import Path

from intentanchor.prompt import LABELS, answer_text, build_messages


def to_pair(s: dict, variant: str = "B") -> dict:
    wrong, = set(LABELS) - {s["label"]}
    reply = lambda label: [{"role": "assistant", "content": answer_text(label, s.get("risk_source"), None)}]
    return {"prompt": build_messages(s["input"], variant), "chosen": reply(s["label"]), "rejected": reply(wrong)}


def main():
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True, help="scripts/mine_hard.py 的输出")
    p.add_argument("--init-adapter", type=Path, required=True, help="SFT 的 LoRA 权重")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-steps", type=int, default=-1, help="冒烟测试用")
    p.add_argument("--variant", choices=["A", "B"], default="B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    train = Dataset.from_list([to_pair(s, a.variant) for s in map(json.loads, a.train.read_text().splitlines())])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run_config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(a).items()},
         "data_sha256": {str(a.train): hashlib.sha256(a.train.read_bytes()).hexdigest()[:16]}, "n_train": len(train)},
        indent=2, ensure_ascii=False))

    tok = AutoTokenizer.from_pretrained(a.model)
    longest = max(len(tok(tok.apply_chat_template(e["prompt"] + e["chosen"], tokenize=False)).input_ids)
                  for e in train)
    assert longest <= a.max_length, f"longest example {longest} tokens > --max-length, answers would be cut off"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, quantization_config=quant,
                                                 device_map={"": 0})
    model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    trainer = DPOTrainer(
        model=model, processing_class=tok, train_dataset=train,
        args=DPOConfig(output_dir=str(a.out / "ckpt"), beta=a.beta, num_train_epochs=a.epochs, max_steps=a.max_steps,
                       learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.03,
                       per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
                       max_length=a.max_length, gradient_checkpointing=True, bf16=True, save_strategy="no",
                       logging_steps=5, seed=a.seed, report_to="none"),
    )
    assert "ref" in trainer.model.peft_config, "TRL did not copy the SFT adapter as the reference"
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.model.save_pretrained(str(a.out / "final"), selected_adapters=["default"])
    (a.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    print(json.dumps({"peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                      "longest_example_tokens": longest}))


if __name__ == "__main__":
    main()
