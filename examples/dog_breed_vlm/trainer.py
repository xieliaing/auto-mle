"""
Dog-breed VLM trainer: bf16 Qwen3-VL base + LoRA on the language model, with
the core's pluggable answer-token losses (ce / label_smoothing / focal).

Mapping from ExperimentConfig:
  - lora.*, train.learning_rate / num_epochs / warmup_ratio / lr_scheduler /
    weight_decay / seed: used as-is.
  - train.batch_size: folded into grad accumulation (the VLM collator takes one
    example per step), so it never raises peak VRAM.
  - train.max_seq_len: ignored; image + 120-breed prompt needs the collator's
    fixed 1280-token window, and a shorter one would cut off the answer.
  - loss weighted_ce: binary-only (Yes-token weight), falls back to ce.
"""
from __future__ import annotations
from pathlib import Path

import torch
from peft import get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments

from auto_research.losses import make_loss_fn
from auto_research.schema import ExperimentConfig, LossConfigSpec
from auto_research.trainer import CustomLossTrainer, make_lora_config

from ._vlm import vlm


def train_one(cfg: ExperimentConfig, train_dataset, model_name: str,
              output_dir: Path, collator=None, **kwargs) -> dict:
    """Returns {"train_loss": float, "adapter_dir": str}"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(model, make_lora_config(cfg.lora))
    model.enable_input_require_grads()  # needed for grad checkpointing on a frozen base

    loss_spec = cfg.loss
    if loss_spec.name == "weighted_ce":
        print("[trainer] weighted_ce is binary-only; using ce for this multi-class task")
        loss_spec = LossConfigSpec(name="ce")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.train.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.train.grad_accum * cfg.train.batch_size,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler,
        weight_decay=cfg.train.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=cfg.train.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = CustomLossTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        data_collator=vlm.VLMCollator(processor),
    )
    trainer.set_loss_fn(make_loss_fn(loss_spec, pos_token_id=0))

    train_out = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    return {
        "train_loss": float(train_out.training_loss),
        "adapter_dir": str(adapter_dir),
    }
