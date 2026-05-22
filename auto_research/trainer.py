"""
Generic QLoRA trainer template.

This file is the starting point for each new task. It is copied into the
task's folder on first init and the agent may freely modify it (e.g. to add
task-specific collation, masking, or callbacks).

The orchestrator imports the task's local copy via experiment_manager's
import_task_module, so edits in tasks/<key>/trainer.py take effect for
that task without touching this template.

Public contract used by the orchestrator:

    def train_one(
        cfg: ExperimentConfig,
        train_dataset,
        model_name: str,
        output_dir: Path,
        collator=None,
    ) -> dict:
        '''Returns {"train_loss": float, "adapter_dir": str}'''
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from auto_research.schema import ExperimentConfig
from auto_research.losses import make_loss_fn


# --- Model + tokenizer loading ------------------------------------------------

def load_model_and_tokenizer(model_name: str) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    return model, tokenizer


def make_lora_config(spec) -> LoraConfig:
    if spec.target_modules == "qv":
        targets = ["q_proj", "v_proj"]
    elif spec.target_modules == "attn":
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif spec.target_modules == "all_linear":
        targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
    else:
        raise ValueError(f"Unknown target_modules preset: {spec.target_modules}")

    return LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )


# --- Custom Trainer with pluggable loss --------------------------------------

class CustomLossTrainer(Trainer):
    """Trainer subclass that delegates loss computation to a callable."""

    def set_loss_fn(self, loss_fn):
        self._loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self._loss_fn(shift_logits, shift_labels)
        return (loss, outputs) if return_outputs else loss


# --- Public entry point ------------------------------------------------------

def train_one(
    cfg: ExperimentConfig,
    train_dataset,
    model_name: str,
    output_dir: Path,
    collator=None,
    pos_token_id: Optional[int] = None,
) -> dict:
    """Train a single experiment. The user's data.py should supply `collator`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_name)

    lora_cfg = make_lora_config(cfg.lora)
    model = get_peft_model(model, lora_cfg)

    if collator is None:
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    loss_fn = make_loss_fn(cfg.loss, pos_token_id=pos_token_id or 0)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.train.num_epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler,
        weight_decay=cfg.train.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        seed=cfg.train.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.set_loss_fn(loss_fn)

    train_out = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    return {
        "train_loss": float(train_out.training_loss),
        "adapter_dir": str(adapter_dir),
    }
