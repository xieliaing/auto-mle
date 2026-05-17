"""
QLoRA trainer wrapper.

Wires together:
  - 4-bit NF4 quantized base model (Qwen3 / Qwen2.5)
  - LoRA adapter via peft_config (handed to SFTTrainer, not get_peft_model)
  - Custom collator that applies the chat template, then masks every token
    except the assistant's answer token so loss is computed on that one position
  - Custom Trainer subclass that swaps in focal / label-smoothing / weighted CE

Memory choices for 16GB VRAM:
  - bnb_4bit_compute_dtype=bfloat16, double quant
  - gradient checkpointing on
  - paged 8-bit AdamW
  - per-device batch size 1, gradient accumulation in config
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
)
from peft import LoraConfig, prepare_model_for_kbit_training

from .schema import ExperimentConfig
from .losses import make_loss_fn, IGNORE_INDEX
from .prompts import get_yes_no_token_ids


# --- Model + tokenizer loading ------------------------------------------------

def load_model_and_tokenizer(model_name: str) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """Load tokenizer + 4-bit quantized base model ready for QLoRA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Most Qwen tokenizers have <|endoftext|> or <|im_end|>; fall back to eos.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT-style; eval will handle its own side

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
    model.config.use_cache = False  # required when grad checkpointing is on
    return model, tokenizer


def make_lora_config(spec) -> LoraConfig:
    """Build a peft LoraConfig from our LoraConfigSpec."""
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


# --- Collator -----------------------------------------------------------------

@dataclass
class ProductPairCollator:
    """
    Builds a batch from raw dataset items: applies the chat template, tokenizes,
    and creates `labels` that only score the answer token.

    Strategy for masking:
      1. We tokenize the FULL conversation (with the assistant turn) -> input_ids.
      2. We tokenize the prompt-only conversation (system + user, no assistant)
         with `add_generation_prompt=True` to get the prompt length.
      3. We set labels[:prompt_len] = -100 so loss only sees the answer + closing
         <|im_end|>. In practice the answer is exactly one token ("Yes" or "No"),
         so this is per-example-1-token supervision.
    """
    tokenizer: "AutoTokenizer"
    max_seq_len: int = 512

    def __call__(self, batch: list[dict]) -> dict:
        full_msgs = [b["messages"] for b in batch]
        # Prompt-only = drop the last (assistant) turn
        prompt_msgs = [m[:-1] for m in full_msgs]

        # Encode full conversations (no padding yet — we pad after masking)
        full_texts = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False,
        )
        prompt_texts = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )

        full_enc = self.tokenizer(
            full_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        prompt_enc = self.tokenizer(
            prompt_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )

        input_ids_list, labels_list, attn_list = [], [], []
        for full_ids, prompt_ids in zip(full_enc["input_ids"], prompt_enc["input_ids"]):
            full_ids = list(full_ids)
            prompt_len = len(prompt_ids)
            # Guard: if truncation cut off the answer, skip masking and accept
            # full-sequence loss (rare with max_seq_len=512 for short titles).
            if prompt_len >= len(full_ids):
                labels = [IGNORE_INDEX] * len(full_ids)
            else:
                labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attn_list.append([1] * len(full_ids))

        # Manual right-pad to the longest in batch
        max_len = max(len(x) for x in input_ids_list)
        pad_id = self.tokenizer.pad_token_id
        for i in range(len(input_ids_list)):
            pad = max_len - len(input_ids_list[i])
            input_ids_list[i] = input_ids_list[i] + [pad_id] * pad
            labels_list[i] = labels_list[i] + [IGNORE_INDEX] * pad
            attn_list[i] = attn_list[i] + [0] * pad

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_list, dtype=torch.long),
        }


# --- Custom Trainer with pluggable loss --------------------------------------

class CustomLossTrainer(Trainer):
    """
    Subclass that uses our custom loss function. The loss callable is set via
    `set_loss_fn(...)` after construction.

    Important detail about shifting: HuggingFace causal LMs *internally* shift
    labels by 1 when you pass labels to the model. We bypass that by NOT
    passing labels to the forward call, computing logits, then doing the shift
    ourselves before delegating to our loss_fn. This keeps our loss
    implementations simple and consistent.
    """

    def set_loss_fn(self, loss_fn):
        self._loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        # Standard causal shift: predict token t from positions < t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self._loss_fn(shift_logits, shift_labels)
        return (loss, outputs) if return_outputs else loss


# --- Public entry point: run one training experiment -------------------------

def train_one(
    cfg: ExperimentConfig,
    train_dataset: Dataset,
    model_name: str,
    output_dir: Path,
) -> dict:
    """
    Train a single experiment end-to-end. Returns a dict of metrics.

    The caller is responsible for evaluation (see evaluator.py); this function
    just trains and saves the LoRA adapter.
    """
    from transformers import TrainingArguments

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_name)
    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    # Apply LoRA
    lora_cfg = make_lora_config(cfg.lora)
    from peft import get_peft_model
    model = get_peft_model(model, lora_cfg)

    collator = ProductPairCollator(tokenizer=tokenizer, max_seq_len=cfg.train.max_seq_len)
    loss_fn = make_loss_fn(cfg.loss, pos_token_id=yes_id)

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
        save_strategy="no",     # we save the adapter manually at the end
        report_to=[],
        seed=cfg.train.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # safer in notebooks
    )

    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.set_loss_fn(loss_fn)

    train_out = trainer.train()
    # Save adapter
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    return {
        "train_loss": float(train_out.training_loss),
        "adapter_dir": str(adapter_dir),
        "yes_token_id": int(yes_id),
        "no_token_id": int(no_id),
    }
