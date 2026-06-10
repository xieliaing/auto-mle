"""
Vision-language LoRA fine-tuning for the Kaggle Dog Breed Identification task.

The AutoMLE orchestrator (run.py) is text-only (AutoModelForCausalLM + a
tokenizer-based collator scoring a single Yes/No token). Dog breed ID is image
classification over 120 breeds, so this is a self-contained VLM path built on the
same ideas the framework uses (chat framing, answer-only loss masking, LoRA on
attention projections) but specialized for a vision-language model.

Model : Qwen/Qwen3-VL-2B-Instruct (bf16 base, frozen) + LoRA adapter on the LM.
Task  : given a dog photo, generate the breed name; accuracy = exact/normalized
        match against the 120 known breeds.

Data is read from this task's data/ folder (labels.csv + train/<id>.jpg), produced
by `python -m kaggle.setup --competition dog-breed-identification`.

Usage (from repo root):
    python kaggle/tasks/dog-breed-identification/vlm_lora.py \
        --train-subset 600 --val-subset 150 --epochs 2

    # full data (slow): --train-subset 0 --val-subset 0
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

TASK_DIR = Path(__file__).resolve().parent
DATA_DIR = TASK_DIR / "data"
IGNORE_INDEX = -100
IMAGE_SIZE = 448  # square resize; 448/28 = 16 -> ~256 visual tokens per image


# --- Data ---------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normalize a breed string for matching (lowercase, alnum-only)."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def load_split(seed: int, val_frac: float = 0.15):
    """Return (train_df, val_df, breeds) from labels.csv with a stratified-ish split."""
    df = pd.read_csv(DATA_DIR / "labels.csv")
    df = df.dropna(subset=["id", "breed"]).reset_index(drop=True)
    breeds = sorted(df["breed"].unique().tolist())
    # Per-breed shuffle then split so every breed appears in both halves.
    rng = random.Random(seed)
    val_idx: list[int] = []
    for _, grp in df.groupby("breed"):
        idx = grp.index.tolist()
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:k])
    val_mask = df.index.isin(val_idx)
    return (
        df[~val_mask].reset_index(drop=True),
        df[val_mask].reset_index(drop=True),
        breeds,
    )


def build_prompt(breeds: list[str]) -> str:
    listing = ", ".join(breeds)
    return (
        "You are an expert dog breed classifier. Identify the breed of the dog "
        "in the image. Answer with exactly one breed name from this list and "
        f"nothing else.\nBreeds: {listing}"
    )


def _load_image(image_id: str, split: str) -> Image.Image:
    img = Image.open(DATA_DIR / split / f"{image_id}.jpg").convert("RGB")
    return img.resize((IMAGE_SIZE, IMAGE_SIZE))


class DogBreedDataset(Dataset):
    """Serves chat messages: user(image + prompt) -> assistant(breed)."""

    def __init__(self, df: pd.DataFrame, prompt: str, with_answer: bool = True):
        self.df = df.reset_index(drop=True)
        self.prompt = prompt
        self.with_answer = with_answer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = _load_image(row["id"], "train")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": self.prompt},
            ]},
        ]
        if self.with_answer:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": row["breed"]}]}
            )
        return {"messages": messages, "breed": row["breed"]}


# --- Collator (answer-only loss masking) -------------------------------------

class VLMCollator:
    def __init__(self, processor, max_len: int = 1280):
        self.processor = processor
        self.max_len = max_len

    def _encode(self, messages, add_generation_prompt: bool):
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

    def __call__(self, batch: list[dict]) -> dict:
        # batch size is 1 in practice (per_device_train_batch_size=1).
        ex = batch[0]
        full = self._encode(ex["messages"], add_generation_prompt=False)
        prompt = self._encode(ex["messages"][:1], add_generation_prompt=True)
        prompt_len = prompt["input_ids"].shape[1]

        labels = full["input_ids"].clone()
        labels[:, :prompt_len] = IGNORE_INDEX  # supervise only the breed answer
        full["labels"] = labels
        # Trim to max_len (text+image tokens); image prefix is identical so the
        # mask boundary stays valid.
        if full["input_ids"].shape[1] > self.max_len:
            for k in ("input_ids", "attention_mask", "labels"):
                full[k] = full[k][:, : self.max_len]
        return full


# --- Eval ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, processor, val_ds: DogBreedDataset, breeds: list[str]) -> float:
    model.eval()
    norm_to_breed = {_norm(b): b for b in breeds}
    correct = 0
    n = len(val_ds)
    for i in range(n):
        ex = val_ds[i]
        prompt_msgs = ex["messages"][:1]  # drop the gold assistant turn
        inputs = processor.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=12, do_sample=False,
                             use_cache=True)
        new = gen[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(new, skip_special_tokens=True)[0]
        pred_n = _norm(text)
        # exact normalized match, else substring containment against known breeds
        match = norm_to_breed.get(pred_n)
        if match is None:
            for bn, b in norm_to_breed.items():
                if bn and (bn in pred_n or pred_n in bn):
                    match = b
                    break
        if match == ex["breed"]:
            correct += 1
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  eval {i + 1}/{n}  running acc={correct / (i + 1):.4f}")
    model.train()
    return correct / n if n else 0.0


# --- Main ---------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="VLM LoRA fine-tune on dog breed ID.")
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--train-subset", type=int, default=600, help="0 = all train")
    p.add_argument("--val-subset", type=int, default=150, help="0 = all val")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(TASK_DIR / "vlm_runs" / "run1"))
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading split from {DATA_DIR} ...")
    train_df, val_df, breeds = load_split(args.seed)
    if args.train_subset:
        train_df = train_df.sample(min(args.train_subset, len(train_df)),
                                   random_state=args.seed).reset_index(drop=True)
    if args.val_subset:
        val_df = val_df.sample(min(args.val_subset, len(val_df)),
                               random_state=args.seed).reset_index(drop=True)
    prompt = build_prompt(breeds)
    print(f"      breeds={len(breeds)}  train={len(train_df)}  val={len(val_df)}")

    print(f"[2/5] Loading {args.model} (bf16) + processor ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False

    print("[3/5] Attaching LoRA (attention projections of the LM) ...")
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()  # needed for grad checkpointing on a frozen base
    model.print_trainable_parameters()

    train_ds = DogBreedDataset(train_df, prompt, with_answer=True)
    val_ds = DogBreedDataset(val_df, prompt, with_answer=True)
    collator = VLMCollator(processor)

    print("[4/5] Training ...")
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=collator)
    train_out = trainer.train()

    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"      adapter saved -> {adapter_dir}")

    print("[5/5] Evaluating on held-out val ...")
    model.config.use_cache = True
    acc = evaluate(model, processor, val_ds, breeds)
    print("\n==============================================")
    print(f"  train_loss = {train_out.training_loss:.4f}")
    print(f"  val_accuracy = {acc:.4f}  (n={len(val_ds)}, {len(breeds)} breeds)")
    print(f"  random baseline ~= {1 / len(breeds):.4f}")
    print("==============================================")


if __name__ == "__main__":
    main()
