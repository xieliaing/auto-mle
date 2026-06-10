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


# --- Multi-class log loss (Kaggle metric) ------------------------------------
#
# A generative VLM emits a string, not a class distribution. To get the 120-way
# probabilities the log-loss metric needs, we score every breed name by its
# teacher-forced sequence log-likelihood given the same image+prompt, then
# softmax over the 120 candidates -> a proper distribution. Kaggle clips each
# probability to [1e-15, 1-1e-15] before taking the log.

def _breed_answer_ids(processor, breeds, prompt, ref_image) -> list[list[int]]:
    """Token ids of each breed name as it follows the generation prompt.

    Derived as (full chat) minus (prompt-only chat) on a reference image, with
    trailing special tokens (e.g. <|im_end|>) stripped so only the breed-name
    tokens are scored. Breed answer tokens are image-independent, so one
    reference image suffices.
    """
    tok = processor.tokenizer
    specials = set(tok.all_special_ids)
    prompt_msgs = [{"role": "user", "content": [
        {"type": "image", "image": ref_image}, {"type": "text", "text": prompt}]}]
    prompt_ids = processor.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")["input_ids"][0].tolist()
    plen = len(prompt_ids)
    out = []
    for b in breeds:
        full_msgs = prompt_msgs + [{"role": "assistant",
                                    "content": [{"type": "text", "text": b}]}]
        full_ids = processor.apply_chat_template(
            full_msgs, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt")["input_ids"][0].tolist()
        ans = full_ids[plen:]
        while ans and ans[-1] in specials:  # drop trailing <|im_end|> / newline-eos
            ans.pop()
        out.append(ans)
    return out


@torch.no_grad()
def _score_image(model, processor, image, prompt, breed_ids, sub_batch=8):
    """Return a (n_breeds,) tensor of summed answer-token log-probs for one image."""
    prefix = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    pre_ids = prefix["input_ids"][0]
    P = pre_ids.shape[0]
    pad_id = processor.tokenizer.pad_token_id or 0
    scores = torch.full((len(breed_ids),), float("-inf"))

    for s in range(0, len(breed_ids), sub_batch):
        group = breed_ids[s:s + sub_batch]
        B = len(group)
        maxA = max(len(a) for a in group)
        input_ids = torch.full((B, P + maxA), pad_id, dtype=torch.long)
        attn = torch.zeros((B, P + maxA), dtype=torch.long)
        for j, ans in enumerate(group):
            input_ids[j, :P] = pre_ids
            input_ids[j, P:P + len(ans)] = torch.tensor(ans, dtype=torch.long)
            attn[j, :P + len(ans)] = 1
        batch = {
            "input_ids": input_ids.to(model.device),
            "attention_mask": attn.to(model.device),
            "pixel_values": prefix["pixel_values"].repeat(B, 1).to(model.device),
            "image_grid_thw": prefix["image_grid_thw"].repeat(B, 1).to(model.device),
        }
        logits = model(**batch).logits  # (B, P+maxA, V)
        # token at absolute pos P+k is predicted by logits at pos P-1+k
        ans_logits = logits[:, P - 1:P - 1 + maxA, :]
        logp = torch.log_softmax(ans_logits.float(), dim=-1)  # (B, maxA, V)
        for j, ans in enumerate(group):
            idx = torch.tensor(ans, device=model.device)
            tok_lp = logp[j, torch.arange(len(ans), device=model.device), idx]
            scores[s + j] = tok_lp.sum().cpu()
    return scores


@torch.no_grad()
def evaluate_logloss(model, processor, val_ds, breeds, prompt, sub_batch=8):
    """Return (logloss, scoring_top1_accuracy) over the val set via breed scoring."""
    model.eval()
    breed_idx = {b: i for i, b in enumerate(breeds)}
    ref_img = val_ds[0]["messages"][0]["content"][0]["image"]
    breed_ids = _breed_answer_ids(processor, breeds, prompt, ref_img)
    n = len(val_ds)
    total_ll, correct = 0.0, 0
    for i in range(n):
        ex = val_ds[i]
        img = ex["messages"][0]["content"][0]["image"]
        scores = _score_image(model, processor, img, prompt, breed_ids, sub_batch)
        probs = torch.softmax(scores, dim=0)
        true_i = breed_idx[ex["breed"]]
        p_true = float(probs[true_i].clamp(1e-15, 1 - 1e-15))
        total_ll += -torch.log(torch.tensor(p_true)).item()
        if int(torch.argmax(scores)) == true_i:
            correct += 1
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  logloss-eval {i + 1}/{n}  "
                  f"running logloss={total_ll / (i + 1):.4f} acc={correct / (i + 1):.4f}")
    model.train()
    return total_ll / n if n else float("nan"), correct / n if n else 0.0


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
    p.add_argument("--lora-target", choices=["qv", "attn", "all_linear"], default="attn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(TASK_DIR / "vlm_runs" / "run1"))
    p.add_argument("--eval-adapter", default=None,
                   help="Path to a trained adapter dir; skips training and only evaluates.")
    p.add_argument("--sub-batch", type=int, default=8,
                   help="Candidate breeds scored per forward pass during log-loss eval.")
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

    val_ds = DogBreedDataset(val_df, prompt, with_answer=True)

    # --- Eval-only mode: load a trained adapter and skip training ------------
    if args.eval_adapter:
        from peft import PeftModel
        print(f"[3/5] Loading adapter {args.eval_adapter} (eval-only) ...")
        model = PeftModel.from_pretrained(model, args.eval_adapter)
        model.config.use_cache = True
        print("[5/5] Evaluating on held-out val ...")
        acc = evaluate(model, processor, val_ds, breeds)
        logloss, acc_scored = evaluate_logloss(
            model, processor, val_ds, breeds, prompt, args.sub_batch)
        print("\n==============================================")
        print(f"  val_accuracy (generation) = {acc:.4f}")
        print(f"  val_accuracy (scoring)    = {acc_scored:.4f}")
        print(f"  val_logloss               = {logloss:.4f}")
        print(f"  random-guess logloss      = {__import__('math').log(len(breeds)):.4f}")
        print("==============================================")
        return

    print(f"[3/5] Attaching LoRA (target={args.lora_target}) ...")
    targets = {
        "qv": ["q_proj", "v_proj"],
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    }[args.lora_target]
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()  # needed for grad checkpointing on a frozen base
    model.print_trainable_parameters()

    train_ds = DogBreedDataset(train_df, prompt, with_answer=True)
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
    logloss, acc_scored = evaluate_logloss(
        model, processor, val_ds, breeds, prompt, args.sub_batch)
    metrics = {
        "model": args.model, "lora_r": args.lora_r, "lora_target": args.lora_target,
        "lr": args.lr, "epochs": args.epochs, "grad_accum": args.grad_accum,
        "train_subset": len(train_df), "val_subset": len(val_ds),
        "n_breeds": len(breeds), "seed": args.seed,
        "train_loss": float(train_out.training_loss), "val_accuracy": float(acc),
        "val_accuracy_scored": float(acc_scored), "val_logloss": float(logloss),
    }
    import json
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n==============================================")
    print(f"  train_loss = {train_out.training_loss:.4f}")
    print(f"  val_accuracy (generation) = {acc:.4f}  (n={len(val_ds)}, {len(breeds)} breeds)")
    print(f"  val_accuracy (scoring)    = {acc_scored:.4f}")
    print(f"  val_logloss               = {logloss:.4f}  (random ~= {1.0:.4f}->{__import__('math').log(len(breeds)):.4f})")
    print(f"  metrics -> {out / 'metrics.json'}")
    print("==============================================")


if __name__ == "__main__":
    main()
