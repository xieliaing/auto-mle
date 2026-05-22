"""
Fast evaluation: instead of generating tokens, we run a single forward pass
per batch and compare logits[Yes] vs logits[No] at the position right after
the prompt. This is ~10x faster than .generate() and exactly equivalent for
this binary task.

We always evaluate in bf16 + KV-cache disabled (we only need one forward).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .prompts import build_messages, get_yes_no_token_ids


PRIMARY_METRIC = "accuracy"


def _load_for_eval(model_name: str, adapter_dir: Optional[str]):
    """Load 4-bit base model + (optional) LoRA adapter for evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batched decoding alignment

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
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def _build_prompt_only_batch(rows, tokenizer, max_seq_len: int):
    """Tokenize prompt-only conversations (with generation prompt) for a batch."""
    msgs_list = [build_messages(r["title1"], r["title2"], answer=None) for r in rows]
    texts = tokenizer.apply_chat_template(
        msgs_list, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,                 # left-padded thanks to padding_side
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return enc


@torch.no_grad()
def evaluate(
    adapter_dir: Optional[str],
    model_name: str,
    eval_data: pd.DataFrame,
    batch_size: int = 8,
    max_seq_len: int = 512,
    progress: bool = True,
    **kwargs,
) -> dict:
    """
    Score every row of eval_data. Returns {accuracy, n_eval, tp, tn, fp, fn,
    precision, recall, f1, yes_token_id, no_token_id}.
    """
    eval_df = eval_data
    model, tokenizer = _load_for_eval(model_name, adapter_dir)
    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    rows = eval_df.to_dict(orient="records")
    n = len(rows)
    correct = 0
    tp = tn = fp = fn = 0

    iterator = range(0, n, batch_size)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="eval", total=(n + batch_size - 1) // batch_size)
        except ImportError:
            pass

    for start in iterator:
        chunk = rows[start:start + batch_size]
        enc = _build_prompt_only_batch(chunk, tokenizer, max_seq_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model(**enc, use_cache=False)
        logits = out.logits  # (B, T, V)
        # With LEFT padding, the last position of each row is always the
        # next-token slot, regardless of true length. Use index -1.
        last = logits[:, -1, :]
        yes_l = last[:, yes_id]
        no_l = last[:, no_id]
        preds = (yes_l > no_l).long().cpu().tolist()

        for r, p in zip(chunk, preds):
            y = int(r["Label"])
            if p == y:
                correct += 1
            if p == 1 and y == 1:
                tp += 1
            elif p == 0 and y == 0:
                tn += 1
            elif p == 1 and y == 0:
                fp += 1
            else:
                fn += 1

    acc = correct / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Free GPU before returning so the next experiment has clean memory
    del model
    torch.cuda.empty_cache()

    return {
        "accuracy": acc,
        "n_eval": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "yes_token_id": int(yes_id), "no_token_id": int(no_id),
    }
