"""
Dog-breed VLM evaluator: greedy-generate a breed name per image and score
exact (normalized) match against the gold breed.

With return_records=True, every example becomes a record whose "image" path
lets defect analysis attach the misclassified photos to the analyst prompt.
"""
from __future__ import annotations

import torch

from ._vlm import vlm

PRIMARY_METRIC = "accuracy"


def _match(text: str, norm_to_breed: dict) -> str | None:
    """Exact normalized match, else substring containment (as in vlm_lora.evaluate)."""
    pred_n = vlm._norm(text)
    if not pred_n:
        return None
    if pred_n in norm_to_breed:
        return norm_to_breed[pred_n]
    for bn, b in norm_to_breed.items():
        if bn in pred_n or pred_n in bn:
            return b
    return None


@torch.no_grad()
def evaluate(adapter_dir: str, model_name: str, eval_data,
             return_records: bool = False, **kwargs) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    norm_to_breed = {vlm._norm(b): b for b in eval_data.breeds}
    n = len(eval_data)
    correct, unmatched, records = 0, 0, []
    for i in range(n):
        ex = eval_data[i]
        inputs = processor.apply_chat_template(
            ex["messages"][:1], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=True)
        text = processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        pred = _match(text, norm_to_breed)
        ok = pred == ex["breed"]
        correct += ok
        unmatched += pred is None
        if return_records:
            image_id = eval_data.df.iloc[i]["id"]
            records.append({
                "id": image_id,
                "image": str(vlm.DATA_DIR / "train" / f"{image_id}.jpg"),
                "expected": ex["breed"],
                "predicted": pred if pred is not None else f"(not a breed) {text}",
                "is_defect": not ok,
            })
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  eval {i + 1}/{n}  running acc={correct / (i + 1):.4f}")

    out = {"accuracy": correct / n if n else 0.0, "n_eval": n, "unmatched": unmatched}
    if return_records:
        out["records"] = records
    return out
