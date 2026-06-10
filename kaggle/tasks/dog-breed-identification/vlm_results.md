# Dog Breed Identification — VLM LoRA Results

Vision-language LoRA fine-tuning of `Qwen/Qwen3-VL-2B-Instruct` on the Kaggle
Dog Breed Identification task (120 breeds). See `vlm_lora.py` for the trainer and
`vlm_autoresearch.py` for the automated config search.

Eval = generate breed name, exact/normalized match against the 120 known breeds.
Random baseline = 1/120 ≈ 0.0083.

## Leaderboard

| run | lora | lr | epochs | train/val | train_loss | val_acc |
|-----|------|----|--------|-----------|-----------|---------|
| run1 (baseline) | r16 attn | 2e-4 | 2 | 600 / 150 | 0.3153 | **0.6600** |

## Run details

### run1 — baseline (2026-06-10)
- Model: `Qwen/Qwen3-VL-2B-Instruct`, bf16 base (frozen) + LoRA on `q/k/v/o_proj`.
- Trainable params: 6.42M (0.30% of 2.13B).
- LoRA r=16, alpha=32, dropout=0.05; lr=2e-4 cosine, warmup 0.03; bf16,
  gradient checkpointing, batch size 1 × grad-accum 8.
- Images resized to 448×448 (~256 visual tokens).
- Train: 600 images, 2 epochs (150 optimizer steps, ~10.4 min on RTX 4090 Laptop).
- **val_accuracy = 0.6600** (99/150), train_loss 0.3153.
- Adapter: `vlm_runs/run1/adapter/` (not tracked).
