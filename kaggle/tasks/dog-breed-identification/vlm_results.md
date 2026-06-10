# Dog Breed Identification — VLM LoRA Results

Vision-language LoRA fine-tuning of `Qwen/Qwen3-VL-2B-Instruct` on the Kaggle
Dog Breed Identification task (120 breeds). Trainer: `vlm_lora.py`; automated
config search: `vlm_autoresearch.py`. Random baseline = 1/120 ≈ 0.0083.

## Leaderboard (sorted by val_accuracy)

| run | lora | lr | epochs | train/val | train_loss | val_acc |
|-----|------|----|--------|-----------|-----------|---------|
| promote_e7_r32_all_linear | r32 all_linear | 0.0002 | 2.0 | 2000 / 150 | 0.1928 | **0.8267** **<- best** |
| e7_r32_all_linear | r32 all_linear | 0.0002 | 2.0 | 600 / 150 | 0.2917 | **0.7267** |
| e6_ep3 | r16 attn | 0.0002 | 3.0 | 600 / 150 | 0.2542 | **0.6933** |
| e3_all_linear | r16 all_linear | 0.0002 | 2.0 | 600 / 150 | 0.2989 | **0.6867** |
| e2_r32 | r32 attn | 0.0002 | 2.0 | 600 / 150 | 0.3093 | **0.6800** |
| e4_lr3e-4 | r16 attn | 0.0003 | 2.0 | 600 / 150 | 0.3125 | **0.6800** |
| run1 | r16 attn | 0.0002 | 2 | 600 / 150 | 0.3153 | **0.6600** |
| e5_lr1e-4 | r16 attn | 0.0001 | 2.0 | 600 / 150 | 0.3365 | **0.6267** |
