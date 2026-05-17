# Auto-Research: Product Comparison via Text-Only QLoRA

An autonomous experimentation pipeline that fine-tunes a small Qwen-family model on the binary product-matching task and iteratively searches for better configurations along three axes:

1. **Fine-tuning strategy** — LoRA rank, alpha, target modules, learning rate, scheduler, epochs.
2. **Data strategy** — class balancing, hard-negative mining, title augmentation, Jaccard filtering, subset size.
3. **Loss function** — standard CE, label smoothing, focal loss, weighted CE.

The pipeline: **Proposer → Trainer → Evaluator → Analyzer → loop**.

## Layout

```
auto_research/
├── run.py                      # entry point
├── configs/default.yaml        # paths, model, budgets
├── auto_research/
│   ├── schema.py               # ExperimentConfig + ExperimentResult dataclasses
│   ├── prompts.py              # text-only chat template (system + user → "Yes"/"No")
│   ├── data.py                 # CSV loader, balancing, hard-neg mining, augment
│   ├── losses.py               # CE / label-smooth / focal / weighted-CE
│   ├── trainer.py              # 4-bit QLoRA + custom answer-token-only loss
│   ├── evaluator.py            # fast logit-based Yes-vs-No scoring
│   ├── proposer.py             # seed plan + Claude-driven proposer
│   ├── analyzer.py             # leaderboard.md + plateau detection
│   └── orchestrator.py         # the autonomous loop (Phase A: explore, Phase B: promote)
└── results/                    # populated at runtime
    ├── results.jsonl
    ├── leaderboard.md
    └── runs/<exp_id>/
        ├── config.json
        └── adapter/            # the LoRA adapter for that run
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set the proposer API key (or pass --no-llm-proposer to skip)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Point at your CSVs and run
python run.py \
    --train data/train_pairs.csv \
    --eval data/test_pairs.csv \
    --model Qwen/Qwen3-1.7B \
    --budget 10 \
    --final-top-k 3
```

After it finishes:
- `results/leaderboard.md` — markdown ranking
- `results/results.jsonl` — one JSON line per run (config + metrics)
- `results/runs/<exp_id>/adapter/` — the LoRA adapter for each run

## Note on the base model

`Qwen3-2B-instruct` is not an officially released checkpoint. The pipeline defaults to `Qwen/Qwen3-1.7B`, which is the closest 2B-class Qwen3 model and fits comfortably in 16GB VRAM with 4-bit QLoRA. If you want a different size, just change `--model` (or edit `configs/default.yaml`):
- `Qwen/Qwen3-4B` — bigger, may need `max_seq_len ≤ 384` and `grad_accum ≥ 32`.
- `Qwen/Qwen2.5-1.5B-Instruct` — solid alternative if you want the Qwen2.5 family.

## How the loop works

**Phase A — Exploration (tier=`small`)**
Trains on a 20K subset, evaluates on a 2K subset (~30–60 min/run on a mobile 4090). The first six rounds follow a fixed seed plan covering: baseline / bigger LoRA / focal / label smoothing / hard negatives / class balancing. From round 7 onward, the proposer (Claude via the Anthropic API) reads `results.jsonl` and proposes the next config in JSON. If the API call fails, the pipeline falls back to a heuristic perturbation of the best-so-far config.

Stops early if either:
- `--target-accuracy` is reached, or
- no improvement >0.2% in the last 4 successful runs.

**Phase B — Promotion (tier=`full`)**
The top-K configs from Phase A are re-trained on the full `train_pairs.csv` and evaluated on the full `test_pairs.csv` for a definitive number.

## Why these design choices

- **4-bit NF4 + double-quant + bf16 compute, paged 8-bit AdamW, gradient checkpointing.** Standard QLoRA recipe for 16GB VRAM. Each experiment uses ~10–12GB, leaving headroom for activations.
- **Custom collator that masks every token except the answer.** The supervised target is exactly one token (`Yes` or `No`). Including prompt tokens in the loss wastes compute and pushes the model away from useful representations.
- **Logit-based eval rather than `.generate()`.** For binary classification we only need to compare `logit[Yes]` vs `logit[No]` at the next-token position. ~10× faster than calling `.generate()`.
- **Tiered evaluation.** Full eval on 33K pairs takes ~6× longer than a 2K subset; running it for every experiment wastes 80%+ of the compute on configs that will be thrown away. We use the small tier for search and the full tier only for confirmation.
- **Seed plan before LLM proposer.** The first six experiments are deterministic, so the LLM has real signal to reason about by round 7. This avoids the failure mode of LLM proposers that hallucinate plausible-sounding but uninformative configs early on.

## Resume

The orchestrator reads `results.jsonl` at start and only runs experiments whose `exp_id` isn't already there. Killing and restarting the process is safe.

## Extending

Adding a new tunable knob = adding one field to the corresponding sub-config in `auto_research/schema.py`, plus the code that consumes it (in `data.py`, `trainer.py`, or `losses.py`). The proposer schema in `proposer.py` should be updated to match.
