# AutoMLE — Agent Program Guide

This is the operational guide for the AutoMLE parent base. Agents must follow this document when conducting LoRA fine-tuning experiments.

---

## Document Reading Order

When entering the project for the first time, read in this order:

1. `program.md` (this file)
2. `configs/default.yaml`
3. `auto_research/schema.py` — experiment config schema
4. `auto_research/orchestrator.py` — the auto-research loop
5. `examples/product_comparison/` — a working reference task

---

## Project Overview

AutoMLE is a parent **template** for autonomous LoRA fine-tuning experiments. The `auto_research/` package is task-agnostic; each new task plugs in by providing a `data.py` and `evaluator.py`. The agent may also edit the per-task `trainer.py` copy.

The closed loop: **Propose → Train → Evaluate → Analyze → Repeat**.

Each task is identified by a **key** (user-supplied or auto-generated). Per key:
- A feature branch `feature/<key>` is created.
- A folder `tasks/<key>/` is created containing the task's `data.py`, `evaluator.py`, and `trainer.py`.
- Each invocation creates a new `tasks/<key>/runs/<timestamp>/` for that run's results.

The codebase accepts these inputs per experiment:

| Input | Flag | Description |
|-------|------|-------------|
| Base checkpoint | `--checkpoint` | HuggingFace model ID or local path |
| Data module | `--data-file` | Python module: `data.py` (see contract below) |
| Evaluator module | `--evaluator-file` | Python module: `evaluator.py` (see contract below) |
| Task key | `--key` | Optional; auto-generated as `task_<random>` if omitted |

---

## Starting a New Task

```bash
python run.py \
  --key my-task \
  --checkpoint <model_id_or_local_path> \
  --data-file path/to/data.py \
  --evaluator-file path/to/evaluator.py
```

### What happens automatically
1. Git repository is initialized if not already present.
2. The key is sanitized (lowercased, `[^a-z0-9_-]` → `_`) or auto-generated.
3. A feature branch `feature/<key>` is created (or reused if it already exists).
4. Folder `tasks/<key>/` is created with `data.py`, `evaluator.py`, and a copy of `auto_research/trainer.py` (the agent may modify the task's copy).
5. `meta.json` is written with task metadata.
6. A new `tasks/<key>/runs/<timestamp>/` is created for this run's results.
7. The auto-research loop begins.

### Skip branch creation (for development)
```bash
python run.py --no-branch --checkpoint ... --data-file ... --evaluator-file ...
```

---

## User-Provided Module Contracts

### `data.py`

Required:
```python
def load_train_dataset(seed: int = 42, subset: int | None = None,
                       balance: str = "none", **kwargs) -> torch.utils.data.Dataset:
    """Return a Dataset whose items are consumable by `get_collator`'s output."""

def load_eval_dataset(seed: int = 42, subset: int | None = None, **kwargs):
    """Return whatever evaluator.evaluate()'s `eval_data` argument expects."""
```

Optional:
```python
def get_collator(tokenizer, max_seq_len: int):
    """Return a data collator. If absent, the trainer falls back to a default."""
```

### `evaluator.py`

Required:
```python
def evaluate(adapter_dir: str, model_name: str, eval_data, **kwargs) -> dict:
    """Return a dict including the primary metric. May also include 'n_eval' etc."""
```

Optional:
```python
PRIMARY_METRIC: str = "accuracy"  # name of the key in the returned dict to optimize
```

### `trainer.py` (template, modifiable per task)

```python
def train_one(cfg: ExperimentConfig, train_dataset, model_name: str,
              output_dir: Path, collator=None, **kwargs) -> dict:
    """Returns {"train_loss": float, "adapter_dir": str}"""
```

See `examples/product_comparison/` for a working reference implementation of all three.

---

## Agent Workflow

### Phase 1: Setup (User Confirms)

| Step | Action | User Confirmation Required? |
|------|--------|----------------------------|
| 1 | Verify environment (CUDA, RAM, VRAM) | Yes |
| 2 | Confirm checkpoint and model size | Yes |
| 3 | Confirm data.py and evaluator.py contracts | Yes |
| 4 | Confirm task key (or accept auto-generated) | Yes |
| 5 | Run smoke test | Yes |
| 6 | Set high-level parameters (budget, strategy) | Yes |
| 7 | Enter auto-research loop | No — agent runs autonomously |

Training goal must be explicitly confirmed before entering Phase 2.

### Phase 2: Auto-Research Loop (Agent Runs Autonomously)

```
baseline → confirm → explore one variable → evaluate → decide → repeat
```

**Baseline run** — train with default LoRA config, record metrics.

**Confirm run** — re-run with identical config. Proceed only if results agree within 5%.

**Exploration order:**
1. `lora_alpha`
2. `lora_alpha` local refinement
3. `learning_rate`
4. `learning_rate` local refinement
5. `lora_r`
6. `target_modules`
7. `dropout`

After any accepted improvement, always do one confirmation run before moving to the next direction.

**Stopping conditions:**
- Budget exhausted (default: 10 exploration runs)
- Plateau: no improvement > 0.2% in last 4 successful runs
- Target metric reached (optional `--target-accuracy`)

### Phase 3: Promotion (Full-Scale Rerun)

Top-K configs from exploration are re-run on the full eval set for definitive numbers.

### Phase 4: Reporting

- `leaderboard.md` — ranked results for this task
- `results.jsonl` — one JSON line per completed run
- `runs/<timestamp>/<exp_id>/adapter/` — LoRA adapter weights
- Agent writes a summary: model, data, baseline metrics, best metrics, effective changes, checkpoint path, next recommendations

---

## LoRA Fine-Tuning Parameters

### Safe Parameter Ranges

| Parameter | Default | Safe Range | Notes |
|-----------|---------|------------|-------|
| `lora_r` | 16 | 4–64 | Higher = more capacity, more VRAM |
| `lora_alpha` | 32 | 8–128 | Effective scale = alpha / r |
| `learning_rate` | 2e-4 | 1e-5–1e-3 | Use cosine scheduler |
| `num_epochs` | 1 | 1–10 | |
| `batch_size` | 1 | 1–16 | |
| `grad_accum` | 16 | 1–64 | Effective batch = batch_size × grad_accum |
| `max_seq_length` | 512 | 128–2048 | Quadratic VRAM cost |
| `dropout` | 0.05 | 0.0–0.2 | |
| `target_modules` | `attn` | `qv`, `attn`, `all_linear` | |

### VRAM Safety Thresholds

| VRAM Utilization | Status | Action |
|-----------------|--------|--------|
| < 60% | Safe | Proceed |
| 60–80% | Caution | Monitor closely |
| > 80% | Danger | Reduce batch_size or enable gradient checkpointing |

### VRAM vs Model Size Guide

| VRAM | Recommended Size |
|------|-----------------|
| 4 GB | 0.5B–1B |
| 6 GB | 1B–1.5B |
| 8 GB | 1.5B–3B |
| 12 GB | 3B (7B requires Q-LoRA) |
| 16 GB | 3B–7B with Q-LoRA |
| 24 GB+ | 7B–13B |

Any change to `lora_r`, `target_modules`, `batch_size`, `max_seq_length`, or disabling `gradient_checkpointing` requires a mini smoke test before the full run.

---

## Decision Rules

### Improvement thresholds

| Outcome | Threshold | Status | Action |
|---------|-----------|--------|--------|
| Improvement | > 1% relative | `pending_confirm` | Run confirmation, then update best |
| Tie | 0–1% relative | `tie` | Record, do not promote; continue to next direction |
| Regression | > 1% relative | `discard` | Roll back config, record reason, continue |
| Crash | OOM / NaN / timeout | `crash` | Follow error handling procedure |

### Confirmation protocol

- Re-run with identical config and data
- Valid if result is within 5% of the original run
- If inconsistent (`unstable`): roll back, do not update best

### One-variable-at-a-time rule

Every round changes exactly **one** parameter. If two things changed, the cause of any result shift is unknown — discard and re-run with a single change.

---

## Error Handling

### OOM (Out of Memory)
1. Stop the current run immediately
2. Revert to the last safe checkpoint
3. Halve `batch_size`; if already at 1, enable `gradient_checkpointing`
4. Record `status=crash, error=oom`
5. After **3 consecutive OOM failures**, pause and notify the user

### NaN Loss
1. Stop the current run
2. Reduce `learning_rate` by 10×
3. Set `gradient_clip_val = 1.0` if not already set
4. Record `status=crash, error=nan_loss`

### Stalled Training
- If a run exceeds 4× the expected time budget: terminate the process
- Save current state and record `status=crash, error=timeout`

### User Interrupt
1. Save `resume_state.json` in the run folder
2. Save the current checkpoint
3. Print: `Run paused. Resume with: python run.py --resume`

---

## Prohibited Actions

1. Adjusting parameters before establishing a confirmed baseline
2. Accepting an improvement after a single run (always confirm)
3. Changing multiple parameters in a single round
4. Asking the user for confirmation on every training round in Phase 2
5. Ignoring VRAM safety boundaries
6. Using different eval data for baseline vs. subsequent runs
7. Writing "experiment complete" when parameter directions remain untested

---

## Task Folder Structure

```
tasks/<key>/
├── meta.json           # checkpoint, branch, source paths, created timestamp
├── data.py             # copy of the provided data module
├── evaluator.py        # copy of the provided evaluator module
├── trainer.py          # copy of auto_research/trainer.py (modifiable)
└── runs/
    └── <timestamp>/
        ├── results.jsonl
        ├── leaderboard.md
        └── <exp_id>/
            ├── config.json
            └── adapter/
```

---

## Resuming an Interrupted Task

The orchestrator reads `results.jsonl` at startup and skips any `exp_id` already present in the current run folder. Killing and restarting is safe:

```bash
python run.py --resume --key <key> --checkpoint ... --data-file ... --evaluator-file ...
```

Reusing the same key reuses the branch and folder; a new `runs/<timestamp>/` is created each invocation.
