# AutoMLE — Agent Program Guide

This is the operational guide for the AutoMLE parent base. Agents must follow this document when conducting LoRA fine-tuning experiments.

---

## Document Reading Order

When entering the project for the first time, read in this order:

1. `program.md` (this file)
2. `configs/default.yaml`
3. `auto_research/schema.py` — experiment config schema
4. `auto_research/orchestrator.py` — the auto-research loop

---

## Project Overview

AutoMLE is a parent base for autonomous LoRA fine-tuning experiments. It follows the **autoresearch** pattern: a closed loop of Propose → Train → Evaluate → Analyze → Repeat.

Each new experiment:
1. Creates a dedicated git feature branch (`feature/<name>-<timestamp>`)
2. Creates a dedicated folder (`experiments/<name>-<timestamp>/`)
3. Copies the provided files into that folder
4. Runs the auto-research LoRA fine-tuning loop

The codebase accepts exactly **three inputs** per experiment:

| Input | Flag | Description |
|-------|------|-------------|
| Base checkpoint | `--checkpoint` | HuggingFace model ID or local path |
| Evaluation file | `--eval-file` | CSV/JSONL evaluation dataset or Python eval script |
| Training file | `--train-file` | Python script implementing the fine-tuning logic |

---

## Starting a New Experiment

### Command

```bash
python run.py \
  --experiment-name <name> \
  --checkpoint <model_id_or_local_path> \
  --eval-file <path/to/eval.csv> \
  --train-file <path/to/train.py>
```

### What happens automatically
1. Git repository is initialized if not already present
2. A feature branch `feature/<name>-<timestamp>` is created
3. Folder `experiments/<name>-<timestamp>/` is created
4. Eval and train files are copied into that folder
5. `meta.json` is written with experiment metadata
6. The auto-research loop begins

### Skip branch creation (for development)
```bash
python run.py --no-branch --checkpoint ... --eval-file ... --train-file ...
```

---

## Agent Workflow

### Phase 1: Setup (User Confirms)

| Step | Action | User Confirmation Required? |
|------|--------|----------------------------|
| 1 | Verify environment (CUDA, RAM, VRAM) | Yes |
| 2 | Confirm checkpoint and model size | Yes |
| 3 | Confirm eval file format | Yes |
| 4 | Confirm training file interface | Yes |
| 5 | Run smoke test | Yes |
| 6 | Set high-level parameters (budget, strategy) | Yes |
| 7 | Enter auto-research loop | No — agent runs autonomously |

Training goal must be explicitly confirmed before entering Phase 2. Never default to a previous experiment's goal.

### Phase 2: Auto-Research Loop (Agent Runs Autonomously)

```
baseline → confirm → explore one variable → evaluate → decide → repeat
```

**Baseline run** — train with default LoRA config, record metrics.

**Confirm run** — re-run with identical config. Proceed only if results agree within 5%.

**Exploration order** (do not skip ahead):
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

- `leaderboard.md` — ranked results for this experiment
- `results.jsonl` — one JSON line per completed run
- `runs/<exp_id>/adapter/` — LoRA adapter weights
- Agent writes a summary: model, data, baseline metrics, best metrics, effective changes, checkpoint path, next recommendations

If any parameter direction was untested or skipped, the report must note this. Only write "complete" if all directions were covered.

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
1. Save `resume_state.json` in the experiment folder
2. Save the current checkpoint
3. Print: `Run paused. Resume with: python run.py --resume`

---

## Prohibited Actions

1. Adjusting parameters before establishing a confirmed baseline
2. Accepting an improvement after a single run (always confirm)
3. Changing multiple parameters in a single round
4. Asking the user for confirmation on every training round in Phase 2
5. Ignoring VRAM safety boundaries
6. Using different eval files for baseline vs. subsequent runs
7. Writing "experiment complete" when parameter directions remain untested

---

## Training File Interface

The Python file provided via `--train-file` must be callable as a subprocess:

```bash
python train.py \
  --checkpoint <model_id_or_path> \
  --data <path_to_training_data> \
  --output-dir <adapter_output_path> \
  --lora-r 16 \
  --lora-alpha 32 \
  --learning-rate 2e-4 \
  --num-epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-seq-length 512 \
  --target-modules attn \
  --dropout 0.05
```

It must print a JSON line to stdout upon completion:

```json
{"train_loss": 0.312, "adapter_dir": "experiments/my-exp/runs/run_001/adapter"}
```

Alternatively, the file may expose a `train()` function:

```python
def train(
    checkpoint: str,
    data_path: str,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 16,
    max_seq_length: int = 512,
    target_modules: str = "attn",
    dropout: float = 0.05,
    **kwargs,
) -> dict:
    """Returns {"train_loss": float, "adapter_dir": str}"""
    ...
```

---

## Evaluation File Interface

### Data file (CSV or JSONL)

Used directly by the built-in evaluator. The file is passed as-is to the evaluation step.

### Python evaluation script

Must be callable as a subprocess:

```bash
python eval.py \
  --adapter-dir <path_to_adapter> \
  --data <eval_data_path>
```

Must print a JSON line to stdout:

```json
{"accuracy": 0.874, "f1": 0.871, "n_eval": 2000}
```

Or expose an `evaluate()` function:

```python
def evaluate(adapter_dir: str, data_path: str, **kwargs) -> dict:
    """Returns {"accuracy": float, "f1": float, "n_eval": int, ...}"""
    ...
```

---

## Experiment Folder Structure

```
experiments/<name>-<timestamp>/
├── meta.json           # checkpoint, file paths, branch, created timestamp
├── train.py            # copy of the provided training file
├── eval.*              # copy of the provided evaluation file
├── results.jsonl       # one JSON line per completed run
├── leaderboard.md      # ranked results table
└── runs/
    └── <run_id>/
        ├── config.json # LoRA config used for this run
        └── adapter/    # LoRA adapter weights
```

---

## Resuming an Interrupted Experiment

The orchestrator reads `results.jsonl` at startup and skips any `exp_id` already present. Killing and restarting is safe:

```bash
python run.py --resume --experiment-name <name>
```

---

*Last updated: 2026-05-16*
