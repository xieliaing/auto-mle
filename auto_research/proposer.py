"""
The proposer agent.

Reads the history of past results and proposes the next ExperimentConfig.
Uses two strategies:

  1. SEED PLAN (rounds 0..N_SEED-1): a fixed sequence of experiments that
     covers the obvious axes: baseline, bigger LoRA, focal loss, label
     smoothing, hard negatives, balanced data. This guarantees coverage even
     if the LLM is unavailable, and gives the LLM strong signal for later
     rounds.

  2. LLM-DRIVEN (rounds N_SEED+): asks Claude to look at all prior runs and
     return a JSON config + hypothesis. Constrained by the schema and a
     prompt that explicitly asks for *meaningfully different* experiments.

The LLM call uses the standard Anthropic SDK. ANTHROPIC_API_KEY must be set.
If the call fails, we fall back to a small heuristic perturbation of the
best-so-far config.
"""
from __future__ import annotations
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .schema import (
    ExperimentConfig, LoraConfigSpec, TrainConfigSpec, DataConfigSpec, LossConfigSpec,
)


PROPOSER_MODEL = os.environ.get("PROPOSER_MODEL", "claude-sonnet-4-5")
PROPOSER_MAX_TOKENS = 1500


# --- Seed plan ---------------------------------------------------------------

def seed_plan() -> List[ExperimentConfig]:
    """A fixed exploration plan that covers the core tunable axes."""
    plan = [
        ExperimentConfig(
            exp_id="seed_01_baseline",
            hypothesis="Baseline: small LoRA on attention layers with standard cross-entropy.",
            tier="small",
            lora=LoraConfigSpec(r=8, alpha=16, target_modules="attn"),
            train=TrainConfigSpec(learning_rate=2e-4, num_epochs=1.0),
            data=DataConfigSpec(),
            loss=LossConfigSpec(name="ce"),
        ),
        ExperimentConfig(
            exp_id="seed_02_bigger_lora",
            hypothesis="Larger LoRA (r=32) on all linear projections gives more capacity.",
            tier="small",
            lora=LoraConfigSpec(r=32, alpha=64, target_modules="all_linear"),
            train=TrainConfigSpec(learning_rate=1e-4, num_epochs=1.0),
            data=DataConfigSpec(),
            loss=LossConfigSpec(name="ce"),
        ),
        ExperimentConfig(
            exp_id="seed_03_focal",
            hypothesis="Focal loss (gamma=2) down-weights easy examples and focuses learning on hard ones.",
            tier="small",
            lora=LoraConfigSpec(r=16, alpha=32, target_modules="attn"),
            train=TrainConfigSpec(learning_rate=2e-4, num_epochs=1.0),
            data=DataConfigSpec(),
            loss=LossConfigSpec(name="focal", focal_gamma=2.0),
        ),
        ExperimentConfig(
            exp_id="seed_04_label_smooth",
            hypothesis="Label smoothing prevents over-confident predictions; small but consistent gains are common.",
            tier="small",
            lora=LoraConfigSpec(r=16, alpha=32, target_modules="attn"),
            train=TrainConfigSpec(learning_rate=2e-4, num_epochs=1.0),
            data=DataConfigSpec(),
            loss=LossConfigSpec(name="label_smoothing", label_smoothing=0.1),
        ),
        ExperimentConfig(
            exp_id="seed_05_lower_lr",
            hypothesis="A lower learning rate (5e-5) with the bigger LoRA can yield a more stable optimum.",
            tier="small",
            lora=LoraConfigSpec(r=32, alpha=64, target_modules="all_linear"),
            train=TrainConfigSpec(learning_rate=5e-5, num_epochs=1.0),
            data=DataConfigSpec(),
            loss=LossConfigSpec(name="ce"),
        ),
        ExperimentConfig(
            exp_id="seed_06_balanced",
            hypothesis="If the dataset is class-imbalanced, downsampling the majority class yields a better-calibrated model.",
            tier="small",
            lora=LoraConfigSpec(r=16, alpha=32, target_modules="attn"),
            train=TrainConfigSpec(learning_rate=2e-4, num_epochs=1.0),
            data=DataConfigSpec(balance="downsample"),
            loss=LossConfigSpec(name="ce"),
        ),
    ]
    return plan


# --- LLM-driven proposer -----------------------------------------------------

PROPOSER_SYSTEM = """You are an ML research advisor for a LoRA fine-tuning project.

Your job: given the history of completed experiments, propose ONE next experiment that is most likely to improve the primary metric. You must balance exploration and exploitation: don't just tweak the current best by epsilon, but also don't propose something wildly disconnected from what the data has shown.

You will respond with ONLY a single JSON object matching the ExperimentConfig schema. No markdown, no commentary, no code fences."""


PROPOSER_USER_TEMPLATE = """## Task
LoRA fine-tuning of base model: {model_name} (4-bit QLoRA). The task's data loader and evaluator are supplied by the user; treat them as a black box and tune the LoRA/training/loss axes below.
{memory_context_section}

## Schema (return JSON matching this shape)
{{
  "exp_id": "string, unique, snake_case",
  "hypothesis": "one sentence: why this experiment might beat the current best",
  "tier": "small | medium | full",
  "lora": {{
    "r": int (4..64),
    "alpha": int (typically 2*r),
    "dropout": float (0..0.2),
    "target_modules": "qv | attn | all_linear"
  }},
  "train": {{
    "learning_rate": float (1e-5 .. 5e-4),
    "num_epochs": float (0.5..3),
    "batch_size": 1,
    "grad_accum": int (4..32),
    "warmup_ratio": float (0..0.1),
    "lr_scheduler": "cosine | linear | constant",
    "weight_decay": float (0..0.1),
    "max_seq_len": int (256..1024),
    "seed": 42
  }},
  "data": {{
    "train_subset": int or null,
    "eval_subset": int or null,
    "balance": "none | downsample | upsample"
  }},
  "loss": {{
    "name": "ce | label_smoothing | focal | weighted_ce",
    "label_smoothing": float (0..0.2),
    "focal_gamma": float (0.5..5),
    "pos_weight": float (0.5..3)
  }},
  "notes": "one to two sentences for the leaderboard"
}}

## Rules
- Use tier="small" unless you've seen at least 3 results AND have strong evidence to promote a config to "full".
- If you propose tier="full", set train_subset=null and eval_subset=null.
- The new exp_id must be different from every previous exp_id.
- Don't repeat a configuration that's already been tried.
- Prefer ONE concrete change vs the current best (so we can attribute improvement). Combining changes is allowed but state why.

## History (most recent first, 'accuracy' is the run's primary metric)
{history}

## Best so far
{best}

Now output the JSON for the next experiment."""


def _format_history(results: List[dict]) -> str:
    if not results:
        return "(none yet — propose a sensible baseline)"
    lines = []
    for r in reversed(results):  # most recent first
        lines.append(json.dumps({
            "exp_id": r["exp_id"],
            "accuracy": round(r["accuracy"], 4),
            "n_eval": r.get("n_eval"),
            "status": r.get("status", "success"),
            "config": r["config"],
        }, indent=None))
    return "\n".join(lines)


def _format_best(results: List[dict]) -> str:
    ok = [r for r in results if r.get("status", "success") == "success"]
    if not ok:
        return "(none)"
    best = max(ok, key=lambda r: r["accuracy"])
    return json.dumps({
        "exp_id": best["exp_id"],
        "accuracy": round(best["accuracy"], 4),
        "config": best["config"],
    }, indent=2)


def _heuristic_fallback(results: List[dict], next_idx: int) -> ExperimentConfig:
    """If the LLM is unreachable, perturb the best-so-far config slightly."""
    ok = [r for r in results if r.get("status", "success") == "success"]
    if not ok:
        return seed_plan()[0]
    best = max(ok, key=lambda r: r["accuracy"])
    cfg = ExperimentConfig.from_dict(best["config"])
    # Bump LoRA rank and learning rate slightly; flip loss if not yet focal
    cfg.exp_id = f"heuristic_{next_idx:03d}"
    cfg.hypothesis = "Heuristic fallback: small perturbation around best-so-far."
    cfg.lora.r = min(64, cfg.lora.r * 2)
    cfg.lora.alpha = cfg.lora.r * 2
    if cfg.loss.name == "ce":
        cfg.loss = LossConfigSpec(name="focal", focal_gamma=2.0)
    cfg.notes = "Auto-generated fallback when proposer LLM was unavailable."
    return cfg


def propose_next(
    results: List[dict],
    model_name: str,
    next_idx: int,
    use_llm: bool = True,
    memory_context: str = "",
) -> ExperimentConfig:
    """
    Propose the next experiment config.

    `results` is a list of dicts loaded from results.jsonl. `next_idx` is just
    used for fallback exp_id naming. `model_name` is the base LLM being fine-
    tuned (provided so the proposer's hypothesis can reference it).
    `memory_context` is an optional string from ExperimentMemory summarizing
    per-sample defect patterns observed across prior iterations.
    """
    # Phase 1: seed plan
    plan = seed_plan()
    if next_idx < len(plan):
        return plan[next_idx]

    # Phase 2: LLM-driven
    if use_llm:
        try:
            return _propose_via_llm(results, model_name, memory_context)
        except Exception as e:
            print(f"[proposer] LLM proposal failed: {type(e).__name__}: {e}")
            print("[proposer] falling back to heuristic")
    return _heuristic_fallback(results, next_idx)


def _propose_via_llm(
    results: List[dict], model_name: str, memory_context: str = ""
) -> ExperimentConfig:
    """Call Claude to propose the next config. Raises on any failure."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed; pip install anthropic") from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    memory_section = f"\n{memory_context}\n" if memory_context else ""
    user_msg = PROPOSER_USER_TEMPLATE.format(
        model_name=model_name,
        memory_context_section=memory_section,
        history=_format_history(results),
        best=_format_best(results),
    )

    resp = client.messages.create(
        model=PROPOSER_MODEL,
        max_tokens=PROPOSER_MAX_TOKENS,
        system=PROPOSER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    # Extract text from response content blocks (handle both text and thinking)
    text_chunks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    raw = "\n".join(text_chunks).strip()

    # Strip code fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        # Drop a leading "json\n" if present
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    # Find the outermost JSON object (defensive against any preamble)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON in proposer response: {raw[:200]!r}")
    cfg_dict = json.loads(raw[start:end + 1])

    cfg = ExperimentConfig.from_dict(cfg_dict)
    return cfg
