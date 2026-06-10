"""
AutoResearch-LoRA driver for the dog-breed VLM task.

Mirrors the auto_research orchestrator's two-phase shape, specialized for the
Qwen3-VL LoRA path (which the text-only orchestrator can't drive):

  Phase A (exploration): run a fixed heuristic plan over the high-leverage LoRA
    / training knobs (rank, target modules, learning rate, epochs) on a small
    train subset, all sharing one fixed val split for comparability.
  Phase B (promotion): re-train the best Phase-A config on a larger train set.

No LLM proposer is used (no API key); the plan is a fixed heuristic sweep — the
same role auto_research's first seed rounds play before the proposer takes over.

Each config runs as a subprocess of vlm_lora.py; results are appended to
vlm_results.jsonl and a sorted leaderboard is regenerated in vlm_results.md.

Usage (from repo root):
    python kaggle/tasks/dog-breed-identification/vlm_autoresearch.py \
        --explore-train 600 --val 150 --promote-train 2000

    --promote-train 0   promote on ALL train data
    --promote-train -1  skip Phase B
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
SCRIPT = TASK_DIR / "vlm_lora.py"
RESULTS_JSONL = TASK_DIR / "vlm_results.jsonl"
RESULTS_MD = TASK_DIR / "vlm_results.md"
RUNS_DIR = TASK_DIR / "vlm_runs"

# Phase A exploration plan: (name, lora_r, lora_target, lr, epochs).
# Baseline (r16/attn/2e-4/2ep) is already recorded as run1, so it is not repeated.
PLAN = [
    ("e2_r32",            32, "attn",       2e-4, 2),
    ("e3_all_linear",     16, "all_linear", 2e-4, 2),
    ("e4_lr3e-4",         16, "attn",       3e-4, 2),
    ("e5_lr1e-4",         16, "attn",       1e-4, 2),
    ("e6_ep3",            16, "attn",       2e-4, 3),
    ("e7_r32_all_linear", 32, "all_linear", 2e-4, 2),
]


def run_one(name, lora_r, lora_target, lr, epochs, train_subset, val_subset, seed):
    """Run vlm_lora.py as a subprocess; return its metrics dict (or None on failure)."""
    out_dir = RUNS_DIR / name
    cmd = [
        sys.executable, str(SCRIPT),
        "--lora-r", str(lora_r), "--lora-target", lora_target,
        "--lr", str(lr), "--epochs", str(epochs),
        "--train-subset", str(train_subset), "--val-subset", str(val_subset),
        "--seed", str(seed), "--output-dir", str(out_dir),
    ]
    print(f"\n{'#' * 64}\n# {name}: r={lora_r} {lora_target} lr={lr} ep={epochs} "
          f"train={train_subset}\n{'#' * 64}", flush=True)
    proc = subprocess.run(cmd)
    mpath = out_dir / "metrics.json"
    if proc.returncode != 0 or not mpath.exists():
        print(f"[warn] {name} failed (rc={proc.returncode}); skipping.", flush=True)
        return None
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["run"] = name
    return m


def append_result(m: dict) -> None:
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(m) + "\n")


def load_results() -> list[dict]:
    if not RESULTS_JSONL.exists():
        return []
    return [json.loads(line) for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]


def regenerate_leaderboard() -> list[dict]:
    rows = sorted(load_results(), key=lambda r: r.get("val_accuracy", 0), reverse=True)
    lines = [
        "# Dog Breed Identification — VLM LoRA Results",
        "",
        "Vision-language LoRA fine-tuning of `Qwen/Qwen3-VL-2B-Instruct` on the Kaggle",
        "Dog Breed Identification task (120 breeds). Trainer: `vlm_lora.py`; automated",
        "config search: `vlm_autoresearch.py`. Random baseline = 1/120 ≈ 0.0083.",
        "",
        "## Leaderboard (sorted by val_accuracy)",
        "",
        "| run | lora | lr | epochs | train/val | train_loss | val_acc |",
        "|-----|------|----|--------|-----------|-----------|---------|",
    ]
    best = rows[0] if rows else None
    for r in rows:
        star = " **<- best**" if r is best else ""
        lines.append(
            f"| {r.get('run','?')} | r{r.get('lora_r','?')} {r.get('lora_target','?')} "
            f"| {r.get('lr','?')} | {r.get('epochs','?')} "
            f"| {r.get('train_subset','?')} / {r.get('val_subset','?')} "
            f"| {r.get('train_loss',float('nan')):.4f} "
            f"| **{r.get('val_accuracy',0):.4f}**{star} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="AutoResearch-LoRA sweep for dog-breed VLM.")
    p.add_argument("--explore-train", type=int, default=600)
    p.add_argument("--val", type=int, default=150)
    p.add_argument("--promote-train", type=int, default=2000,
                   help="Phase B train size; 0 = all data, -1 = skip promotion.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    print(f"=== Phase A: exploration ({len(PLAN)} configs, "
          f"train={args.explore_train}, val={args.val}) ===", flush=True)
    for name, r, tgt, lr, ep in PLAN:
        m = run_one(name, r, tgt, lr, ep, args.explore_train, args.val, args.seed)
        if m:
            append_result(m)
            rows = regenerate_leaderboard()
            print(f"  -> {name}: val_acc={m['val_accuracy']:.4f} "
                  f"(best so far {rows[0]['run']}={rows[0]['val_accuracy']:.4f})", flush=True)

    rows = regenerate_leaderboard()
    if not rows:
        print("[error] no results to promote.", flush=True)
        return
    best = rows[0]
    print(f"\n=== Phase A done. Best: {best['run']} "
          f"(val_acc={best['val_accuracy']:.4f}) ===", flush=True)

    if args.promote_train == -1:
        print("Skipping Phase B (--promote-train -1).", flush=True)
        return

    print(f"\n=== Phase B: promote {best['run']} on "
          f"{'ALL' if args.promote_train == 0 else args.promote_train} train images ===", flush=True)
    m = run_one(
        f"promote_{best['run']}", best["lora_r"], best["lora_target"],
        best["lr"], best["epochs"], args.promote_train, args.val, args.seed,
    )
    if m:
        append_result(m)
        rows = regenerate_leaderboard()
        print(f"\n=== FINAL best: {rows[0]['run']} "
              f"val_acc={rows[0]['val_accuracy']:.4f} ===", flush=True)


if __name__ == "__main__":
    main()
