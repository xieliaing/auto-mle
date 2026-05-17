"""
AutoMLE — parent base entry point.

See program.md for full usage instructions.

Usage:
    python run.py \\
        --experiment-name my-exp \\
        --checkpoint Qwen/Qwen3-1.7B \\
        --eval-file data/eval.csv \\
        --train-file custom_train.py

Inputs:
    --checkpoint   Base model: HuggingFace model ID or local checkpoint path.
    --eval-file    Evaluation dataset (CSV/JSONL) or Python eval script.
    --train-file   Python training script implementing the fine-tuning logic.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="AutoMLE — LoRA fine-tuning experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See program.md for the full agent guide and file interfaces.",
    )
    p.add_argument("--config", default="configs/default.yaml",
                   help="YAML file with default settings (default: configs/default.yaml).")

    # --- Required experiment inputs -------------------------------------------
    p.add_argument("--checkpoint", default=None,
                   help="Base model: HuggingFace model ID or local checkpoint path.")
    p.add_argument("--eval-file", default=None,
                   help="Evaluation file: CSV/JSONL dataset or Python eval script.")
    p.add_argument("--train-file", default=None,
                   help="Training Python file implementing the fine-tuning logic.")

    # --- Experiment identity --------------------------------------------------
    p.add_argument("--experiment-name", default=None,
                   help="Name for this experiment (used for branch and folder naming).")

    # --- Auto-research loop controls -----------------------------------------
    p.add_argument("--budget", type=int, default=None,
                   help="Max exploration runs before promotion phase (default: 10).")
    p.add_argument("--final-top-k", type=int, default=None,
                   help="Top-K configs to re-run at full scale (default: 3).")
    p.add_argument("--target-accuracy", type=float, default=None,
                   help="Stop exploration early when this metric is reached.")
    p.add_argument("--no-llm-proposer", action="store_true",
                   help="Use heuristic-only proposer (no Anthropic API calls).")
    p.add_argument("--skip-full-phase", action="store_true",
                   help="Skip the final full-tier promotion runs.")

    # --- Branch management ---------------------------------------------------
    p.add_argument("--no-branch", action="store_true",
                   help="Skip git branch creation (run in current branch).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from a previous run (the orchestrator skips completed exp_ids).")

    return p.parse_args(argv)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main(argv=None):
    args = parse_args(argv)

    cfg_path = Path(args.config)
    defaults = _load_yaml(cfg_path) if cfg_path.exists() else {}

    def pick(arg_val, key, fallback=None):
        if arg_val is not None:
            return arg_val
        return defaults.get(key, fallback)

    # Resolve the three required inputs
    checkpoint  = pick(args.checkpoint,  "checkpoint",  None)
    eval_file   = pick(args.eval_file,   "eval_file",   None)
    train_file  = pick(args.train_file,  "train_file",  None)

    experiment_name = args.experiment_name or defaults.get("experiment_name", "experiment")
    budget       = pick(args.budget,         "budget",       10)
    final_top_k  = pick(args.final_top_k,    "final_top_k",  3)
    target_acc   = pick(args.target_accuracy, "target_accuracy", None)

    # --- Validate required inputs --------------------------------------------
    missing = [flag for flag, val in [
        ("--checkpoint", checkpoint),
        ("--eval-file",  eval_file),
        ("--train-file", train_file),
    ] if val is None]

    if missing:
        print(f"[error] Required arguments missing: {', '.join(missing)}", file=sys.stderr)
        print("        See program.md for usage instructions.", file=sys.stderr)
        sys.exit(1)

    for flag, path in [("--eval-file", eval_file), ("--train-file", train_file)]:
        if not Path(path).exists():
            print(f"[error] File not found for {flag}: {path}", file=sys.stderr)
            sys.exit(2)

    # --- Warn if LLM proposer key is missing ---------------------------------
    use_llm = not args.no_llm_proposer
    if use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        print("[warn] ANTHROPIC_API_KEY not set; LLM proposer will fall back to heuristics.")
        print("       Pass --no-llm-proposer to silence this warning.")

    # --- Create experiment branch + folder -----------------------------------
    if not args.no_branch:
        try:
            from experiment_manager import create_experiment
            experiment = create_experiment(
                experiment_name=experiment_name,
                checkpoint=checkpoint,
                eval_file=eval_file,
                train_file=train_file,
            )
            exp_folder  = Path(experiment["folder"])
            eval_file   = experiment["eval_file"]
            train_file  = experiment["train_file"]
            print(f"[experiment] Branch : {experiment['branch']}")
            print(f"[experiment] Folder : {exp_folder}")
        except Exception as e:
            print(f"[warn] Branch creation failed ({e}); continuing without branch management.")
            exp_folder = Path("experiments") / experiment_name
            exp_folder.mkdir(parents=True, exist_ok=True)
    else:
        exp_folder = Path("experiments") / experiment_name
        exp_folder.mkdir(parents=True, exist_ok=True)

    results_dir = exp_folder / "results"

    # --- Print run summary ---------------------------------------------------
    print(f"checkpoint  : {checkpoint}")
    print(f"eval file   : {eval_file}")
    print(f"train file  : {train_file}")
    print(f"results dir : {results_dir}")
    print(f"budget      : {budget} exploration runs, top-{final_top_k} full reruns")

    # --- Import and run the orchestrator ------------------------------------
    try:
        from auto_research.orchestrator import orchestrate
    except ImportError as e:
        print(f"[error] Missing dependency: {e}", file=sys.stderr)
        print("        Install with: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(3)

    summary = orchestrate(
        train_csv=eval_file,     # eval file is the primary data source
        eval_csv=eval_file,
        model_name=checkpoint,
        results_dir=str(results_dir),
        budget=budget,
        final_top_k=final_top_k,
        use_llm_proposer=use_llm,
        target_accuracy=target_acc,
        skip_full_phase=args.skip_full_phase,
    )

    print("\n=== DONE ===")
    print(f"Leaderboard : {summary['leaderboard']}")
    print(f"All results : {summary['results']}")
    if summary.get("best"):
        b = summary["best"]
        print(f"Best        : {b['exp_id']}  accuracy={b['accuracy']:.4f}")


if __name__ == "__main__":
    main()
