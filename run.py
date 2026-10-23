"""
AutoMLE — parent base entry point.

See program.md for full usage instructions.

Usage:
    python run.py \\
        --key my-task \\
        --checkpoint Qwen/Qwen3-1.7B \\
        --data-file path/to/data.py \\
        --evaluator-file path/to/evaluator.py

Inputs:
    --checkpoint        Base model: HuggingFace model ID or local checkpoint path.
    --data-file         Python module exposing load_train_dataset / load_eval_dataset.
    --evaluator-file    Python module exposing evaluate(adapter_dir, model_name, eval_data).
    --key               Task key (omit to auto-generate; sanitized to [a-z0-9_-]).
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
    p.add_argument("--data-file", default=None,
                   help="Python module: data.py exposing load_train_dataset / load_eval_dataset.")
    p.add_argument("--evaluator-file", default=None,
                   help="Python module: evaluator.py exposing evaluate(adapter_dir, model_name, eval_data).")

    # --- Task identity --------------------------------------------------------
    p.add_argument("--key", default=None,
                   help="Task key (used for branch + folder). Auto-generated if omitted.")

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

    # --- Golden-set defect analysis ------------------------------------------
    p.add_argument("--no-defect-analysis", action="store_true",
                   help="Disable post-training public-set defect analysis + LLM improvement plan.")
    p.add_argument("--analysis-provider", choices=["anthropic", "openai"], default=None,
                   help="LLM provider for defect analysis (default: anthropic).")
    p.add_argument("--analysis-model", default=None,
                   help="Override the analysis model (default: provider's default).")

    # --- Branch management ---------------------------------------------------
    p.add_argument("--no-branch", action="store_true",
                   help="Skip git branch creation (run in current branch).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from a previous run (the orchestrator skips completed exp_ids).")

    return p.parse_args(argv)


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv=None):
    args = parse_args(argv)

    cfg_path = Path(args.config)
    defaults = _load_yaml(cfg_path) if cfg_path.exists() else {}

    def pick(arg_val, key, fallback=None):
        if arg_val is not None:
            return arg_val
        return defaults.get(key, fallback)

    checkpoint     = pick(args.checkpoint,     "checkpoint",     None)
    data_file      = pick(args.data_file,      "data_file",      None)
    evaluator_file = pick(args.evaluator_file, "evaluator_file", None)

    task_key    = args.key or defaults.get("key")
    budget      = pick(args.budget,           "budget",      10)
    final_top_k = pick(args.final_top_k,      "final_top_k", 3)
    target_acc  = pick(args.target_accuracy,  "target_accuracy", None)

    defect_analysis   = not args.no_defect_analysis and defaults.get("defect_analysis", True)
    analysis_provider = pick(args.analysis_provider, "analysis_provider", "anthropic")
    analysis_model    = pick(args.analysis_model,    "analysis_model",    None)
    analysis_max_ex   = defaults.get("analysis_max_examples", 50)

    # --- Validate required inputs --------------------------------------------
    missing = [flag for flag, val in [
        ("--checkpoint",     checkpoint),
        ("--data-file",      data_file),
        ("--evaluator-file", evaluator_file),
    ] if val is None]
    if missing:
        print(f"[error] Required arguments missing: {', '.join(missing)}", file=sys.stderr)
        print("        See program.md for usage instructions.", file=sys.stderr)
        sys.exit(1)

    for flag, path in [("--data-file", data_file), ("--evaluator-file", evaluator_file)]:
        if not Path(path).exists():
            print(f"[error] File not found for {flag}: {path}", file=sys.stderr)
            sys.exit(2)

    use_llm = not args.no_llm_proposer
    if use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        print("[warn] ANTHROPIC_API_KEY not set; LLM proposer will fall back to heuristics.")
        print("       Pass --no-llm-proposer to silence this warning.")

    key_env = "ANTHROPIC_API_KEY" if analysis_provider == "anthropic" else "OPENAI_API_KEY"
    if defect_analysis and not os.environ.get(key_env):
        print(f"[warn] {key_env} not set; defect analysis will save prompts but skip the LLM call.")
        print("       Pass --no-defect-analysis to disable it.")

    # --- Create / resume task folder + branch --------------------------------
    if not args.no_branch:
        from experiment_manager import setup_task
        task = setup_task(
            key=task_key,
            data_file=data_file,
            evaluator_file=evaluator_file,
            checkpoint=checkpoint,
        )
        data_file      = task["data_file"]
        evaluator_file = task["evaluator_file"]
        trainer_file   = task["trainer_file"]
        run_folder     = Path(task["run_folder"])
        print(f"[task] Key      : {task['key']}")
        print(f"[task] Branch   : {task['branch']}")
        print(f"[task] Folder   : {task['task_folder']}")
        print(f"[task] Run      : {run_folder}")
    else:
        # Headless mode: use a sibling 'trainer.py' next to the data/evaluator
        # files (falls back to auto_research/trainer.py).
        data_dir = Path(data_file).resolve().parent
        candidate = data_dir / "trainer.py"
        trainer_file = str(candidate if candidate.exists()
                            else Path("auto_research") / "trainer.py")
        run_folder = Path("runs") / (task_key or "adhoc")
        run_folder.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint     : {checkpoint}")
    print(f"data file      : {data_file}")
    print(f"evaluator file : {evaluator_file}")
    print(f"trainer file   : {trainer_file}")
    print(f"results dir    : {run_folder}")
    print(f"budget         : {budget} exploration runs, top-{final_top_k} full reruns")

    try:
        from auto_research.orchestrator import orchestrate
    except ImportError as e:
        print(f"[error] Missing dependency: {e}", file=sys.stderr)
        print("        Install with: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(3)

    summary = orchestrate(
        data_path=data_file,
        evaluator_path=evaluator_file,
        trainer_path=trainer_file,
        model_name=checkpoint,
        results_dir=str(run_folder),
        budget=budget,
        final_top_k=final_top_k,
        use_llm_proposer=use_llm,
        target_accuracy=target_acc,
        skip_full_phase=args.skip_full_phase,
        defect_analysis=defect_analysis,
        analysis_provider=analysis_provider,
        analysis_model=analysis_model,
        analysis_max_examples=analysis_max_ex,
    )

    print("\n=== DONE ===")
    print(f"Leaderboard : {summary['leaderboard']}")
    print(f"All results : {summary['results']}")
    if summary.get("best"):
        b = summary["best"]
        print(f"Best        : {b['exp_id']}  metric={b['accuracy']:.4f}")
    for p in summary.get("private", []):
        print(f"Private     : {p['exp_id']}  public={p['public']:.4f}  private={p['private']:.4f}")


if __name__ == "__main__":
    main()
