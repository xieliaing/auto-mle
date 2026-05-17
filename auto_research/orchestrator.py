"""
The autonomous loop: propose -> run -> evaluate -> analyze -> repeat.

Two phases:
  Phase A: exploration on tier='small'. Runs until budget is hit or plateau.
  Phase B: promotion. Take top-K configs, re-run them at tier='full' for
           definitive numbers on the full test set.
"""
from __future__ import annotations
import gc
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from .schema import ExperimentConfig, ExperimentResult
from .data import load_csv, apply_data_strategy, ProductPairDataset
from .trainer import train_one
from .evaluator import evaluate
from .proposer import propose_next
from .analyzer import (
    load_results, append_result, write_leaderboard,
    best_so_far, has_plateaued, top_k,
)


def _free_gpu():
    """Clear GPU between experiments to avoid cumulative fragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_one_experiment(
    cfg: ExperimentConfig,
    train_csv: str,
    eval_csv: str,
    model_name: str,
    runs_dir: Path,
) -> ExperimentResult:
    """Train + evaluate one config. Captures OOM and other errors gracefully."""
    cfg.apply_tier()
    run_dir = runs_dir / cfg.exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the config alongside the run for reproducibility
    (run_dir / "config.json").write_text(cfg.to_json())

    started = time.time()
    try:
        # Build datasets
        train_df_raw = load_csv(train_csv)
        eval_df_raw = load_csv(eval_csv)
        train_df = apply_data_strategy(train_df_raw, cfg.data, cfg.train.seed, is_train=True)
        eval_df = apply_data_strategy(eval_df_raw, cfg.data, cfg.train.seed, is_train=False)

        train_ds = ProductPairDataset(
            train_df, augment=cfg.data.augment_titles, seed=cfg.train.seed,
        )

        # Train
        train_out = train_one(cfg, train_ds, model_name, run_dir)
        _free_gpu()

        # Evaluate
        eval_out = evaluate(
            eval_df,
            model_name=model_name,
            adapter_dir=train_out["adapter_dir"],
            batch_size=8,
            max_seq_len=cfg.train.max_seq_len,
        )
        _free_gpu()

        return ExperimentResult(
            exp_id=cfg.exp_id,
            config=cfg.to_dict(),
            accuracy=float(eval_out["accuracy"]),
            n_eval=int(eval_out["n_eval"]),
            train_loss=float(train_out["train_loss"]),
            runtime_sec=float(time.time() - started),
            status="success",
            extra={
                "f1": eval_out["f1"],
                "precision": eval_out["precision"],
                "recall": eval_out["recall"],
                "tp": eval_out["tp"], "tn": eval_out["tn"],
                "fp": eval_out["fp"], "fn": eval_out["fn"],
            },
        )
    except torch.cuda.OutOfMemoryError as e:
        _free_gpu()
        return ExperimentResult(
            exp_id=cfg.exp_id, config=cfg.to_dict(),
            accuracy=0.0, n_eval=0, train_loss=float("nan"),
            runtime_sec=float(time.time() - started),
            status="failed_oom", error=str(e)[:500],
        )
    except Exception as e:
        _free_gpu()
        return ExperimentResult(
            exp_id=cfg.exp_id, config=cfg.to_dict(),
            accuracy=0.0, n_eval=0, train_loss=float("nan"),
            runtime_sec=float(time.time() - started),
            status="failed_other", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[:1500]}",
        )


def orchestrate(
    train_csv: str,
    eval_csv: str,
    model_name: str,
    results_dir: str | Path,
    budget: int = 10,
    final_top_k: int = 3,
    use_llm_proposer: bool = True,
    target_accuracy: Optional[float] = None,
    skip_full_phase: bool = False,
) -> dict:
    """
    Run the full pipeline end-to-end.

    Returns a summary dict with the path to the leaderboard and the best result.
    """
    results_dir = Path(results_dir)
    runs_dir = results_dir / "runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    results_path = results_dir / "results.jsonl"
    leaderboard_path = results_dir / "leaderboard.md"

    # Resume: load any prior results so we don't duplicate work
    all_results = load_results(results_path)
    completed_ids = {r["exp_id"] for r in all_results}
    next_idx = len(all_results)

    print(f"[orchestrate] Resuming with {len(all_results)} prior results. Budget: {budget}.")

    # --- Phase A: exploration at tier='small' ----------------------------------
    while next_idx < budget:
        cfg = propose_next(
            results=all_results,
            model_name=model_name,
            next_idx=next_idx,
            use_llm=use_llm_proposer,
        )
        # Defensive: ensure unique id
        if cfg.exp_id in completed_ids:
            cfg.exp_id = f"{cfg.exp_id}_v{next_idx}"
        completed_ids.add(cfg.exp_id)

        print(f"\n=== [{next_idx + 1}/{budget}] {cfg.exp_id} ===")
        print(f"hypothesis: {cfg.hypothesis}")

        result = run_one_experiment(cfg, train_csv, eval_csv, model_name, runs_dir)
        result_dict = result.to_dict()
        append_result(results_path, result_dict)
        all_results.append(result_dict)
        write_leaderboard(all_results, leaderboard_path)

        if result.status == "success":
            print(f"  -> accuracy = {result.accuracy:.4f} (n={result.n_eval}, "
                  f"loss={result.train_loss:.4f}, {result.runtime_sec:.0f}s)")
        else:
            print(f"  -> FAILED: {result.status}: {result.error[:200]}")

        # Early stop conditions
        if target_accuracy is not None:
            best = best_so_far(all_results)
            if best is not None and best["accuracy"] >= target_accuracy:
                print(f"[orchestrate] target accuracy {target_accuracy} reached — stopping exploration.")
                break
        if has_plateaued(all_results, window=4, min_delta=0.002):
            print("[orchestrate] plateau detected (no improvement >0.2% in last 4 runs) — stopping exploration.")
            break

        next_idx += 1

    # --- Phase B: promote top-K to tier='full' ---------------------------------
    summary = {
        "leaderboard": str(leaderboard_path),
        "results": str(results_path),
        "best": best_so_far(all_results),
    }
    if skip_full_phase:
        return summary

    leaders = top_k(all_results, k=final_top_k)
    if not leaders:
        print("[orchestrate] no successful runs — skipping full-tier phase.")
        return summary

    print(f"\n=== Phase B: promoting top {len(leaders)} to tier='full' ===")
    for r in leaders:
        cfg = ExperimentConfig.from_dict(r["config"])
        cfg.tier = "full"
        cfg.exp_id = f"{cfg.exp_id}_FULL"
        if cfg.exp_id in completed_ids:
            print(f"[orchestrate] {cfg.exp_id} already exists, skipping.")
            continue
        cfg.notes = f"Full-tier rerun of {r['exp_id']} (small acc={r['accuracy']:.4f})."

        print(f"\n=== FULL: {cfg.exp_id} ===")
        result = run_one_experiment(cfg, train_csv, eval_csv, model_name, runs_dir)
        result_dict = result.to_dict()
        append_result(results_path, result_dict)
        all_results.append(result_dict)
        write_leaderboard(all_results, leaderboard_path)
        if result.status == "success":
            print(f"  -> FULL accuracy = {result.accuracy:.4f} (n={result.n_eval})")
        else:
            print(f"  -> FAILED: {result.status}")

    summary["best"] = best_so_far(all_results)
    return summary
