"""
The autonomous loop: propose -> run -> evaluate -> analyze -> repeat.

Two phases:
  Phase A: exploration on tier='small'. Runs until budget is hit or plateau.
  Phase B: promotion. Take top-K configs, re-run them at tier='full' for
           definitive numbers on the full test set.

The orchestrator is task-agnostic: it dynamically imports the task's
data.py, evaluator.py, and trainer.py from a task folder. Their contracts:

    data.py
        load_train_dataset(seed: int, subset: int | None, balance: str, **kw)
        load_eval_dataset(seed: int, subset: int | None, **kw)
        get_collator(tokenizer, max_seq_len) -> collator             # optional

    evaluator.py
        evaluate(adapter_dir, model_name, eval_data, **kw) -> dict
        PRIMARY_METRIC: str                                          # optional, default "accuracy"

    trainer.py
        train_one(cfg, train_dataset, model_name, output_dir, collator=None, **kw)
          -> {"train_loss": float, "adapter_dir": str}
"""
from __future__ import annotations
import gc
import importlib.util
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch

from .schema import ExperimentConfig, ExperimentResult
from .proposer import propose_next
from .analyzer import (
    load_results, append_result, write_leaderboard,
    best_so_far, has_plateaued, top_k,
)


# --- Dynamic task-module loading --------------------------------------------

def _load_task_module(name: str, path: str | Path) -> ModuleType:
    """Import a Python file as a module, supporting relative imports in the
    file's directory (treats the parent dir as a package). The parent dir
    must contain an __init__.py for relative imports to resolve.
    """
    import sys
    p = Path(path).resolve()
    parent = p.parent
    pkg_name = f"_automle_task_{parent.name}"

    if (parent / "__init__.py").exists():
        # Register the parent dir as a package so relative imports work.
        if pkg_name not in sys.modules:
            pkg_spec = importlib.util.spec_from_file_location(
                pkg_name, parent / "__init__.py",
                submodule_search_locations=[str(parent)],
            )
            pkg = importlib.util.module_from_spec(pkg_spec)
            sys.modules[pkg_name] = pkg
            pkg_spec.loader.exec_module(pkg)
        full_name = f"{pkg_name}.{p.stem}"
        spec = importlib.util.spec_from_file_location(full_name, p)
    else:
        full_name = name
        spec = importlib.util.spec_from_file_location(full_name, p)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _maybe_get_tokenizer(model_name: str):
    """Load just the tokenizer (cheap, no GPU) so we can build the collator."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# --- One experiment ----------------------------------------------------------

def run_one_experiment(
    cfg: ExperimentConfig,
    data_mod: ModuleType,
    evaluator_mod: ModuleType,
    trainer_mod: ModuleType,
    model_name: str,
    runs_dir: Path,
    primary_metric: str,
) -> ExperimentResult:
    """Train + evaluate one config. Captures OOM and other errors gracefully."""
    cfg.apply_tier()
    run_dir = runs_dir / cfg.exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.to_json())

    started = time.time()
    try:
        train_ds = data_mod.load_train_dataset(
            seed=cfg.train.seed,
            subset=cfg.data.train_subset,
            balance=cfg.data.balance,
        )
        eval_data = data_mod.load_eval_dataset(
            seed=cfg.train.seed,
            subset=cfg.data.eval_subset,
        )

        collator = None
        if hasattr(data_mod, "get_collator"):
            tokenizer = _maybe_get_tokenizer(model_name)
            collator = data_mod.get_collator(tokenizer, cfg.train.max_seq_len)

        train_out = trainer_mod.train_one(
            cfg, train_ds, model_name, run_dir, collator=collator,
        )
        _free_gpu()

        eval_out = evaluator_mod.evaluate(
            train_out["adapter_dir"], model_name, eval_data,
            max_seq_len=cfg.train.max_seq_len,
        )
        _free_gpu()

        primary = float(eval_out.get(primary_metric, 0.0))
        n_eval = int(eval_out.get("n_eval", 0))
        extra = {k: v for k, v in eval_out.items() if k not in (primary_metric, "n_eval")}

        return ExperimentResult(
            exp_id=cfg.exp_id,
            config=cfg.to_dict(),
            accuracy=primary,
            n_eval=n_eval,
            train_loss=float(train_out["train_loss"]),
            runtime_sec=float(time.time() - started),
            status="success",
            extra=extra,
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


# --- Orchestration loop ------------------------------------------------------

def orchestrate(
    data_path: str | Path,
    evaluator_path: str | Path,
    trainer_path: str | Path,
    model_name: str,
    results_dir: str | Path,
    budget: int = 10,
    final_top_k: int = 3,
    use_llm_proposer: bool = True,
    target_accuracy: Optional[float] = None,
    skip_full_phase: bool = False,
) -> dict:
    """Run the full pipeline end-to-end against a task's plug-in modules."""
    data_mod = _load_task_module("task_data", data_path)
    evaluator_mod = _load_task_module("task_evaluator", evaluator_path)
    trainer_mod = _load_task_module("task_trainer", trainer_path)
    primary_metric = getattr(evaluator_mod, "PRIMARY_METRIC", "accuracy")

    results_dir = Path(results_dir)
    runs_dir = results_dir / "runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    results_path = results_dir / "results.jsonl"
    leaderboard_path = results_dir / "leaderboard.md"

    all_results = load_results(results_path)
    completed_ids = {r["exp_id"] for r in all_results}
    next_idx = len(all_results)

    print(f"[orchestrate] Resuming with {len(all_results)} prior results. "
          f"Budget: {budget}. Primary metric: {primary_metric}.")

    # --- Phase A: exploration ---------------------------------------------
    while next_idx < budget:
        cfg = propose_next(
            results=all_results,
            model_name=model_name,
            next_idx=next_idx,
            use_llm=use_llm_proposer,
        )
        if cfg.exp_id in completed_ids:
            cfg.exp_id = f"{cfg.exp_id}_v{next_idx}"
        completed_ids.add(cfg.exp_id)

        print(f"\n=== [{next_idx + 1}/{budget}] {cfg.exp_id} ===")
        print(f"hypothesis: {cfg.hypothesis}")

        result = run_one_experiment(
            cfg, data_mod, evaluator_mod, trainer_mod,
            model_name, runs_dir, primary_metric,
        )
        result_dict = result.to_dict()
        append_result(results_path, result_dict)
        all_results.append(result_dict)
        write_leaderboard(all_results, leaderboard_path)

        if result.status == "success":
            print(f"  -> {primary_metric} = {result.accuracy:.4f} (n={result.n_eval}, "
                  f"loss={result.train_loss:.4f}, {result.runtime_sec:.0f}s)")
        else:
            print(f"  -> FAILED: {result.status}: {result.error[:200]}")

        if target_accuracy is not None:
            best = best_so_far(all_results)
            if best is not None and best["accuracy"] >= target_accuracy:
                print(f"[orchestrate] target {primary_metric} {target_accuracy} reached — stopping exploration.")
                break
        if has_plateaued(all_results, window=4, min_delta=0.002):
            print("[orchestrate] plateau detected (no improvement >0.2% in last 4 runs) — stopping exploration.")
            break

        next_idx += 1

    # --- Phase B: promote top-K -------------------------------------------
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
        cfg.notes = f"Full-tier rerun of {r['exp_id']} (small {primary_metric}={r['accuracy']:.4f})."

        print(f"\n=== FULL: {cfg.exp_id} ===")
        result = run_one_experiment(
            cfg, data_mod, evaluator_mod, trainer_mod,
            model_name, runs_dir, primary_metric,
        )
        result_dict = result.to_dict()
        append_result(results_path, result_dict)
        all_results.append(result_dict)
        write_leaderboard(all_results, leaderboard_path)
        if result.status == "success":
            print(f"  -> FULL {primary_metric} = {result.accuracy:.4f} (n={result.n_eval})")
        else:
            print(f"  -> FAILED: {result.status}")

    summary["best"] = best_so_far(all_results)
    return summary
