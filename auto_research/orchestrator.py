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
        load_public_dataset(**kw) -> eval_data | None                # optional, see "Public/private protocol"

    evaluator.py
        evaluate(adapter_dir, model_name, eval_data, **kw) -> dict
        PRIMARY_METRIC: str                                          # optional, default "accuracy"
        With return_records=True, may add "records" (per-example defects;
        see defect_analysis.py).

    trainer.py
        train_one(cfg, train_dataset, model_name, output_dir, collator=None, **kw)
          -> {"train_loss": float, "adapter_dir": str}

Public/private protocol (when data.load_public_dataset returns a set):
  - public set: scored every run; ranks runs (leaderboard, plateau, top-K)
    and its defects feed the LLM analysis.
  - eval set (private): scored only on Phase-B promoted runs, reported as
    extra["private_score"]. Never used for selection; examples never sent.
Without a public set, the eval set ranks runs and defect analysis is off.
"""
from __future__ import annotations
import gc
import importlib.util
import json
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
from .defect_analysis import analyze_and_plan, plan_context

PUBLIC_RECORDS_FILE = "public_records.jsonl"


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
    public_data=None,
    collect_records: bool = False,
    score_private: bool = False,
) -> ExperimentResult:
    """Train + evaluate one config. Captures OOM and other errors gracefully.

    Without public_data, the eval set is scored as the run's metric (classic).
    With public_data, the public set is the run's metric (and, if
    collect_records, its per-example records go to <run_dir>/public_records.jsonl);
    the eval set is private and only scored when score_private is set.
    """
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
        eval_data = None
        if public_data is None or score_private:
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
            train_out["adapter_dir"], model_name,
            eval_data if public_data is None else public_data,
            max_seq_len=cfg.train.max_seq_len,
            **({"return_records": True} if collect_records else {}),
        )
        _free_gpu()
        records = eval_out.pop("records", None)

        primary = float(eval_out.get(primary_metric, 0.0))
        n_eval = int(eval_out.get("n_eval", 0))
        extra = {k: v for k, v in eval_out.items() if k not in (primary_metric, "n_eval")}

        if public_data is not None and score_private:
            private_out = evaluator_mod.evaluate(
                train_out["adapter_dir"], model_name, eval_data,
                max_seq_len=cfg.train.max_seq_len,
            )
            _free_gpu()
            extra["private_score"] = float(private_out.get(primary_metric, 0.0))
            extra["private_n_eval"] = int(private_out.get("n_eval", 0))

        if collect_records:
            if records is None:
                print("  [defects] evaluator returned no 'records'; skipping defect analysis.")
            else:
                with open(run_dir / PUBLIC_RECORDS_FILE, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, default=str) + "\n")

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


def _defect_analysis(run_dir: Path, result_dict: dict, history: list,
                     model_name: str, primary_metric: str, **llm_kw) -> Optional[dict]:
    """Run public-set defect analysis for one finished run. Never raises."""
    rec_path = run_dir / PUBLIC_RECORDS_FILE
    if result_dict["status"] != "success" or not rec_path.exists():
        return None
    try:
        with open(rec_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return analyze_and_plan(
            run_dir, result_dict, records, history,
            model_name=model_name, primary_metric=primary_metric, **llm_kw,
        )
    except Exception as e:
        print(f"  [defects] analysis failed: {type(e).__name__}: {e}")
        return None


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
    defect_analysis: bool = True,
    analysis_provider: str = "anthropic",
    analysis_model: Optional[str] = None,
    analysis_max_examples: int = 50,
    analysis_base_url: Optional[str] = None,
    proposer_provider: str = "anthropic",
    hardware: str = "",
) -> dict:
    """Run the full pipeline end-to-end against a task's plug-in modules."""
    data_mod = _load_task_module("task_data", data_path)
    evaluator_mod = _load_task_module("task_evaluator", evaluator_path)
    trainer_mod = _load_task_module("task_trainer", trainer_path)
    primary_metric = getattr(evaluator_mod, "PRIMARY_METRIC", "accuracy")

    public_data = None
    if hasattr(data_mod, "load_public_dataset"):
        public_data = data_mod.load_public_dataset()
    if public_data is None:
        print("[orchestrate] No public set (data.load_public_dataset missing or returned None): "
              "the eval set ranks runs; defect analysis is disabled.")
        defect_analysis = False
    else:
        print("[orchestrate] Public/private protocol: public set ranks runs and drives defect analysis; "
              "eval set is private, scored only on Phase-B promoted runs.")
    llm_kw = dict(provider=analysis_provider, model=analysis_model,
                  max_examples=analysis_max_examples, base_url=analysis_base_url,
                  hardware=hardware)
    eval_kw = dict(public_data=public_data, collect_records=defect_analysis)

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
    plan = None  # latest defect-analysis plan; its next_config overrides the proposer
    while next_idx < budget:
        cfg = None
        if plan and plan.get("next_config"):
            try:
                nc = dict(plan["next_config"])
                nc.setdefault("exp_id", f"plan_{next_idx:03d}")
                cfg = ExperimentConfig.from_dict(nc)
                print("[orchestrate] using next_config from defect-analysis plan")
            except Exception as e:
                print(f"[orchestrate] plan next_config invalid ({type(e).__name__}: {e}); using proposer")
        if cfg is None:
            cfg = propose_next(
                results=all_results,
                model_name=model_name,
                next_idx=next_idx,
                use_llm=use_llm_proposer,
                skill_context=plan_context(plan) if plan else "",
                provider=proposer_provider,
                # A local proposer shares the analysis endpoint and model.
                llm_model=analysis_model if proposer_provider == "local" else None,
                base_url=analysis_base_url,
            )
        if cfg.exp_id in completed_ids:
            cfg.exp_id = f"{cfg.exp_id}_v{next_idx}"
        completed_ids.add(cfg.exp_id)

        print(f"\n=== [{next_idx + 1}/{budget}] {cfg.exp_id} ===")
        print(f"hypothesis: {cfg.hypothesis}")

        result = run_one_experiment(
            cfg, data_mod, evaluator_mod, trainer_mod,
            model_name, runs_dir, primary_metric, **eval_kw,
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

        if defect_analysis:
            plan = _defect_analysis(runs_dir / cfg.exp_id, result_dict, all_results,
                                    model_name, primary_metric, **llm_kw)

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
            model_name, runs_dir, primary_metric, **eval_kw,
            score_private=public_data is not None,
        )
        result_dict = result.to_dict()
        append_result(results_path, result_dict)
        all_results.append(result_dict)
        write_leaderboard(all_results, leaderboard_path)
        if result.status == "success":
            print(f"  -> FULL {primary_metric} = {result.accuracy:.4f} (n={result.n_eval})")
            if "private_score" in result.extra:
                print(f"  -> PRIVATE {primary_metric} = {result.extra['private_score']:.4f} "
                      f"(n={result.extra['private_n_eval']})")
        else:
            print(f"  -> FAILED: {result.status}")
        # Report only: no further runs consume the plan after Phase B.
        if defect_analysis:
            _defect_analysis(runs_dir / cfg.exp_id, result_dict, all_results,
                             model_name, primary_metric, **llm_kw)

    summary["best"] = best_so_far(all_results)
    # Final private numbers for the promoted runs (public/private protocol only).
    summary["private"] = [
        {"exp_id": r["exp_id"], "public": r["accuracy"], "private": r["extra"]["private_score"]}
        for r in all_results if "private_score" in r.get("extra", {})
    ]
    return summary
