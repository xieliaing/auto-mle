"""
Kaggle experimental loop with defect pattern analysis.

Augments the auto_research loop with three per-iteration steps:
  1. Save per-example predictions to <run_dir>/predictions.csv
  2. Compare with previous iteration → DefectReport (fixed/new/persistent errors)
  3. Update learning_journal; ask skill_proposer for next directions
  4. Pass proposed skills into the LLM experiment proposer

Works for both LoRA fine-tuning (provide trainer_path + model_name) and
traditional ML (set trainer_path=None; the evaluator trains the model itself,
receiving train_data + config via **kwargs).

Evaluator contract extension (backward-compatible):
  evaluate(adapter_dir, model_name, eval_data, **kwargs) -> dict
  kwargs always contains:
    predictions_path  — path where evaluator should save predictions.csv
    train_data        — training DataFrame/Dataset (for traditional ML)
    config            — current ExperimentConfig as dict
    output_dir        — run directory path (str)
  Return dict may include:
    "predictions_df"  — pd.DataFrame(sample_id, true_label, pred_label)
                        (if returned, the loop saves it; evaluator may also save
                        directly to predictions_path)
"""
from __future__ import annotations

import gc
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from auto_research.schema import ExperimentConfig, ExperimentResult
from auto_research.proposer import propose_next
from auto_research.analyzer import (
    load_results, append_result, write_leaderboard,
    best_so_far, has_plateaued, top_k,
)
from kaggle.defect_analyzer import analyze_defects
from kaggle.learning_journal import LearningJournal
from kaggle.skill_proposer import propose_skills

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Utilities (local copies so kaggle/loop.py has no hard dep on auto_research
# internal privates)
# ---------------------------------------------------------------------------

def _load_module(name: str, path: str | Path):
    p = Path(path).resolve()
    parent = p.parent
    pkg_name = f"_kaggle_task_{parent.name}"
    if (parent / "__init__.py").exists():
        if pkg_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                pkg_name, parent / "__init__.py",
                submodule_search_locations=[str(parent)],
            )
            pkg = importlib.util.module_from_spec(spec)
            sys.modules[pkg_name] = pkg
            spec.loader.exec_module(pkg)
        full = f"{pkg_name}.{p.stem}"
        spec = importlib.util.spec_from_file_location(full, p)
    else:
        full = name
        spec = importlib.util.spec_from_file_location(full, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _free_gpu() -> None:
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _get_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def _run_one(
    cfg: ExperimentConfig,
    data_mod,
    evaluator_mod,
    trainer_mod,
    model_name: Optional[str],
    runs_dir: Path,
    primary_metric: str,
) -> tuple[ExperimentResult, Optional[Path]]:
    """
    Train + evaluate one config. Saves predictions.csv if the evaluator
    returns a 'predictions_df' or writes to the passed predictions_path.
    Returns (ExperimentResult, path-to-predictions-csv-or-None).
    """
    cfg.apply_tier()
    run_dir = runs_dir / cfg.exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    preds_path_target = run_dir / "predictions.csv"

    started = time.time()
    preds_saved: Optional[Path] = None

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

        if trainer_mod is not None and model_name:
            # LoRA fine-tuning path
            collator = None
            if hasattr(data_mod, "get_collator"):
                collator = data_mod.get_collator(_get_tokenizer(model_name), cfg.train.max_seq_len)
            train_out = trainer_mod.train_one(cfg, train_ds, model_name, run_dir, collator=collator)
            _free_gpu()
            adapter_dir = train_out["adapter_dir"]
            train_loss = float(train_out["train_loss"])
        else:
            # Traditional ML path: evaluator handles training via kwargs
            adapter_dir = str(run_dir)
            train_loss = float("nan")

        eval_out = evaluator_mod.evaluate(
            adapter_dir,
            model_name or "",
            eval_data,
            max_seq_len=cfg.train.max_seq_len,
            predictions_path=str(preds_path_target),
            train_data=train_ds,
            config=cfg.to_dict(),
            output_dir=str(run_dir),
        )
        _free_gpu()

        # Extract predictions_df if returned (not JSON-serializable; save separately)
        predictions_df = eval_out.pop("predictions_df", None)
        if isinstance(predictions_df, pd.DataFrame):
            predictions_df.to_csv(preds_path_target, index=False)
            preds_saved = preds_path_target
            print(f"  [loop] predictions -> {preds_path_target.name} ({len(predictions_df)} rows)")
        elif preds_path_target.exists():
            # Evaluator wrote it directly via predictions_path kwarg
            preds_saved = preds_path_target

        primary = float(eval_out.get(primary_metric, 0.0))
        n_eval = int(eval_out.get("n_eval", 0))
        extra = {k: v for k, v in eval_out.items() if k not in (primary_metric, "n_eval")}

        result = ExperimentResult(
            exp_id=cfg.exp_id,
            config=cfg.to_dict(),
            accuracy=primary,
            n_eval=n_eval,
            train_loss=train_loss,
            runtime_sec=float(time.time() - started),
            status="success",
            extra=extra,
        )

    except Exception as e:
        oom = _HAS_TORCH and isinstance(e, torch.cuda.OutOfMemoryError)
        _free_gpu()
        result = ExperimentResult(
            exp_id=cfg.exp_id, config=cfg.to_dict(),
            accuracy=0.0, n_eval=0, train_loss=float("nan"),
            runtime_sec=float(time.time() - started),
            status="failed_oom" if oom else "failed_other",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[:1500]}",
        )

    return result, preds_saved


def _prev_preds(all_results: list, runs_dir: Path) -> tuple[Optional[Path], str]:
    """Return (path, exp_id) of the last successful run's predictions.csv."""
    for r in reversed(all_results):
        if r.get("status") == "success":
            p = runs_dir / r["exp_id"] / "predictions.csv"
            if p.exists():
                return p, r["exp_id"]
    return None, ""


def _skills_context(skills: list[dict]) -> str:
    if not skills:
        return ""
    lines = ["## Skill proposals from defect analysis (address these in your next hypothesis)"]
    for s in skills[:5]:
        lines.append(
            f"  [{s.get('priority','?')}] {s.get('name','?')}: {s.get('rationale','')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

def orchestrate_with_analysis(
    data_path,
    evaluator_path,
    trainer_path=None,
    model_name: Optional[str] = None,
    results_dir: str | Path = "results",
    budget: int = 10,
    final_top_k: int = 3,
    use_llm_proposer: bool = True,
    target_accuracy: Optional[float] = None,
    skip_full_phase: bool = False,
    # Defect analysis options
    enable_defect_analysis: bool = True,
    task_context: str = "Kaggle competition task.",
    provider: str = "anthropic",
    ai_model: Optional[str] = None,
    ai_api_key: Optional[str] = None,
    ai_base_url: Optional[str] = None,
    # Sampling options (passed to defect_analyzer.analyze_defects)
    sample_size: int = 1000,
    sampling_strategy: str = "stratified_score_band",
    threshold: float = 0.5,
    near_delta: float = 0.15,
    mid_delta: float = 0.35,
) -> dict:
    """
    Kaggle loop: propose -> train -> evaluate -> analyze defects
    -> update journal -> propose skills -> repeat.

    trainer_path=None + model_name=None: traditional ML mode.
    Provide both for LoRA fine-tuning mode.

    Sampling controls (passed to defect_analyzer):
      sample_size       — rows to analyse per iteration (default 1000, 0=all)
      sampling_strategy — 'stratified_score_band' uses prev model's confidence
                          scores to oversample near-threshold and high-confidence
                          errors; requires 'confidence' column in predictions.csv
      threshold         — decision boundary (default 0.5 for binary classification)
      near_delta        — near-threshold band half-width (default 0.15)
      mid_delta         — mid-confidence band half-width (default 0.35)
    """
    data_mod = _load_module("task_data", data_path)
    evaluator_mod = _load_module("task_evaluator", evaluator_path)
    trainer_mod = _load_module("task_trainer", trainer_path) if trainer_path else None
    primary_metric = getattr(evaluator_mod, "PRIMARY_METRIC", "accuracy")

    results_dir = Path(results_dir)
    runs_dir = results_dir / "runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    results_path = results_dir / "results.jsonl"
    leaderboard_path = results_dir / "leaderboard.md"
    journal = LearningJournal(results_dir / "learning_journal.jsonl")

    all_results = load_results(results_path)
    completed = {r["exp_id"] for r in all_results}
    next_idx = len(all_results)
    learned_skills: list[dict] = []

    mode = "LoRA" if trainer_mod else "traditional ML"
    print(
        f"[kaggle/loop] mode={mode} metric={primary_metric} budget={budget} "
        f"defect_analysis={'on' if enable_defect_analysis else 'off'} "
        f"resuming_from={next_idx}"
    )

    # --- Phase A: exploration -------------------------------------------------
    while next_idx < budget:
        cfg = propose_next(
            results=all_results,
            model_name=model_name or "traditional_ml",
            next_idx=next_idx,
            use_llm=use_llm_proposer,
            skill_context=_skills_context(learned_skills),
        )
        if cfg.exp_id in completed:
            cfg.exp_id = f"{cfg.exp_id}_v{next_idx}"
        completed.add(cfg.exp_id)

        print(f"\n=== [{next_idx + 1}/{budget}] {cfg.exp_id} ===")
        print(f"hypothesis: {cfg.hypothesis}")

        result, curr_preds = _run_one(
            cfg, data_mod, evaluator_mod, trainer_mod,
            model_name, runs_dir, primary_metric,
        )
        result_dict = result.to_dict()

        # --- Defect analysis --------------------------------------------------
        defect_report = None
        new_skills: list[dict] = []

        if enable_defect_analysis and curr_preds and result.status == "success":
            prev_preds, prev_exp_id = _prev_preds(all_results, runs_dir)
            if prev_preds:
                try:
                    defect_report = analyze_defects(
                        prev_preds, curr_preds,
                        iteration=next_idx,
                        exp_id=cfg.exp_id,
                        prev_exp_id=prev_exp_id,
                        sample_size=sample_size,
                        sampling_strategy=sampling_strategy,
                        threshold=threshold,
                        near_delta=near_delta,
                        mid_delta=mid_delta,
                    )
                    print(defect_report.format_summary())
                    result_dict.setdefault("extra", {})["defect"] = {
                        "n_fixed": defect_report.n_fixed,
                        "n_new_errors": defect_report.n_new_errors,
                        "net_change": defect_report.net_change,
                    }
                except Exception as e:
                    print(f"[loop] Defect analysis failed: {e}")

            if defect_report:
                try:
                    new_skills = propose_skills(
                        defect_summary=defect_report.format_summary(),
                        journal_summary=journal.format_for_proposer(),
                        task_context=task_context,
                        provider=provider,
                        model=ai_model,
                        api_key=ai_api_key,
                        base_url=ai_base_url,
                    )
                    if new_skills:
                        learned_skills = new_skills
                        print(f"[loop] Skills proposed: {[s.get('name') for s in new_skills]}")
                except Exception as e:
                    print(f"[loop] Skill proposer failed: {e}")

        # --- Journal update ---------------------------------------------------
        journal.add_entry(
            iteration=next_idx,
            exp_id=cfg.exp_id,
            metrics={
                primary_metric: result.accuracy,
                "n_eval": result.n_eval,
                "train_loss": result.train_loss,
                "status": result.status,
            },
            defect_report=defect_report.to_dict() if defect_report else None,
            proposed_skills=new_skills,
        )

        append_result(results_path, result_dict)
        all_results.append(result_dict)
        write_leaderboard(all_results, leaderboard_path)

        if result.status == "success":
            print(
                f"  -> {primary_metric}={result.accuracy:.4f} "
                f"n={result.n_eval} loss={result.train_loss:.4f} "
                f"({result.runtime_sec:.0f}s)"
            )
        else:
            print(f"  -> FAILED [{result.status}]: {result.error[:200]}")

        if target_accuracy is not None:
            best = best_so_far(all_results)
            if best and best["accuracy"] >= target_accuracy:
                print(f"[loop] Target {primary_metric}={target_accuracy} reached — stopping.")
                break
        if has_plateaued(all_results, window=4, min_delta=0.002):
            print("[loop] Plateau detected — stopping exploration.")
            break

        next_idx += 1

    # --- Phase B: promote top-K -----------------------------------------------
    summary = {
        "leaderboard": str(leaderboard_path),
        "results": str(results_path),
        "journal": str(journal.path),
        "best": best_so_far(all_results),
    }
    if skip_full_phase:
        return summary

    leaders = top_k(all_results, k=final_top_k)
    if not leaders:
        print("[loop] No successful runs — skipping full-tier phase.")
        return summary

    print(f"\n=== Phase B: promoting top {len(leaders)} to tier='full' ===")
    for r in leaders:
        cfg = ExperimentConfig.from_dict(r["config"])
        cfg.tier = "full"
        cfg.exp_id = f"{cfg.exp_id}_FULL"
        if cfg.exp_id in completed:
            print(f"[loop] {cfg.exp_id} already done, skipping.")
            continue
        cfg.notes = f"Full-tier rerun of {r['exp_id']} (small {primary_metric}={r['accuracy']:.4f})."
        print(f"\n=== FULL: {cfg.exp_id} ===")
        result, _ = _run_one(cfg, data_mod, evaluator_mod, trainer_mod, model_name, runs_dir, primary_metric)
        result_dict = result.to_dict()
        append_result(results_path, result_dict)
        all_results.append(result_dict)
        write_leaderboard(all_results, leaderboard_path)
        if result.status == "success":
            print(f"  -> FULL {primary_metric}={result.accuracy:.4f}")
        else:
            print(f"  -> FAILED [{result.status}]")

    summary["best"] = best_so_far(all_results)
    return summary
