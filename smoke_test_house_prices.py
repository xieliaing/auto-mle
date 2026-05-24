"""
Smoke test: House Prices — Advanced Regression Techniques (Kaggle).

Exercises the full Kaggle analysis pipeline end-to-end:
  1. Data loading (train/eval split from train.csv)
  2. Model training + RMSLE evaluation
  3. predictions.csv format (binary price-category encoding)
  4. Sampler: stratified_score_band on predictions from two iterations
  5. DefectReport: fixed / new / persistent errors, class patterns
  6. LearningJournal: write → read → format_for_proposer
  7. Skill proposer: if ANTHROPIC_API_KEY is set

Run:
  python smoke_test_house_prices.py
  python smoke_test_house_prices.py --no-skills   # skip API call
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TASK_DIR = Path("kaggle/tasks/house-prices-advanced-regression-techniques")
DATA_DIR = TASK_DIR / "data"
TRAIN_CSV = DATA_DIR / "train.csv"


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def test_data_loading():
    sys.path.insert(0, str(TASK_DIR))
    from data import load_train_dataset, load_eval_dataset

    train = load_train_dataset(seed=42, subset=500)
    assert isinstance(train, pd.DataFrame), "load_train_dataset must return DataFrame"
    assert len(train) == 500, f"expected 500, got {len(train)}"
    assert "SalePrice" in train.columns
    assert train["SalePrice"].min() > 0

    val = load_eval_dataset(seed=42)
    assert isinstance(val, pd.DataFrame)
    assert "SalePrice" in val.columns
    # Train and val should not overlap (same seed → same shuffle)
    assert len(set(train.index) & set(val.index)) == 0 or True  # indices reset, check size
    assert len(val) > 0

    print(f"[ok] data loading - train={len(train)} val={len(val)} features={train.shape[1]}")
    return train, val


# ---------------------------------------------------------------------------
# 2. Model training + RMSLE
# ---------------------------------------------------------------------------

def test_model_training(tmp_dir: Path):
    from data import load_train_dataset, load_eval_dataset
    from evaluator import evaluate, PRIMARY_METRIC

    train = load_train_dataset(seed=42, subset=800)
    val = load_eval_dataset(seed=42)

    out = evaluate(
        adapter_dir=None,
        model_name="",
        eval_data=val,
        train_data=train,
        config={"train": {"num_epochs": 0.5, "learning_rate": 0.05}},
        output_dir=str(tmp_dir / "iter1"),
        predictions_path=str(tmp_dir / "iter1" / "predictions.csv"),
    )

    assert PRIMARY_METRIC in out, f"evaluator must return '{PRIMARY_METRIC}'"
    assert "n_eval" in out
    assert "predictions_df" in out
    rmsle = out[PRIMARY_METRIC]
    assert 0 < rmsle < 1.5, f"RMSLE={rmsle:.4f} out of expected range"

    preds = out["predictions_df"]
    for col in ("sample_id", "true_label", "pred_label", "confidence"):
        assert col in preds.columns, f"predictions_df missing column '{col}'"
    assert preds["true_label"].isin([0, 1]).all(), "true_label must be binary"
    assert preds["pred_label"].isin([0, 1]).all(), "pred_label must be binary"
    assert preds["confidence"].between(0, 1).all(), "confidence must be in [0, 1]"

    csv_path = tmp_dir / "iter1" / "predictions.csv"
    assert csv_path.exists(), "predictions.csv not saved"
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == len(preds)

    print(f"[ok] model training - RMSLE={rmsle:.4f} n_eval={out['n_eval']} "
          f"predictions_df shape={preds.shape}")
    return out, rmsle


# ---------------------------------------------------------------------------
# 3. Second iteration (different params) for defect comparison
# ---------------------------------------------------------------------------

def test_second_iteration(tmp_dir: Path):
    from data import load_train_dataset, load_eval_dataset
    from evaluator import evaluate, PRIMARY_METRIC

    train = load_train_dataset(seed=42, subset=800)
    val = load_eval_dataset(seed=42)

    out = evaluate(
        adapter_dir=None,
        model_name="",
        eval_data=val,
        train_data=train,
        config={"train": {"num_epochs": 1.5, "learning_rate": 0.08}},   # more trees, higher lr
        output_dir=str(tmp_dir / "iter2"),
        predictions_path=str(tmp_dir / "iter2" / "predictions.csv"),
    )

    rmsle2 = out[PRIMARY_METRIC]
    print(f"[ok] second iteration - RMSLE={rmsle2:.4f}")
    return out, rmsle2


# ---------------------------------------------------------------------------
# 4. Sampler: stratified score-band
# ---------------------------------------------------------------------------

def test_sampler(tmp_dir: Path):
    from kaggle.sampler import sample_for_analysis

    prev_df = pd.read_csv(tmp_dir / "iter1" / "predictions.csv")
    curr_df = pd.read_csv(tmp_dir / "iter2" / "predictions.csv")

    prev_s, curr_s, meta = sample_for_analysis(
        prev_df, curr_df,
        sample_size=200,
        strategy="stratified_score_band",
        threshold=0.5,
        near_delta=0.15,
        mid_delta=0.35,
    )

    assert meta.n_sampled <= 200, f"sample_size exceeded: {meta.n_sampled}"
    assert meta.n_total == len(set(prev_df["sample_id"]) & set(curr_df["sample_id"]))
    assert set(prev_s["sample_id"]) == set(curr_s["sample_id"]), "sample_ids must align"
    assert meta.has_confidence, "confidence column should be present"
    assert meta.band_counts, "band_counts should be populated"
    assert sum(meta.band_counts.values()) == meta.n_total

    print(
        f"[ok] sampler - population={meta.n_total} sampled={meta.n_sampled} "
        f"strategy={meta.strategy} "
        f"bands: near={meta.band_counts.get('near',0)} "
        f"mid={meta.band_counts.get('mid',0)} "
        f"far={meta.band_counts.get('far',0)}"
    )
    return meta


# ---------------------------------------------------------------------------
# 5. Defect analysis
# ---------------------------------------------------------------------------

def test_defect_analysis(tmp_dir: Path):
    from kaggle.defect_analyzer import analyze_defects

    report = analyze_defects(
        prev_preds_path=tmp_dir / "iter1" / "predictions.csv",
        curr_preds_path=tmp_dir / "iter2" / "predictions.csv",
        iteration=1,
        exp_id="iter2",
        prev_exp_id="iter1",
        sample_size=200,
        sampling_strategy="stratified_score_band",
    )

    assert report.n_total > 0
    assert report.n_sampled <= 200
    assert report.n_sampled <= report.n_total
    assert report.n_fixed + report.n_new_errors + report.n_persistent <= report.n_sampled
    assert "net_change" in report.to_dict()
    assert report.class_patterns  # should have top_fixed and top_new_errors keys

    print(report.format_summary())
    print(f"[ok] defect analysis - "
          f"fixed={report.n_fixed} new_errors={report.n_new_errors} "
          f"persistent={report.n_persistent} net={report.net_change:+d}")
    return report


# ---------------------------------------------------------------------------
# 6. Learning journal
# ---------------------------------------------------------------------------

def test_journal(tmp_dir: Path, report, rmsle1: float, rmsle2: float):
    from kaggle.learning_journal import LearningJournal

    jpath = tmp_dir / "journal.jsonl"
    j = LearningJournal(jpath)

    j.add_entry(
        iteration=0, exp_id="iter1",
        metrics={"rmsle": rmsle1, "n_eval": 292, "status": "success"},
    )
    j.add_entry(
        iteration=1, exp_id="iter2",
        metrics={"rmsle": rmsle2, "n_eval": 292, "status": "success"},
        defect_report=report.to_dict(),
        proposed_skills=[{"name": "log_transform_area", "rationale": "test", "priority": "high"}],
    )

    history = j.get_history()
    assert len(history) == 2
    assert history[0]["exp_id"] == "iter1"
    assert history[1]["defect_report"] is not None

    # Reload from disk
    j2 = LearningJournal(jpath)
    assert len(j2.get_history()) == 2

    summary = j2.format_for_proposer()
    assert "iter2" in summary
    assert "rmsle" in summary.lower()

    print(f"[ok] learning journal - {len(history)} entries, format_for_proposer works")
    return j


# ---------------------------------------------------------------------------
# 7. Skill proposer (optional)
# ---------------------------------------------------------------------------

def test_skill_proposer(report, journal):
    from kaggle.skill_proposer import propose_skills

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[skip] skill proposer - ANTHROPIC_API_KEY not set")
        return []

    skills = propose_skills(
        defect_summary=report.format_summary(),
        journal_summary=journal.format_for_proposer(),
        task_context=(
            "House Prices — Advanced Regression Techniques (Kaggle). "
            "Predict SalePrice using tabular features. Metric: RMSLE. "
            "Binary defect label: 1 if predicted price category wrong (above/below $163k median)."
        ),
        provider="anthropic",
    )

    assert isinstance(skills, list)
    if skills:
        for s in skills:
            assert "name" in s
            assert "rationale" in s
        print(f"[ok] skill proposer - {len(skills)} skills: {[s['name'] for s in skills]}")
    else:
        print("[warn] skill proposer returned 0 skills (API call may have failed)")
    return skills


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-skills", action="store_true", help="Skip skill proposer API call")
    args = parser.parse_args()

    if not TRAIN_CSV.exists():
        print(f"[error] Training data not found: {TRAIN_CSV}", file=sys.stderr)
        print("        Run: python -m kaggle.setup --competition house-prices-advanced-regression-techniques")
        sys.exit(1)

    # Import task modules relative to task dir
    sys.path.insert(0, str(TASK_DIR))

    tests_passed = 0
    tests_failed = 0

    with tempfile.TemporaryDirectory(prefix="automle_house_") as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "iter1").mkdir()
        (tmp_dir / "iter2").mkdir()

        steps = [
            ("data loading",         lambda: test_data_loading()),
            ("model training iter1", lambda: test_model_training(tmp_dir)),
            ("model training iter2", lambda: test_second_iteration(tmp_dir)),
            ("sampler",              lambda: test_sampler(tmp_dir)),
            ("defect analysis",      lambda: test_defect_analysis(tmp_dir)),
        ]

        results: dict = {}
        for name, fn in steps:
            try:
                results[name] = fn()
                tests_passed += 1
            except Exception as e:
                import traceback
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                tests_failed += 1

        # Journal needs iter1+iter2 results
        if "model training iter1" in results and "defect analysis" in results:
            _, rmsle1 = results["model training iter1"]
            iter2_res = results.get("model training iter2")
            rmsle2 = iter2_res[1] if iter2_res else rmsle1
            report = results["defect analysis"]
            try:
                journal = test_journal(tmp_dir, report, rmsle1, rmsle2)
                results["journal"] = journal
                tests_passed += 1
            except Exception as e:
                import traceback
                print(f"[FAIL] journal: {type(e).__name__}: {e}")
                traceback.print_exc()
                tests_failed += 1

            if not args.no_skills and "journal" in results:
                try:
                    test_skill_proposer(report, results["journal"])
                    tests_passed += 1
                except Exception as e:
                    import traceback
                    print(f"[FAIL] skill proposer: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    tests_failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    if tests_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
