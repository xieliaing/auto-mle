"""
Smoke test for non-GPU parts of the pipeline.

Validates:
  - auto_research.schema round-trips through dict/JSON
  - tier presets behave correctly
  - proposer.seed_plan returns 6 distinct experiments using only the
    task-agnostic schema fields
  - analyzer.write_leaderboard + plateau detection + top_k work
  - proposer._heuristic_fallback works without ANTHROPIC_API_KEY
  - the product-comparison example exposes the data/evaluator interface
    and its prompts + dataset round-trip

Run with:  python smoke_test.py
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

from auto_research.schema import (
    ExperimentConfig, LoraConfigSpec, TrainConfigSpec, DataConfigSpec, LossConfigSpec,
    ExperimentResult,
)
from auto_research.proposer import seed_plan, _heuristic_fallback
from auto_research.analyzer import (
    append_result, load_results, write_leaderboard, has_plateaued, top_k,
)

from examples.product_comparison.prompts import build_messages, label_to_answer
from examples.product_comparison.data import (
    load_train_dataset, load_eval_dataset, ProductPairDataset, _jaccard,
)


# --- auto_research/ tests ----------------------------------------------------

def test_schema_roundtrip():
    cfg = ExperimentConfig(
        exp_id="t01",
        hypothesis="testing",
        lora=LoraConfigSpec(r=16, alpha=32),
        loss=LossConfigSpec(name="focal", focal_gamma=2.0),
    )
    j = cfg.to_json()
    parsed = json.loads(j)
    cfg2 = ExperimentConfig.from_dict(parsed)
    assert cfg2.exp_id == "t01"
    assert cfg2.lora.r == 16
    assert cfg2.loss.name == "focal"
    assert cfg2.loss.focal_gamma == 2.0
    print("[ok] schema roundtrip")


def test_dataconfig_is_task_agnostic():
    fields = set(DataConfigSpec.__dataclass_fields__)
    assert fields == {"train_subset", "eval_subset", "balance"}, (
        f"DataConfigSpec should be task-agnostic; got fields={fields}"
    )
    print("[ok] DataConfigSpec is task-agnostic")


def test_apply_tier():
    cfg = ExperimentConfig(exp_id="t02", hypothesis="x", tier="full")
    cfg.apply_tier()
    assert cfg.data.train_subset is None
    assert cfg.data.eval_subset is None
    cfg2 = ExperimentConfig(exp_id="t03", hypothesis="x", tier="medium")
    cfg2.apply_tier()
    assert cfg2.data.train_subset == 50_000
    print("[ok] tier preset application")


def test_seed_plan():
    plan = seed_plan()
    assert len(plan) == 6
    ids = [p.exp_id for p in plan]
    assert len(set(ids)) == 6
    losses_used = {p.loss.name for p in plan}
    assert "focal" in losses_used
    assert "label_smoothing" in losses_used
    assert any(p.data.balance != "none" for p in plan)
    print(f"[ok] seed plan ({len(plan)} experiments, losses={sorted(losses_used)})")


def test_heuristic_fallback():
    cfg = _heuristic_fallback([], next_idx=0)
    assert cfg.exp_id == "seed_01_baseline"

    fake_results = [{
        "exp_id": "seed_01_baseline",
        "config": ExperimentConfig(
            exp_id="seed_01_baseline", hypothesis="x",
            lora=LoraConfigSpec(r=8, alpha=16),
            loss=LossConfigSpec(name="ce"),
        ).to_dict(),
        "accuracy": 0.85,
        "status": "success",
    }]
    cfg2 = _heuristic_fallback(fake_results, next_idx=1)
    assert cfg2.exp_id == "heuristic_001"
    assert cfg2.lora.r == 16
    assert cfg2.loss.name == "focal"
    print("[ok] heuristic fallback")


def test_analyzer():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        results_path = td / "results.jsonl"
        leaderboard_path = td / "leaderboard.md"

        accs = [0.80, 0.86, 0.87, 0.870, 0.871, 0.870, 0.871]
        for i, acc in enumerate(accs):
            r = ExperimentResult(
                exp_id=f"exp_{i}",
                config=ExperimentConfig(
                    exp_id=f"exp_{i}", hypothesis="t",
                    lora=LoraConfigSpec(r=8 * (1 + i % 2)),
                ).to_dict(),
                accuracy=acc, n_eval=2000, train_loss=0.1,
                runtime_sec=600.0, status="success",
                extra={"f1": acc - 0.01},
            )
            append_result(results_path, r.to_dict())

        loaded = load_results(results_path)
        assert len(loaded) == len(accs)
        write_leaderboard(loaded, leaderboard_path)
        content = leaderboard_path.read_text()
        assert "Auto-Research Leaderboard" in content

        top = top_k(loaded, k=3)
        assert len(top) == 3
        assert top[0]["accuracy"] >= top[1]["accuracy"] >= top[2]["accuracy"]
        assert top[0]["accuracy"] == max(accs)

        assert has_plateaued(loaded, window=4, min_delta=0.002)

        for i, acc in enumerate([0.70, 0.75, 0.80, 0.85, 0.90]):
            r = ExperimentResult(
                exp_id=f"climb_{i}",
                config=ExperimentConfig(exp_id=f"climb_{i}", hypothesis="t").to_dict(),
                accuracy=acc, n_eval=100, train_loss=0.1,
                runtime_sec=1.0, status="success", extra={"f1": acc},
            )
            results_path2 = td / "climb.jsonl"
            append_result(results_path2, r.to_dict())
        climbing = load_results(td / "climb.jsonl")
        assert not has_plateaued(climbing, window=4, min_delta=0.002)

        print("[ok] analyzer (leaderboard, plateau detection both directions, top-k)")


def test_loss_dispatcher():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[skip] loss dispatcher (torch not installed in this env)")
        return
    from auto_research.losses import make_loss_fn
    for name in ("ce", "label_smoothing", "focal", "weighted_ce"):
        spec = LossConfigSpec(name=name)
        fn = make_loss_fn(spec, pos_token_id=42)
        assert callable(fn)
    print("[ok] loss dispatcher")


# --- examples/product_comparison tests ---------------------------------------

def make_synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Produce a tiny synthetic pairs dataframe."""
    import random
    rng = random.Random(seed)
    rows = []
    brands = ["Apple", "Samsung", "Sony", "Dell", "HP", "Lenovo", "Bose", "JBL"]
    products = ["Phone", "Laptop", "Headphones", "Speaker", "Tablet"]
    for i in range(n):
        b = rng.choice(brands)
        p = rng.choice(products)
        size = rng.choice([8, 16, 32, 64, 128, 256])
        title1 = f"{b} {p} Pro {size}GB Model {i % 7}"
        if rng.random() < 0.5:
            title2 = f"{b} {p} Pro {size}GB"
            label = 1
        else:
            other = rng.choice([x for x in brands if x != b])
            title2 = f"{other} {p} Pro {size}GB"
            label = 0
        rows.append(dict(
            title1=title1, title2=title2,
            image1="dummy.jpg", image2="dummy.jpg",
            Label=label,
        ))
    return pd.DataFrame(rows)


def test_prompts():
    msgs = build_messages("Apple iPhone 15 128GB", "iPhone 15 128GB", answer="Yes")
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "Yes"
    assert label_to_answer(1) == "Yes"
    assert label_to_answer(0) == "No"
    msgs2 = build_messages("a", "b")
    assert len(msgs2) == 2
    print("[ok] prompt construction")


def test_jaccard():
    j = _jaccard("Apple iPhone 15", "iPhone 15 Apple")
    assert j == 1.0
    j2 = _jaccard("Apple iPhone 15", "Samsung Galaxy S24")
    assert j2 == 0.0
    print("[ok] jaccard (identical=1.0, disjoint=0.0)")


def test_example_data_interface(tmp_train_path: Path, tmp_eval_path: Path):
    # load_train_dataset honors balance + subset, returns a Dataset
    ds = load_train_dataset(
        seed=42, subset=100, balance="downsample",
        train_path=str(tmp_train_path),
    )
    assert isinstance(ds, ProductPairDataset)
    assert len(ds) == 100
    item = ds[0]
    assert "messages" in item and "label" in item
    assert "input_ids" in ds.column_names

    # load_eval_dataset returns a DataFrame, subsetted
    ev = load_eval_dataset(seed=42, subset=50, eval_path=str(tmp_eval_path))
    assert isinstance(ev, pd.DataFrame)
    assert len(ev) == 50
    print(f"[ok] example data interface (train Dataset n=100, eval df n=50)")


def test_example_evaluator_contract():
    from examples.product_comparison import evaluator
    assert hasattr(evaluator, "evaluate")
    assert getattr(evaluator, "PRIMARY_METRIC", None) == "accuracy"
    import inspect
    sig = inspect.signature(evaluator.evaluate)
    params = list(sig.parameters)
    # First three positional should be (adapter_dir, model_name, eval_data)
    assert params[:3] == ["adapter_dir", "model_name", "eval_data"], params
    print("[ok] example evaluator contract")


# --- runner ------------------------------------------------------------------

if __name__ == "__main__":
    # Write a tiny synthetic CSV pair for the data-interface test
    tmp = Path(tempfile.mkdtemp(prefix="automle_smoke_"))
    train_path = tmp / "train.csv"
    eval_path = tmp / "eval.csv"
    make_synthetic_df(500, seed=0).to_csv(train_path, index=False)
    make_synthetic_df(200, seed=1).to_csv(eval_path, index=False)

    tests = [
        test_schema_roundtrip,
        test_dataconfig_is_task_agnostic,
        test_apply_tier,
        test_seed_plan,
        test_heuristic_fallback,
        test_analyzer,
        test_loss_dispatcher,
        test_prompts,
        test_jaccard,
        lambda: test_example_data_interface(train_path, eval_path),
        test_example_evaluator_contract,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            name = getattr(t, "__name__", "<lambda>")
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"all {len(tests)} smoke tests passed")
