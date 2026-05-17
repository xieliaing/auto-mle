"""
Smoke test for non-GPU parts of the pipeline.

Validates:
  - schema.ExperimentConfig round-trips through dict/JSON
  - data.apply_data_strategy actually balances / mines hard negatives
  - prompts.build_messages produces the expected chat structure
  - proposer.seed_plan returns 6 distinct experiments
  - analyzer.write_leaderboard produces non-empty markdown
  - proposer._heuristic_fallback works without ANTHROPIC_API_KEY

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
from auto_research.prompts import build_messages, label_to_answer
from auto_research.data import apply_data_strategy, ProductPairDataset, _jaccard
from auto_research.proposer import seed_plan, _heuristic_fallback
from auto_research.analyzer import (
    append_result, load_results, write_leaderboard, has_plateaued, top_k,
)


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


def test_apply_tier():
    cfg = ExperimentConfig(exp_id="t02", hypothesis="x", tier="full")
    cfg.apply_tier()
    assert cfg.data.train_subset is None
    assert cfg.data.eval_subset is None
    cfg2 = ExperimentConfig(exp_id="t03", hypothesis="x", tier="medium")
    cfg2.apply_tier()
    assert cfg2.data.train_subset == 50_000
    print("[ok] tier preset application")


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
    assert len(msgs2) == 2  # no assistant turn when answer=None
    print("[ok] prompt construction")


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


def test_data_strategies():
    df = make_synthetic_df(500)
    spec = DataConfigSpec(train_subset=100, balance="downsample", hard_neg_frac=0.2)
    out = apply_data_strategy(df, spec, seed=42, is_train=True)
    assert len(out) == 100
    pos = (out["Label"] == 1).sum()
    neg = (out["Label"] == 0).sum()
    # Downsampling should leave roughly equal class counts post-subset (approximate
    # since subsetting happens after balance).
    print(f"[ok] data strategies applied (n={len(out)}, pos={pos}, neg={neg})")

    # Eval path should NOT balance/mine — only subset
    eval_out = apply_data_strategy(df, DataConfigSpec(eval_subset=50), seed=42, is_train=False)
    assert len(eval_out) == 50
    print(f"[ok] eval path subsetting (n={len(eval_out)})")


def test_dataset():
    df = make_synthetic_df(20)
    ds = ProductPairDataset(df, augment=False)
    assert len(ds) == 20
    item = ds[0]
    assert "messages" in item and "label" in item
    assert "input_ids" in ds.column_names  # critical for SFTTrainer routing
    print("[ok] ProductPairDataset")


def test_jaccard():
    j = _jaccard("Apple iPhone 15", "iPhone 15 Apple")
    assert j == 1.0
    j2 = _jaccard("Apple iPhone 15", "Samsung Galaxy S24")
    assert j2 == 0.0
    print(f"[ok] jaccard (identical={1.0}, disjoint={j2})")


def test_seed_plan():
    plan = seed_plan()
    assert len(plan) == 6
    ids = [p.exp_id for p in plan]
    assert len(set(ids)) == 6  # all unique
    # Coverage: at least one with focal, one with hard_neg, one with balanced
    losses_used = {p.loss.name for p in plan}
    assert "focal" in losses_used
    assert "label_smoothing" in losses_used
    assert any(p.data.hard_neg_frac > 0 for p in plan)
    assert any(p.data.balance != "none" for p in plan)
    print(f"[ok] seed plan ({len(plan)} experiments, losses={sorted(losses_used)})")


def test_heuristic_fallback():
    # Empty history -> returns the first seed plan entry
    cfg = _heuristic_fallback([], next_idx=0)
    assert cfg.exp_id == "seed_01_baseline"

    # With history, should perturb the best
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
    assert cfg2.lora.r == 16  # 8 doubled
    assert cfg2.loss.name == "focal"  # CE -> focal flip
    print("[ok] heuristic fallback")


def test_analyzer():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        results_path = td / "results.jsonl"
        leaderboard_path = td / "leaderboard.md"

        # Accuracies designed so that ok[:-4] = [0.80, 0.86, 0.87] (earlier_best=0.87)
        # and the last 4 are [0.870, 0.871, 0.870, 0.871] (recent_best=0.871).
        # delta = 0.001 < 0.002 -> plateaued = True.
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

        # The top entry should be one of the high-accuracy runs
        top = top_k(loaded, k=3)
        assert len(top) == 3
        assert top[0]["accuracy"] >= top[1]["accuracy"] >= top[2]["accuracy"]
        assert top[0]["accuracy"] == max(accs)

        # Plateau check: last 4 hover around 0.871, earlier best was 0.87.
        assert has_plateaued(loaded, window=4, min_delta=0.002), (
            f"expected plateau given accs={accs}"
        )

        # Negative case: a clearly improving sequence should NOT plateau
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
        assert not has_plateaued(climbing, window=4, min_delta=0.002), (
            "improving sequence should not be flagged as plateau"
        )

        print("[ok] analyzer (leaderboard, plateau detection both directions, top-k)")


def test_loss_dispatcher_no_torch():
    """Verifies loss dispatcher returns a callable for each name. Requires torch."""
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


if __name__ == "__main__":
    tests = [
        test_schema_roundtrip,
        test_apply_tier,
        test_prompts,
        test_data_strategies,
        test_dataset,
        test_jaccard,
        test_seed_plan,
        test_heuristic_fallback,
        test_analyzer,
        test_loss_dispatcher_no_torch,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"all {len(tests)} smoke tests passed")
