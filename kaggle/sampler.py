"""
Adaptive stratified sampling for defect analysis.

Rather than loading full prediction files (potentially millions of rows) and
sampling randomly, we use the previous model's predicted confidence scores to
draw a budget-bounded, pattern-rich sample.

Baseline strategy: stratified_score_band (binary classification)
───────────────────────────────────────────────────────────────
Divides predictions into three bands based on |confidence - threshold|:

  near  (|s - t| < near_delta,  default 0.15):  the uncertain zone — most
        classification errors cluster here; assigned 50% of the sample budget.

  mid   (near_delta ≤ |s - t| < mid_delta, default 0.35):  moderate confidence;
        assigned 30% of the budget.

  far   (|s - t| ≥ mid_delta):  high-confidence zone — errors here are
        "surprise" misclassifications that reveal systematic issues;
        assigned 20% of the budget.

Within each band, wrong predictions (prev model was wrong) are prioritised:
  • near/mid wrongs: sorted by |s - t| ascending (most borderline first)
  • far wrongs: sorted by |s - t| descending (most overconfident errors first)
  • Correct predictions fill the remaining per-band budget.

If a band has fewer total rows than its budget, the unused budget rolls over
into a global reserve that is distributed across all remaining unsampled rows
(wrongs first) after the initial per-band pass.

Fallback (no confidence column):
  Stratify by the four (true_label, pred_label) cells — TP, FP, FN, TN.
  Error cells (FP + FN) receive 70% of the budget.

Future strategies to add (one at a time):
  - multi_class_score_band: per-class confidence bands for N-class problems
  - regression_residual_band: band by |prediction - true_value| / std(residuals)
  - uncertainty_ensemble: band by disagreement across an ensemble
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------

@dataclass
class SampleMeta:
    """Metadata returned alongside the sampled DataFrames."""
    strategy: str
    n_total: int          # rows common to both prediction files
    n_sampled: int        # rows actually selected
    has_confidence: bool
    band_counts: dict     # {"near": n, "mid": n, "far": n} or {} for fallback
    band_budgets: dict    # {"near": budget, "mid": budget, "far": budget} or {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_band(
    df: pd.DataFrame,
    budget: int,
    asc_sort: bool,
    rng: int,
) -> tuple[pd.DataFrame, int]:
    """
    Draw up to `budget` rows from `df`.

    Prioritises rows where _wrong=True, sorted by _dist (ascending if asc_sort,
    descending otherwise).  Correct rows fill remaining capacity.
    Returns (sample, leftover_budget).
    """
    if budget <= 0 or df.empty:
        return pd.DataFrame(columns=df.columns), budget

    if len(df) <= budget:
        return df, budget - len(df)

    wrong = df[df["_wrong"]].sort_values("_dist", ascending=asc_sort)
    correct = df[~df["_wrong"]]

    n_wrong = min(len(wrong), budget)
    taken_wrong = wrong.iloc[:n_wrong]
    remaining = budget - n_wrong
    n_correct = min(len(correct), remaining)
    taken_correct = correct.sample(n_correct, random_state=rng) if n_correct > 0 else pd.DataFrame(columns=df.columns)

    return pd.concat([taken_wrong, taken_correct], ignore_index=True), 0


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _stratified_score_band_binary(
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    sample_size: int,
    threshold: float,
    near_delta: float,
    mid_delta: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SampleMeta]:
    """
    Stratified score-band sampling for binary classification.
    Uses the previous model's 'confidence' column (prob of positive class).
    """
    common_ids = set(prev_df["sample_id"]) & set(curr_df["sample_id"])
    n_total = len(common_ids)

    prev = prev_df[prev_df["sample_id"].isin(common_ids)].copy()
    curr = curr_df[curr_df["sample_id"].isin(common_ids)].copy()

    if n_total <= sample_size:
        meta = SampleMeta(
            strategy="stratified_score_band",
            n_total=n_total, n_sampled=n_total,
            has_confidence=True,
            band_counts={}, band_budgets={},
        )
        return prev, curr, meta

    # Annotate for sampling
    prev["_dist"] = (prev["confidence"] - threshold).abs()
    prev["_wrong"] = prev["true_label"] != prev["pred_label"]

    near_mask = prev["_dist"] < near_delta
    mid_mask = (prev["_dist"] >= near_delta) & (prev["_dist"] < mid_delta)
    far_mask = prev["_dist"] >= mid_delta

    near_budget = int(sample_size * 0.50)
    mid_budget = int(sample_size * 0.30)
    far_budget = sample_size - near_budget - mid_budget

    band_counts = {
        "near": int(near_mask.sum()),
        "mid": int(mid_mask.sum()),
        "far": int(far_mask.sum()),
    }
    band_budgets = {"near": near_budget, "mid": mid_budget, "far": far_budget}

    # Per-band sampling: near/mid sort asc (borderline first); far sort desc (overconfident first)
    near_sample, near_left = _sample_band(prev[near_mask], near_budget, asc_sort=True, rng=random_state)
    mid_sample, mid_left = _sample_band(prev[mid_mask], mid_budget, asc_sort=True, rng=random_state + 1)
    far_sample, far_left = _sample_band(prev[far_mask], far_budget, asc_sort=False, rng=random_state + 2)

    # Distribute reserve: fill from unsampled rows, wrongs first across all bands
    reserve = near_left + mid_left + far_left
    sampled_ids: set = (
        set(near_sample["sample_id"])
        | set(mid_sample["sample_id"])
        | set(far_sample["sample_id"])
    )
    if reserve > 0:
        unsampled = prev[~prev["sample_id"].isin(sampled_ids)]
        reserve_sample, _ = _sample_band(unsampled, reserve, asc_sort=True, rng=random_state + 3)
    else:
        reserve_sample = pd.DataFrame(columns=prev.columns)

    selected = pd.concat([near_sample, mid_sample, far_sample, reserve_sample], ignore_index=True)

    # Drop internal helper columns before returning
    selected = selected.drop(columns=["_dist", "_wrong"])
    selected_ids = set(selected["sample_id"])
    curr_filtered = curr[curr["sample_id"].isin(selected_ids)].copy()

    meta = SampleMeta(
        strategy="stratified_score_band",
        n_total=n_total,
        n_sampled=len(selected),
        has_confidence=True,
        band_counts=band_counts,
        band_budgets=band_budgets,
    )
    return selected, curr_filtered, meta


def _fallback_label_stratified(
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SampleMeta]:
    """
    Fallback when no confidence score is available.
    Stratifies by (true_label, pred_label) cell; error cells get 70% of budget.
    """
    common_ids = set(prev_df["sample_id"]) & set(curr_df["sample_id"])
    n_total = len(common_ids)
    prev = prev_df[prev_df["sample_id"].isin(common_ids)].copy()
    curr = curr_df[curr_df["sample_id"].isin(common_ids)].copy()

    if n_total <= sample_size:
        meta = SampleMeta(
            strategy="fallback_label_stratified", n_total=n_total, n_sampled=n_total,
            has_confidence=False, band_counts={}, band_budgets={},
        )
        return prev, curr, meta

    prev["_wrong"] = prev["true_label"] != prev["pred_label"]
    wrong = prev[prev["_wrong"]]
    correct = prev[~prev["_wrong"]]

    n_wrong_budget = min(len(wrong), int(sample_size * 0.70))
    n_correct_budget = sample_size - n_wrong_budget

    wrong_sample = wrong.sample(n_wrong_budget, random_state=random_state) if n_wrong_budget > 0 else pd.DataFrame(columns=prev.columns)
    correct_sample = correct.sample(min(len(correct), n_correct_budget), random_state=random_state + 1) if n_correct_budget > 0 else pd.DataFrame(columns=prev.columns)

    selected = pd.concat([wrong_sample, correct_sample], ignore_index=True)
    selected = selected.drop(columns=["_wrong"])
    selected_ids = set(selected["sample_id"])
    curr_filtered = curr[curr["sample_id"].isin(selected_ids)].copy()

    meta = SampleMeta(
        strategy="fallback_label_stratified", n_total=n_total, n_sampled=len(selected),
        has_confidence=False, band_counts={}, band_budgets={},
    )
    return selected, curr_filtered, meta


# ---------------------------------------------------------------------------
# Registry & public API
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "stratified_score_band": _stratified_score_band_binary,
    # Future: "multi_class_score_band", "regression_residual_band", ...
}


def sample_for_analysis(
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    sample_size: int = 1000,
    strategy: str = "stratified_score_band",
    threshold: float = 0.5,
    near_delta: float = 0.15,
    mid_delta: float = 0.35,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, SampleMeta]:
    """
    Draw an adaptive, pattern-rich sample from aligned prediction DataFrames.

    Parameters
    ----------
    prev_df, curr_df : DataFrames with at minimum [sample_id, true_label, pred_label].
                       Binary classification: include a 'confidence' column
                       (predicted probability of the positive class, 0.0–1.0).
    sample_size      : Total rows to select.  0 = use all rows (no sampling).
    strategy         : Sampling strategy.  Currently: 'stratified_score_band'.
    threshold        : Decision boundary (default 0.5 for binary classification).
    near_delta       : Half-width of the near-threshold band (default 0.15).
    mid_delta        : Half-width of the mid-confidence band (default 0.35).
    random_state     : RNG seed for reproducibility.

    Returns
    -------
    (sampled_prev, sampled_curr, SampleMeta)
    Both DataFrames contain the same set of sample_ids.
    """
    if sample_size == 0:
        # 0 means no limit — return full intersection
        common = set(prev_df["sample_id"]) & set(curr_df["sample_id"])
        n = len(common)
        p = prev_df[prev_df["sample_id"].isin(common)]
        c = curr_df[curr_df["sample_id"].isin(common)]
        return p, c, SampleMeta("none", n, n, False, {}, {})

    has_conf = (
        "confidence" in prev_df.columns
        and prev_df["confidence"].notna().mean() > 0.9  # at least 90% non-null
    )

    if strategy == "stratified_score_band" and has_conf:
        return _stratified_score_band_binary(
            prev_df, curr_df, sample_size,
            threshold=threshold,
            near_delta=near_delta,
            mid_delta=mid_delta,
            random_state=random_state,
        )

    if strategy == "stratified_score_band" and not has_conf:
        # Silently fall back — confidence column absent or too sparse
        return _fallback_label_stratified(prev_df, curr_df, sample_size, random_state)

    if strategy in _STRATEGIES:
        return _STRATEGIES[strategy](
            prev_df, curr_df, sample_size,
            threshold=threshold, near_delta=near_delta, mid_delta=mid_delta,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown sampling strategy: {strategy!r}. "
        f"Available: {list(_STRATEGIES.keys())}"
    )
