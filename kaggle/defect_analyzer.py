"""
Per-example defect analysis: compare predictions across consecutive iterations.

Identifies which mistakes were fixed (wrong→correct), which new mistakes appeared
(correct→wrong), and patterns by class and feature bucket.

Instead of loading full prediction files, an adaptive stratified sampler
(kaggle/sampler.py) selects a budget-bounded, pattern-rich subset before
analysis.  All reported counts and rates are over the sample; n_total records
the full population size for reference.

predictions.csv required columns: sample_id, true_label, pred_label
Optional column:  confidence  (predicted probability of positive class, 0.0–1.0)
                  — enables score-band sampling and calibration patterns
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from kaggle.sampler import SampleMeta, sample_for_analysis


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DefectReport:
    iteration: int
    exp_id: str
    prev_exp_id: str
    # Population size (before sampling)
    n_total: int
    # Sample actually analysed
    n_sampled: int
    sampling: dict           # SampleMeta as dict (strategy, band counts/budgets)
    # Error counts within the sample
    n_fixed: int             # wrong → correct
    n_new_errors: int        # correct → wrong
    n_persistent: int        # wrong → wrong
    net_change: int          # n_fixed - n_new_errors
    fixed_sample: List[Any] = field(default_factory=list)
    new_error_sample: List[Any] = field(default_factory=list)
    class_patterns: Dict[str, Any] = field(default_factory=dict)
    feature_patterns: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fixed_sample"] = [int(x) if hasattr(x, "item") else x for x in self.fixed_sample]
        d["new_error_sample"] = [int(x) if hasattr(x, "item") else x for x in self.new_error_sample]
        return d

    def format_summary(self) -> str:
        pct = lambda n: f"{100 * n / max(self.n_sampled, 1):.1f}%"
        strategy = self.sampling.get("strategy", "?")
        lines = [
            f"Defect analysis: {self.prev_exp_id} → {self.exp_id}",
            f"  Population / sample : {self.n_total} / {self.n_sampled}"
            + (f"  (strategy={strategy})" if strategy != "none" else ""),
        ]
        # Band breakdown
        bc = self.sampling.get("band_counts", {})
        bb = self.sampling.get("band_budgets", {})
        if bc:
            parts = [f"{b}: {bc.get(b, 0)} rows (budget {bb.get(b, 0)})" for b in ("near", "mid", "far")]
            lines.append("  Score bands         : " + " | ".join(parts))
        lines += [
            f"  Fixed (wrong→right) : {self.n_fixed} ({pct(self.n_fixed)})",
            f"  New errors          : {self.n_new_errors} ({pct(self.n_new_errors)})",
            f"  Persistent errors   : {self.n_persistent} ({pct(self.n_persistent)})",
            f"  Net Δ               : {'+' if self.net_change >= 0 else ''}{self.net_change}",
        ]
        if self.class_patterns.get("top_fixed"):
            lines.append("  Fixed confusion transitions (top 3):")
            for t, c in list(self.class_patterns["top_fixed"].items())[:3]:
                lines.append(f"    * {t}: x{c}")
        if self.class_patterns.get("top_new_errors"):
            lines.append("  New error transitions (top 3):")
            for t, c in list(self.class_patterns["top_new_errors"].items())[:3]:
                lines.append(f"    * {t}: x{c}")
        if self.feature_patterns:
            lines.append("  Feature error concentrations (lift > 1.5x):")
            for feat, info in list(self.feature_patterns.items())[:3]:
                lines.append(
                    f"    * {feat} [{info['bucket']}]: "
                    f"{info['error_rate_pct']}% errors (lift {info['lift']}x)"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal analysis helpers
# ---------------------------------------------------------------------------

def _load_preds(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("sample_id", "true_label", "pred_label"):
        if col not in df.columns:
            raise ValueError(
                f"predictions.csv missing required column '{col}': {path}\n"
                f"  Available: {list(df.columns)}"
            )
    return df


def _class_transition_patterns(df_slice: pd.DataFrame, prev_col: str, curr_col: str) -> dict:
    if df_slice.empty:
        return {}
    return (
        df_slice.apply(
            lambda r: f"true={r['true_label']} | prev={r[prev_col]} -> curr={r[curr_col]}",
            axis=1,
        )
        .value_counts()
        .head(5)
        .to_dict()
    )


def _feature_error_patterns(
    error_ids: List[Any],
    eval_df: pd.DataFrame,
    total_n: int,
    max_feats: int = 5,
) -> dict:
    if not error_ids or eval_df is None:
        return {}

    df = eval_df.copy()
    if "sample_id" not in df.columns:
        df.insert(0, "sample_id", df.index)

    skip = {"sample_id", "true_label", "pred_label", "label", "target", "confidence"}
    feat_cols = [c for c in df.columns if c not in skip]
    overall_err_rate = len(error_ids) / max(total_n, 1)
    scores: list = []

    for col in feat_cols:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                df["_b"] = pd.qcut(
                    df[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
                )
            else:
                df["_b"] = df[col].astype(str)

            total_per = df["_b"].value_counts()
            err_per = df[df["sample_id"].isin(error_ids)]["_b"].value_counts()
            rates = (err_per / total_per.reindex(err_per.index)).dropna()
            df.drop(columns=["_b"], inplace=True, errors="ignore")

            if rates.empty:
                continue
            worst = str(rates.idxmax())
            rate = float(rates.max())
            lift = rate / max(overall_err_rate, 1e-9)
            if lift > 1.5:
                scores.append((lift, col, {
                    "bucket": worst,
                    "error_rate_pct": round(rate * 100, 1),
                    "overall_error_rate_pct": round(overall_err_rate * 100, 1),
                    "lift": round(lift, 2),
                }))
        except Exception:
            df.drop(columns=["_b"], inplace=True, errors="ignore")

    scores.sort(key=lambda x: -x[0])
    return {col: info for _, col, info in scores[:max_feats]}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_defects(
    prev_preds_path: str | Path,
    curr_preds_path: str | Path,
    iteration: int,
    exp_id: str,
    prev_exp_id: str,
    eval_df: Optional[pd.DataFrame] = None,
    # Sampling parameters
    sample_size: int = 1000,
    sampling_strategy: str = "stratified_score_band",
    threshold: float = 0.5,
    near_delta: float = 0.15,
    mid_delta: float = 0.35,
    random_state: int = 42,
) -> DefectReport:
    """
    Compare per-example predictions between two iterations, using adaptive
    stratified sampling to stay within the sample budget.

    Parameters
    ----------
    prev_preds_path, curr_preds_path
        Paths to predictions.csv files from consecutive iterations.
        Required columns: sample_id, true_label, pred_label
        Optional:         confidence (binary classification probability)
    eval_df
        Original evaluation features (same row order / index as predictions).
        Enables feature-level pattern analysis.  Pass None to skip.
    sample_size
        Maximum rows to analyse.  0 = no limit (analyse all).
    sampling_strategy
        'stratified_score_band' (default) or 'none' for no sampling.
    threshold, near_delta, mid_delta
        Decision threshold and band half-widths for score-band sampling.
    random_state
        RNG seed for reproducibility.
    """
    prev_full = _load_preds(prev_preds_path)
    curr_full = _load_preds(curr_preds_path)

    n_total = len(set(prev_full["sample_id"]) & set(curr_full["sample_id"]))

    # --- Adaptive stratified sampling ----------------------------------------
    prev_s, curr_s, meta = sample_for_analysis(
        prev_full, curr_full,
        sample_size=sample_size,
        strategy=sampling_strategy,
        threshold=threshold,
        near_delta=near_delta,
        mid_delta=mid_delta,
        random_state=random_state,
    )

    print(
        f"  [defect] population={n_total} sampled={meta.n_sampled} "
        f"strategy={meta.strategy} has_confidence={meta.has_confidence}"
    )
    if meta.band_counts:
        bc = meta.band_counts
        bb = meta.band_budgets
        print(
            f"  [defect] bands — "
            f"near: {bc.get('near',0)} rows / {bb.get('near',0)} budget  "
            f"mid: {bc.get('mid',0)} / {bb.get('mid',0)}  "
            f"far: {bc.get('far',0)} / {bb.get('far',0)}"
        )

    # --- Defect identification on sample ------------------------------------
    common_ids = set(prev_s["sample_id"]) & set(curr_s["sample_id"])
    prev_idx = prev_s[prev_s["sample_id"].isin(common_ids)].set_index("sample_id")
    curr_idx = curr_s[curr_s["sample_id"].isin(common_ids)].set_index("sample_id")

    m = prev_idx[["true_label", "pred_label"]].rename(columns={"pred_label": "pred_label_prev"})
    m["pred_label_curr"] = curr_idx["pred_label"]
    m["ok_prev"] = m["true_label"] == m["pred_label_prev"]
    m["ok_curr"] = m["true_label"] == m["pred_label_curr"]
    m = m.reset_index()

    fixed_mask = (~m["ok_prev"]) & m["ok_curr"]
    new_err_mask = m["ok_prev"] & (~m["ok_curr"])
    persist_mask = (~m["ok_prev"]) & (~m["ok_curr"])

    fixed_ids = m.loc[fixed_mask, "sample_id"].tolist()
    new_err_ids = m.loc[new_err_mask, "sample_id"].tolist()
    persist_ids = m.loc[persist_mask, "sample_id"].tolist()

    # --- Class-level patterns -----------------------------------------------
    class_pats = {
        "top_fixed": _class_transition_patterns(m[fixed_mask], "pred_label_prev", "pred_label_curr"),
        "top_new_errors": _class_transition_patterns(m[new_err_mask], "pred_label_prev", "pred_label_curr"),
    }

    # --- Feature patterns (on current error set: new + persistent) ----------
    curr_err_ids = new_err_ids + persist_ids
    feat_pats = (
        _feature_error_patterns(curr_err_ids, eval_df, len(common_ids))
        if eval_df is not None else {}
    )

    return DefectReport(
        iteration=iteration,
        exp_id=exp_id,
        prev_exp_id=prev_exp_id,
        n_total=n_total,
        n_sampled=len(common_ids),
        sampling={
            "strategy": meta.strategy,
            "has_confidence": meta.has_confidence,
            "band_counts": meta.band_counts,
            "band_budgets": meta.band_budgets,
        },
        n_fixed=len(fixed_ids),
        n_new_errors=len(new_err_ids),
        n_persistent=len(persist_ids),
        net_change=len(fixed_ids) - len(new_err_ids),
        fixed_sample=fixed_ids[:50],
        new_error_sample=new_err_ids[:50],
        class_patterns=class_pats,
        feature_patterns=feat_pats,
    )
