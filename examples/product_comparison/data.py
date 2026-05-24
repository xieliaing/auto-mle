"""
Product-comparison data module.

Implements the AutoMLE data interface for the product-matching binary task:

  load_train_dataset(seed, subset, **kwargs) -> torch.utils.data.Dataset
  load_eval_dataset(seed, subset, **kwargs)  -> pandas.DataFrame
  get_collator(tokenizer, max_seq_len)       -> ProductPairCollator

Source CSV schema:
    title1: str, title2: str, Label: int   (image columns are ignored)

Data paths are taken from kwargs ('train_path' / 'eval_path') or, as a
fallback, the AUTOMLE_TRAIN_DATA / AUTOMLE_EVAL_DATA environment variables.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import os
import random
import re

import pandas as pd

from .prompts import build_messages, label_to_answer


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(t.lower() for t in _TOKEN_RE.findall(str(s)))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --- Loading ------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load and lightly validate a pairs CSV."""
    df = pd.read_csv(path)
    required = {"title1", "title2", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.dropna(subset=["title1", "title2", "Label"]).copy()
    df["title1"] = df["title1"].astype(str)
    df["title2"] = df["title2"].astype(str)
    df["Label"] = df["Label"].astype(int)
    return df.reset_index(drop=True)


def _resolve_path(kwargs_key: str, env_key: str, kwargs: dict) -> str:
    path = kwargs.get(kwargs_key) or os.environ.get(env_key)
    if not path:
        raise ValueError(
            f"Provide '{kwargs_key}' via kwargs or set ${env_key}."
        )
    return path


# --- Data strategies ----------------------------------------------------------

def _augment_title(title: str, rng: random.Random) -> str:
    ops = []
    if rng.random() < 0.3:
        ops.append("lower")
    if rng.random() < 0.2:
        ops.append("drop")
    if rng.random() < 0.1:
        ops.append("dup_space")
    out = title
    if "lower" in ops:
        out = out.lower()
    if "drop" in ops:
        toks = out.split()
        if len(toks) > 4:
            i = rng.randrange(len(toks))
            toks.pop(i)
            out = " ".join(toks)
    if "dup_space" in ops:
        out = out.replace(" ", "  ", 1)
    return out


def _mine_hard_negatives(
    df: pd.DataFrame, frac: float, rng: random.Random,
) -> pd.DataFrame:
    if frac <= 0.0:
        return df
    neg = df[df["Label"] == 0].reset_index(drop=True)
    if len(neg) < 2:
        return df
    bucket: dict[str, list[int]] = {}
    for i, t in enumerate(neg["title2"].tolist()):
        for tok in _tokens(t):
            bucket.setdefault(tok, []).append(i)
    new_title2 = neg["title2"].tolist()
    for i, t1 in enumerate(neg["title1"].tolist()):
        if rng.random() >= frac:
            continue
        cand_set: set[int] = set()
        for tok in _tokens(t1):
            cand_set.update(bucket.get(tok, [])[:8])
        cand_set.discard(i)
        if not cand_set:
            continue
        j = rng.choice(list(cand_set))
        new_title2[i] = neg["title2"].iloc[j]
    neg = neg.copy()
    neg["title2"] = new_title2
    pos = df[df["Label"] == 1]
    return pd.concat([pos, neg], ignore_index=True).sample(
        frac=1.0, random_state=rng.randint(0, 1 << 31)
    ).reset_index(drop=True)


def _balance(df: pd.DataFrame, mode: str, rng: random.Random) -> pd.DataFrame:
    if mode == "none":
        return df
    pos = df[df["Label"] == 1]
    neg = df[df["Label"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return df
    seed = rng.randint(0, 1 << 31)
    if mode == "downsample":
        n = min(len(pos), len(neg))
        return pd.concat([pos.sample(n, random_state=seed),
                          neg.sample(n, random_state=seed + 1)],
                         ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    if mode == "upsample":
        n = max(len(pos), len(neg))
        return pd.concat([pos.sample(n, replace=True, random_state=seed),
                          neg.sample(n, replace=True, random_state=seed + 1)],
                         ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return df


# --- Torch dataset ------------------------------------------------------------

try:
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore
except ImportError:
    _TorchDataset = object  # type: ignore


class ProductPairDataset(_TorchDataset):
    """
    Holds the post-processed DataFrame and serves dicts:
        {"messages": [...chat...], "label": int, "title1": str, "title2": str}
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        t1, t2 = row["title1"], row["title2"]
        if self.augment:
            t1 = _augment_title(t1, self._rng)
            t2 = _augment_title(t2, self._rng)
        label = int(row["Label"])
        return {
            "messages": build_messages(t1, t2, answer=label_to_answer(label)),
            "label": label,
            "title1": t1,
            "title2": t2,
        }

    @property
    def column_names(self):
        return ["input_ids", "labels", "messages", "label", "title1", "title2"]


# --- AutoMLE data interface ---------------------------------------------------

def load_train_dataset(
    seed: int = 42,
    subset: Optional[int] = None,
    balance: str = "none",
    hard_neg_frac: float = 0.0,
    augment_titles: bool = False,
    min_jaccard: float = 0.0,
    max_jaccard: float = 1.0,
    **kwargs,
):
    """Build the training Dataset for the product-comparison task."""
    rng = random.Random(seed)
    df = load_csv(_resolve_path("train_path", "AUTOMLE_TRAIN_DATA", kwargs))

    if min_jaccard > 0.0 or max_jaccard < 1.0:
        jac = df.apply(lambda r: _jaccard(r["title1"], r["title2"]), axis=1)
        df = df[(jac >= min_jaccard) & (jac <= max_jaccard)].reset_index(drop=True)

    df = _balance(df, balance, rng)
    if hard_neg_frac > 0.0:
        df = _mine_hard_negatives(df, hard_neg_frac, rng)

    if subset is not None and subset < len(df):
        df = df.sample(subset, random_state=rng.randint(0, 1 << 31)).reset_index(drop=True)

    return ProductPairDataset(df, augment=augment_titles, seed=seed)


def load_eval_dataset(
    seed: int = 42,
    subset: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Return the eval frame; the product evaluator iterates rows directly."""
    rng = random.Random(seed)
    df = load_csv(_resolve_path("eval_path", "AUTOMLE_EVAL_DATA", kwargs))
    if subset is not None and subset < len(df):
        df = df.sample(subset, random_state=rng.randint(0, 1 << 31)).reset_index(drop=True)
    return df


def get_collator(tokenizer, max_seq_len: int):
    """Return the product-pair collator that masks loss to the answer token."""
    from .trainer import ProductPairCollator
    return ProductPairCollator(tokenizer=tokenizer, max_seq_len=max_seq_len)
