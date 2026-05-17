"""
Data loading and per-experiment data-strategy application.

Loads `train_pairs.csv` and `test_pairs.csv` lazily, then applies whatever
DataConfigSpec asks for (subset size, balance, hard-negative mining,
augmentation, jaccard filtering). All transformations are deterministic given
`seed` so runs are reproducible.

Schema of input CSVs:
    title1: str, title2: str, image1: str, image2: str, Label: int
We ignore the image columns in this text-only pipeline.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import random
import re

import pandas as pd

# torch is only needed for the ProductPairDataset class. We import it lazily
# inside that class so the rest of this module (loading, balancing, hard-neg
# mining, augmentation, jaccard filtering) is usable in environments without
# torch — useful for unit tests and data-prep scripts.

from .schema import DataConfigSpec
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
    # Coerce types and drop unusable rows
    df = df.dropna(subset=["title1", "title2", "Label"]).copy()
    df["title1"] = df["title1"].astype(str)
    df["title2"] = df["title2"].astype(str)
    df["Label"] = df["Label"].astype(int)
    return df.reset_index(drop=True)


# --- Data strategies ----------------------------------------------------------

def _augment_title(title: str, rng: random.Random) -> str:
    """Lightweight title augmentation for robustness."""
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
    """
    Replace `frac` of the negative pairs with synthetic hard negatives:
    for each negative we keep, we also (with probability `frac`) swap title2
    with a *different* title from another negative whose Jaccard overlap with
    title1 is high. This keeps label valid (still a negative) but makes the
    pair lexically harder.

    Implementation note: this is O(n) using a token-bucket index, not O(n^2).
    """
    if frac <= 0.0:
        return df

    neg = df[df["Label"] == 0].reset_index(drop=True)
    if len(neg) < 2:
        return df

    # Build a small inverted index from token -> list of row indices
    bucket: dict[str, list[int]] = {}
    for i, t in enumerate(neg["title2"].tolist()):
        for tok in _tokens(t):
            bucket.setdefault(tok, []).append(i)

    new_title2 = neg["title2"].tolist()
    for i, t1 in enumerate(neg["title1"].tolist()):
        if rng.random() >= frac:
            continue
        # Find candidates that share at least one token with title1
        cand_set: set[int] = set()
        for tok in _tokens(t1):
            cand_set.update(bucket.get(tok, [])[:8])  # cap to keep it cheap
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


def apply_data_strategy(
    df: pd.DataFrame,
    spec: DataConfigSpec,
    seed: int,
    is_train: bool,
) -> pd.DataFrame:
    """Apply the DataConfigSpec to a raw frame. Eval frames only get subsetting."""
    rng = random.Random(seed)
    out = df

    # 1. Jaccard filter (train only - it's a cleaning op)
    if is_train and (spec.min_jaccard > 0.0 or spec.max_jaccard < 1.0):
        jac = out.apply(lambda r: _jaccard(r["title1"], r["title2"]), axis=1)
        out = out[(jac >= spec.min_jaccard) & (jac <= spec.max_jaccard)].reset_index(drop=True)

    # 2. Balance (train only)
    if is_train:
        out = _balance(out, spec.balance, rng)

    # 3. Hard negatives (train only)
    if is_train and spec.hard_neg_frac > 0.0:
        out = _mine_hard_negatives(out, spec.hard_neg_frac, rng)

    # 4. Subset
    subset = spec.train_subset if is_train else spec.eval_subset
    if subset is not None and subset < len(out):
        out = out.sample(subset, random_state=rng.randint(0, 1 << 31)).reset_index(drop=True)

    # Title augmentation flag is consumed by the Dataset (per-batch), not here.
    return out


# --- Torch dataset ------------------------------------------------------------

def _get_torch_dataset_base():
    """Lazy import of torch.utils.data.Dataset so this module works without torch."""
    from torch.utils.data import Dataset  # type: ignore
    return Dataset


class ProductPairDataset:
    """
    Holds the post-processed DataFrame and serves dicts:
        {"messages": [...chat...], "label": int, "title1": str, "title2": str}

    Tokenization happens in the collator. Augmentation, if enabled, runs here
    so it's stochastic per-epoch.

    Note on inheritance: torch.utils.data.Dataset is just an abstract base
    that requires __len__ and __getitem__. We provide both, and we register
    this class as a virtual subclass at first instantiation so isinstance
    checks in the trainer still pass. This keeps the module importable
    without torch.
    """

    _registered = False

    def __init__(self, df: pd.DataFrame, augment: bool = False, seed: int = 0):
        if not ProductPairDataset._registered:
            try:
                base = _get_torch_dataset_base()
                base.register(ProductPairDataset)
                ProductPairDataset._registered = True
            except ImportError:
                # OK to skip; only matters if a torch DataLoader gets used.
                pass
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

    # SFTTrainer uses this attribute to decide whether to run .map()
    @property
    def column_names(self):
        # Including 'input_ids' tells SFTTrainer the data is pre-processed and
        # the custom collator will handle encoding.
        return ["input_ids", "labels", "messages", "label", "title1", "title2"]


# --- Convenience --------------------------------------------------------------

def load_train_eval(
    train_path: str, eval_path: str, data_spec: DataConfigSpec, seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = apply_data_strategy(load_csv(train_path), data_spec, seed, is_train=True)
    eval_df = apply_data_strategy(load_csv(eval_path), data_spec, seed, is_train=False)
    return train_df, eval_df
