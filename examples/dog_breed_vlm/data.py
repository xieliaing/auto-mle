"""
Dog-breed VLM data module: runs the Kaggle dog-breed task (Qwen3-VL + LoRA,
see kaggle/tasks/dog-breed-identification/vlm_lora.py) under the core
orchestrator, so it gets the public/private protocol and defect analysis.

Splits labels.csv once with a fixed seed (independent of the run's seed):
  train   ~85%                 load_train_dataset
  public  half of the holdout  load_public_dataset  (ranks runs, defects -> LLM)
  private other half           load_eval_dataset    (Phase-B final score only)
The holdout is split per breed, so every breed appears in both halves.

The core tier sizes are text-scale (tier 'small' = 20k examples), so any
numeric train subset is capped at $DOG_EXPLORE_TRAIN (default 1000) to keep
exploration runs short; tier 'full' (subset=None) trains on all of train.

Data: python -m kaggle.setup --competition dog-breed-identification

Run (from repo root; --no-branch so the sibling trainer.py is used):
    python run.py --no-branch --key dog-breed-vlm \\
        --checkpoint Qwen/Qwen3-VL-2B-Instruct \\
        --data-file examples/dog_breed_vlm/data.py \\
        --evaluator-file examples/dog_breed_vlm/evaluator.py \\
        --analysis-provider local --proposer-provider local \\
        --analysis-base-url http://<spark>:8000/v1
"""
from __future__ import annotations
import os
import random
from functools import lru_cache
from typing import Optional

import pandas as pd

from ._vlm import vlm

SPLIT_SEED = 42  # fixed so the public/private sets never move between runs
EXPLORE_TRAIN = int(os.environ.get("DOG_EXPLORE_TRAIN", "1000"))


@lru_cache(maxsize=1)
def _splits():
    """(train, public, private, breeds)."""
    train, holdout, breeds = vlm.load_split(SPLIT_SEED)
    rng = random.Random(SPLIT_SEED)
    public_idx = []
    for _, grp in holdout.groupby("breed"):
        idx = grp.index.tolist()
        rng.shuffle(idx)
        public_idx.extend(idx[: len(idx) // 2])
    is_public = holdout.index.isin(public_idx)
    return (train, holdout[is_public].reset_index(drop=True),
            holdout[~is_public].reset_index(drop=True), breeds)


def _dataset(df: pd.DataFrame, breeds: list[str]):
    ds = vlm.DogBreedDataset(df, vlm.build_prompt(breeds))
    ds.breeds = breeds  # the evaluator matches generations against these
    return ds


def _sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is not None and n < len(df):
        return df.sample(n, random_state=seed).reset_index(drop=True)
    return df


def _balance(df: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    if mode == "none":
        return df
    counts = df["breed"].value_counts()
    n = counts.min() if mode == "downsample" else counts.max()
    groups = [g.sample(n, replace=(mode == "upsample"), random_state=seed)
              for _, g in df.groupby("breed")]
    return pd.concat(groups).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_train_dataset(seed: int = 42, subset: Optional[int] = None,
                       balance: str = "none", **kwargs):
    train, _, _, breeds = _splits()
    if subset is not None:
        subset = min(subset, EXPLORE_TRAIN)
    return _dataset(_sample(_balance(train, balance, seed), subset, seed), breeds)


def load_eval_dataset(seed: int = 42, subset: Optional[int] = None, **kwargs):
    """The private set."""
    _, _, private, breeds = _splits()
    return _dataset(_sample(private, subset, seed), breeds)


def load_public_dataset(**kwargs):
    _, public, _, breeds = _splits()
    return _dataset(public, breeds)
