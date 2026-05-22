"""
Typed configuration schema for experiments.

This is the contract that the proposer, runner, and analyzer all agree on.
Every experiment config that is run, logged, or proposed must validate against
ExperimentConfig. Adding a new tunable knob means adding a field here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional
import json


# --- Sub-configs --------------------------------------------------------------

@dataclass
class LoraConfigSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # "attn" = q,k,v,o ; "all_linear" = also gate/up/down ; "qv" = minimal
    target_modules: Literal["attn", "all_linear", "qv"] = "attn"


@dataclass
class TrainConfigSpec:
    learning_rate: float = 2e-4
    num_epochs: float = 1.0
    batch_size: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.03
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    weight_decay: float = 0.0
    max_seq_len: int = 512
    seed: int = 42


@dataclass
class DataConfigSpec:
    # How many training/eval examples to use this run. None = use all.
    train_subset: Optional[int] = 20_000
    eval_subset: Optional[int] = 2_000
    # Class balancing strategy (applies when the task's data.py honors it).
    balance: Literal["none", "downsample", "upsample"] = "none"


@dataclass
class LossConfigSpec:
    # Standard CE is the baseline; others are more aggressive.
    name: Literal["ce", "label_smoothing", "focal", "weighted_ce"] = "ce"
    label_smoothing: float = 0.0       # used when name == "label_smoothing"
    focal_gamma: float = 2.0           # used when name == "focal"
    pos_weight: float = 1.0            # used when name == "weighted_ce"


# --- Top-level experiment config ---------------------------------------------

@dataclass
class ExperimentConfig:
    """One experiment = one (data, lora, train, loss) combination."""

    # Required identification
    exp_id: str
    hypothesis: str  # one-line "why we're trying this"

    # Tier controls compute budget. The proposer should use 'small' until the
    # final round, then promote top-3 to 'full'.
    tier: Literal["small", "medium", "full"] = "small"

    # Sub-configs
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    train: TrainConfigSpec = field(default_factory=TrainConfigSpec)
    data: DataConfigSpec = field(default_factory=DataConfigSpec)
    loss: LossConfigSpec = field(default_factory=LossConfigSpec)

    # Optional notes from the proposer or analyzer
    notes: str = ""

    # ---- IO helpers --------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        # Defensive: accept partial/nested dicts from the proposer's JSON output
        return cls(
            exp_id=d["exp_id"],
            hypothesis=d.get("hypothesis", ""),
            tier=d.get("tier", "small"),
            lora=LoraConfigSpec(**d.get("lora", {})),
            train=TrainConfigSpec(**d.get("train", {})),
            data=DataConfigSpec(**d.get("data", {})),
            loss=LossConfigSpec(**d.get("loss", {})),
            notes=d.get("notes", ""),
        )

    # ---- Tier presets ------------------------------------------------------

    def apply_tier(self) -> "ExperimentConfig":
        """Override compute-related fields based on tier. Mutates and returns self."""
        if self.tier == "small":
            self.data.train_subset = self.data.train_subset or 20_000
            self.data.eval_subset = self.data.eval_subset or 2_000
        elif self.tier == "medium":
            self.data.train_subset = 50_000
            self.data.eval_subset = 5_000
        elif self.tier == "full":
            self.data.train_subset = None  # all training data
            self.data.eval_subset = None   # all eval data
        return self


# --- Result schema ------------------------------------------------------------

@dataclass
class ExperimentResult:
    """What we log after an experiment finishes (or fails)."""

    exp_id: str
    config: dict          # ExperimentConfig.to_dict()
    accuracy: float       # primary metric on test_pairs.csv
    n_eval: int           # how many eval samples were scored
    train_loss: float
    runtime_sec: float
    status: Literal["success", "failed_oom", "failed_other"] = "success"
    error: str = ""
    extra: dict = field(default_factory=dict)  # e.g. precision/recall/f1, per-class

    def to_dict(self) -> dict:
        return asdict(self)
