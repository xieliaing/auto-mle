"""
Custom loss functions for the answer-token-only objective.

All losses here take logits of shape (B, T, V) and labels of shape (B, T) where
non-answer positions are masked to -100. They reduce the answer position(s) to
a single scalar with mean reduction.

Why custom losses are useful for this task:
  - focal: down-weights easy examples; helpful when most pairs are obviously
    different and the model gets them right early in training.
  - label_smoothing: prevents over-confidence; small but reliable gain.
  - weighted_ce: handle class imbalance without resampling the dataset.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import LossConfigSpec


IGNORE_INDEX = -100


def _flatten_unmasked(logits: torch.Tensor, labels: torch.Tensor):
    """Return (logits_kept, labels_kept) over only positions where label != -100.

    logits: (B, T, V)
    labels: (B, T)
    Standard causal-LM shift is assumed to have already happened upstream
    (i.e. labels are aligned to logits so that labels[i] is the target for
    logits[i]).
    """
    mask = labels.ne(IGNORE_INDEX)
    if mask.sum() == 0:
        # Defensive: degenerate batch. Return a 0 loss with grad so the step is a no-op.
        return logits.new_zeros((0, logits.size(-1))), labels.new_zeros((0,))
    return logits[mask], labels[mask]


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    lg, lb = _flatten_unmasked(logits, labels)
    if lg.size(0) == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(lg, lb)


def label_smoothing_loss(
    logits: torch.Tensor, labels: torch.Tensor, smoothing: float,
) -> torch.Tensor:
    lg, lb = _flatten_unmasked(logits, labels)
    if lg.size(0) == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(lg, lb, label_smoothing=smoothing)


def focal_loss(
    logits: torch.Tensor, labels: torch.Tensor, gamma: float,
) -> torch.Tensor:
    """Multi-class focal loss: -(1 - p_t)^gamma * log(p_t)."""
    lg, lb = _flatten_unmasked(logits, labels)
    if lg.size(0) == 0:
        return logits.sum() * 0.0
    log_probs = F.log_softmax(lg, dim=-1)
    log_pt = log_probs.gather(-1, lb.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    focal_factor = (1.0 - pt).clamp(min=1e-8).pow(gamma)
    return -(focal_factor * log_pt).mean()


def weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float,
    pos_token_id: int,
) -> torch.Tensor:
    """
    Cross-entropy where examples whose target is the positive token get
    `pos_weight` extra weight. We can't use F.cross_entropy(weight=...) cleanly
    because the weight vector is over the full vocab; instead we re-weight at
    the example level.
    """
    lg, lb = _flatten_unmasked(logits, labels)
    if lg.size(0) == 0:
        return logits.sum() * 0.0
    per_ex = F.cross_entropy(lg, lb, reduction="none")
    weights = torch.where(
        lb == pos_token_id,
        torch.full_like(per_ex, pos_weight),
        torch.ones_like(per_ex),
    )
    return (per_ex * weights).sum() / weights.sum().clamp(min=1.0)


# --- Dispatcher ---------------------------------------------------------------

def make_loss_fn(spec: LossConfigSpec, pos_token_id: int):
    """
    Return a callable (logits, labels) -> scalar loss.

    `pos_token_id` is the token ID for "Yes" (positive class) - needed for
    weighted_ce. It's ignored by other losses.
    """
    name = spec.name
    if name == "ce":
        return lambda logits, labels: cross_entropy_loss(logits, labels)
    if name == "label_smoothing":
        s = float(spec.label_smoothing)
        return lambda logits, labels: label_smoothing_loss(logits, labels, s)
    if name == "focal":
        g = float(spec.focal_gamma)
        return lambda logits, labels: focal_loss(logits, labels, g)
    if name == "weighted_ce":
        w = float(spec.pos_weight)
        return lambda logits, labels: weighted_ce_loss(logits, labels, w, pos_token_id)
    raise ValueError(f"Unknown loss name: {name}")
