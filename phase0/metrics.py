"""Evaluation metrics beyond top-1 accuracy.

- **Expected calibration error (ECE)**: how far the model's stated
  confidence is from its actual accuracy. Predictions are binned by
  confidence (max softmax probability); ECE is the accuracy-vs-confidence
  gap averaged over bins, weighted by bin population. A model that says
  "90% sure" and is right 90% of the time has ECE ~= 0. Distilled
  students inherit the teacher's softened posterior and are typically
  better calibrated than hard-label baselines.
- **Reliability diagram**: the per-bin data behind ECE (confidence on x,
  accuracy on y); a perfectly calibrated model lies on the diagonal.
- **Teacher-student agreement**: fraction of test clips where student and
  teacher pick the same class — measures *fidelity to the teacher*, which
  is distinct from accuracy (they can agree on a wrong answer).
- **Mean KL to teacher**: average KL divergence from the teacher's
  posterior to the student's at temperature 1 — the distribution-level
  version of agreement.
- **Confusion matrix**: counts of (true class, predicted class) pairs;
  shows *which* errors remain, not just how many.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    conf, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += (mask.float().mean()
                      * (correct[mask].mean() - conf[mask].mean()).abs()).item()
    return total


def reliability_bins(probs: torch.Tensor, labels: torch.Tensor,
                     n_bins: int = 15) -> dict[str, list[float]]:
    """Per-bin (confidence, accuracy, weight) triples for plotting."""
    conf, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    out: dict[str, list[float]] = {"confidence": [], "accuracy": [], "weight": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            out["confidence"].append(conf[mask].mean().item())
            out["accuracy"].append(correct[mask].mean().item())
            out["weight"].append(mask.float().mean().item())
    return out


def agreement(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> float:
    return (student_logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean().item()


def mean_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> float:
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    ).item()


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor,
                     num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels.numpy(), preds.numpy()):
        matrix[t, p] += 1
    return matrix
