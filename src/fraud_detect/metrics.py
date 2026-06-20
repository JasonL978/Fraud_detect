"""Evaluation metrics for highly imbalanced fraud detection.

The headline number is recall@1%FPR: "how much fraud do we catch while blocking
only 1% of legitimate transactions?" Accuracy and ROC-AUC are misleading at
~0.13% prevalence, so they are not the primary metrics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(
    y_true: Sequence[int],
    scores: Sequence[float],
    fpr_target: float = 0.01,
) -> float:
    """Recall (TPR) at the highest operating point whose FPR <= ``fpr_target``.

    Returns 0.0 if no threshold achieves an FPR at or below the target.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    allowed = fpr <= fpr_target
    if not allowed.any():
        return 0.0
    return float(tpr[allowed].max())


def pr_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve (a.k.a. average precision)."""
    return float(average_precision_score(y_true, scores))


def evaluate(y_true: Sequence[int], scores: Sequence[float]) -> dict:
    """Return the metric bundle. PR-AUC and recall@1%FPR are the headline."""
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    return {
        "pr_auc": pr_auc(y_true, scores),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
        "roc_auc": float(roc_auc_score(y_true, scores)),  # secondary only
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_rate": float(n_pos / len(y_true)) if len(y_true) else 0.0,
    }
