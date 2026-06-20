"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from fraud_detect.metrics import evaluate, pr_auc, recall_at_fpr


def test_recall_at_fpr_perfect_separation():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert recall_at_fpr(y, scores, fpr_target=0.0) == 1.0


def test_recall_at_fpr_known_imperfect_case():
    # 5 neg in [0.0,0.4], 5 pos in {0.05, 0.6..0.9}. One positive (0.05) is
    # buried below the negatives. At FPR=0 (threshold above all negatives) we
    # catch the 4 high positives -> recall 0.8.
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.05, 0.6, 0.7, 0.8, 0.9]
    assert recall_at_fpr(y, scores, fpr_target=0.0) == pytest.approx(0.8)


def test_recall_at_fpr_unreachable_target_returns_zero():
    # All scores identical -> any positive threshold yields FPR 1.0; nothing
    # achieves FPR <= 0 except the trivial (0,0) point -> recall 0.
    y = [0, 1]
    scores = [0.5, 0.5]
    assert recall_at_fpr(y, scores, fpr_target=0.0) == 0.0


def test_pr_auc_perfect_is_one():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_evaluate_bundle_shape():
    y = [0, 0, 0, 1]
    scores = [0.1, 0.2, 0.3, 0.9]
    m = evaluate(y, scores)
    assert set(m) == {
        "pr_auc", "recall_at_1pct_fpr", "roc_auc", "n_pos", "n_neg", "base_rate",
    }
    assert m["n_pos"] == 1 and m["n_neg"] == 3
    assert m["base_rate"] == pytest.approx(0.25)
