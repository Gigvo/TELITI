"""Metric correctness — MVP_PLAN.md step 1.3.

Getting a metric subtly wrong is worse than having no metric: it produces a confident
number that nobody can tell is broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.metrics import (
    best_f1_threshold,
    evaluate,
    metrics_table,
    threshold_for_precision,
)


@pytest.fixture
def imbalanced():
    """96 legitimate, 4 fraud — roughly EMSCAD's ratio."""
    y = np.array([0] * 96 + [1] * 4)
    return y


def test_perfect_scores_give_perfect_metrics(imbalanced):
    m = evaluate("perfect", imbalanced, imbalanced.astype(float))
    assert m.pr_auc == pytest.approx(1.0)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.false_negatives == 0


def test_majority_class_predictor_is_exposed(imbalanced):
    """The whole reason accuracy is not the headline metric.

    A model outputting a constant scores 96% accuracy and catches zero fraud.
    """
    scores = np.full(len(imbalanced), 0.01)
    m = evaluate("constant", imbalanced, scores)
    assert m.true_positives == 0 or m.recall == pytest.approx(1.0)
    assert m.pr_auc < 0.10, "PR-AUC must expose a useless model"
    assert m.majority_class_accuracy == pytest.approx(0.96)


def test_notes_flag_a_model_that_catches_nothing(imbalanced):
    scores = np.where(imbalanced == 1, 0.0, 1.0).astype(float)  # perfectly inverted
    m = evaluate("inverted", imbalanced, scores)
    assert m.pr_auc < 0.5
    assert any("zero fraud" in n or "no better" in n for n in m.notes)


def test_single_class_input_is_rejected():
    """Every metric here is undefined on one class; failing loudly beats NaN."""
    with pytest.raises(ValueError, match="single class"):
        evaluate("bad", np.zeros(10, dtype=int), np.random.rand(10))


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="labels but"):
        evaluate("bad", np.array([0, 1, 0]), np.array([0.1, 0.2]))


def test_roc_auc_flatters_imbalanced_data_more_than_pr_auc(imbalanced):
    """Documents WHY PR-AUC is primary: with a large true-negative pool, ROC-AUC
    stays high even when precision is poor."""
    rng = np.random.default_rng(0)
    scores = np.where(imbalanced == 1, rng.uniform(0.5, 0.9, len(imbalanced)),
                      rng.uniform(0.0, 0.7, len(imbalanced)))
    m = evaluate("mediocre", imbalanced, scores)
    assert m.roc_auc > m.pr_auc


def test_best_f1_threshold_is_within_score_range(imbalanced):
    rng = np.random.default_rng(1)
    scores = rng.random(len(imbalanced))
    t = best_f1_threshold(imbalanced, scores)
    assert scores.min() <= t <= scores.max()


def test_threshold_for_precision_achieves_the_target(imbalanced):
    scores = np.where(imbalanced == 1, 0.9, 0.1).astype(float)
    t = threshold_for_precision(imbalanced, scores, 0.99)
    assert t is not None
    predicted = (scores >= t).astype(int)
    hits = predicted.sum()
    assert hits > 0
    assert (predicted * imbalanced).sum() / hits >= 0.99


def test_threshold_for_precision_returns_none_when_unreachable(imbalanced):
    """Unreachable is information, not an error."""
    rng = np.random.default_rng(2)
    assert threshold_for_precision(imbalanced, rng.random(len(imbalanced)), 0.999) is None


def test_explicit_threshold_is_respected(imbalanced):
    scores = np.where(imbalanced == 1, 0.9, 0.1).astype(float)
    assert evaluate("t", imbalanced, scores, threshold=0.5).threshold == 0.5


def test_confusion_counts_sum_to_n(imbalanced):
    rng = np.random.default_rng(3)
    m = evaluate("m", imbalanced, rng.random(len(imbalanced)))
    assert m.true_negatives + m.false_positives + m.false_negatives + m.true_positives == m.n


def test_metrics_table_orders_by_pr_auc(imbalanced):
    strong = evaluate("strong", imbalanced, imbalanced.astype(float))
    rng = np.random.default_rng(4)
    weak = evaluate("weak", imbalanced, rng.random(len(imbalanced)))
    table = metrics_table([weak, strong])
    assert table.index("strong") < table.index("weak")
