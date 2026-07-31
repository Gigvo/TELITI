"""Shared metric computation — MVP_PLAN.md steps 1.3, 2.1, 4.2.

Every model in this project reports through here, so baseline, transformer and fused
numbers are always computed the same way and are directly comparable.

## Why PR-AUC and not accuracy

EMSCAD is 4.83% fraud. A model that predicts "real" for every posting scores **95.2%
accuracy** and catches zero scams. Accuracy is not merely a weak metric at this ratio,
it is actively misleading, so `evaluate()` reports it with `accuracy_is_misleading`
alongside and the report templates say so in words.

Average precision (PR-AUC) is the primary metric because it summarises performance
across all thresholds on the *positive* class, which is the rare one we care about.
ROC-AUC is reported too but flatters imbalanced classifiers — a large true-negative
pool makes the false-positive rate look small even when precision is poor.

## Threshold selection

`best_f1_threshold` maximises F1 on the fraud class. That is the right default for
comparing models, but it is NOT how the shipped Rendah/Sedang/Tinggi boundaries get
chosen — those come from a precision target (step 3.2), because a false positive
against a real company is more costly to us than a miss (concept paper section 3.6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    """Evaluation of one model on one split."""

    name: str
    n: int
    n_positive: int
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    accuracy: float
    accuracy_is_misleading: bool
    brier: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def majority_class_accuracy(self) -> float:
        """What "predict everything is real" would score."""
        return 1.0 - (self.n_positive / self.n) if self.n else float("nan")


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximising F1 on the positive (fraud) class."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more element than thresholds.
    denominator = precision[:-1] + recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0, 2 * precision[:-1] * recall[:-1] / denominator, 0.0)
    if not len(f1):
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def threshold_for_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float
) -> float | None:
    """Lowest threshold achieving at least `target_precision` on the fraud class.

    Used in step 3.2 to place the Rendah/Sedang/Tinggi boundaries from a stated
    precision target rather than a round number. Returns None when the target is
    unreachable, which is information, not an error.
    """
    precision, _, thresholds = precision_recall_curve(y_true, y_score)
    viable = [
        threshold
        for threshold, p in zip(thresholds, precision[:-1])
        if p >= target_precision
    ]
    return float(min(viable)) if viable else None


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> Metrics:
    """Compute the full metric set for one model's scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) != len(y_score):
        raise ValueError(f"{len(y_true)} labels but {len(y_score)} scores")
    if len(set(y_true.tolist())) < 2:
        raise ValueError(
            f"{name}: y_true contains a single class; every metric here is undefined."
        )

    if threshold is None:
        threshold = best_f1_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_positive = int(y_true.sum())
    accuracy = float((y_pred == y_true).mean())

    notes: list[str] = []
    majority = 1.0 - n_positive / len(y_true)
    if accuracy <= majority + 1e-9:
        notes.append(
            f"accuracy {accuracy:.4f} is no better than always predicting 'real' "
            f"({majority:.4f})"
        )
    if tp == 0:
        notes.append("model caught zero fraud at this threshold")

    return Metrics(
        name=name,
        n=len(y_true),
        n_positive=n_positive,
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
        accuracy=accuracy,
        accuracy_is_misleading=True,
        brier=float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        notes=notes,
    )


def metrics_table(results: list[Metrics]) -> str:
    """Markdown comparison table, best PR-AUC first."""
    ordered = sorted(results, key=lambda m: m.pr_auc, reverse=True)
    lines = [
        "| model | PR-AUC | ROC-AUC | precision | recall | F1 | thresh | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in ordered:
        lines.append(
            f"| {m.name} | **{m.pr_auc:.4f}** | {m.roc_auc:.4f} | {m.precision:.4f} | "
            f"{m.recall:.4f} | {m.f1:.4f} | {m.threshold:.3f} | "
            f"{m.false_positives} | {m.false_negatives} |"
        )
    return "\n".join(lines)
