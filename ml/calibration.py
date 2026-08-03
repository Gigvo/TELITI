"""Probability calibration — MVP_PLAN.md step 2.3.

A classifier can rank well and still be badly calibrated: it may separate scams from
real postings almost perfectly while reporting "0.9" on cases that are right only
60% of the time. Ranking metrics like PR-AUC cannot see this, because they only care
about order.

That matters here more than usual. The concept paper's entire premise is an
*Integrity Score* the user is asked to interpret — "18/100" has to mean something.
Without calibration it is an arbitrary monotone transform of an arbitrary number,
and section 3.3's claim that the score is a calibrated confidence is unsupported.

## Platt scaling

Fits a one-dimensional logistic regression on the model's logit margin:

    p_calibrated = sigmoid(a * margin + b)

where `margin = logit_fraud - logit_real`. Two parameters, fitted on a held-out
split the model never trained on and never selected against — that is why
`prepare_data.py` produces a separate `calib` split rather than reusing `val`.
Calibrating on data used for checkpoint selection produces probabilities that look
calibrated there and are overconfident everywhere else.

## Metrics

- **Brier score**: mean squared error of the probabilities. Lower is better.
  Sensitive to both calibration and discrimination.
- **ECE** (Expected Calibration Error): bin predictions by confidence, compare the
  average predicted probability in each bin with the observed frequency, and average
  the gaps weighted by bin size. This is the number the gate is written against
  (<= 0.05) because it directly answers "when this system says 0.8, is it right 80%
  of the time?"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationReport:
    brier: float
    ece: float
    max_calibration_error: float
    bins: list[dict]

    def as_dict(self) -> dict:
        return {
            "brier": self.brier,
            "ece": self.ece,
            "max_calibration_error": self.max_calibration_error,
            "bins": self.bins,
        }


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return float(np.mean((probabilities - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> CalibrationReport:
    """Bin by predicted probability and compare predicted vs observed frequency.

    Equal-width bins over [0, 1]. Empty bins contribute nothing and are reported
    with `count: 0` rather than silently dropped — a model whose predictions all
    land in two bins is telling you something, and hiding the empty ones conceals it.
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities):
        raise ValueError(f"{len(y_true)} labels but {len(probabilities)} probabilities")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    worst = 0.0
    bins: list[dict] = []

    for lower, upper in zip(edges[:-1], edges[1:]):
        # Include the right edge in the final bin so p == 1.0 is not dropped.
        in_bin = (probabilities > lower) & (probabilities <= upper)
        if lower == 0.0:
            in_bin |= probabilities == 0.0

        count = int(in_bin.sum())
        if count == 0:
            bins.append(
                {
                    "lower": round(float(lower), 3),
                    "upper": round(float(upper), 3),
                    "count": 0,
                    "mean_predicted": None,
                    "observed_frequency": None,
                    "gap": None,
                }
            )
            continue

        mean_predicted = float(probabilities[in_bin].mean())
        observed = float(y_true[in_bin].mean())
        gap = abs(mean_predicted - observed)

        ece += (count / total) * gap
        worst = max(worst, gap)
        bins.append(
            {
                "lower": round(float(lower), 3),
                "upper": round(float(upper), 3),
                "count": count,
                "mean_predicted": round(mean_predicted, 4),
                "observed_frequency": round(observed, 4),
                "gap": round(gap, 4),
            }
        )

    return CalibrationReport(
        brier=brier_score(y_true, probabilities),
        ece=float(ece),
        max_calibration_error=float(worst),
        bins=bins,
    )


class PlattCalibrator:
    """Two-parameter logistic calibration on a decision margin.

    Deliberately not `sklearn.calibration.CalibratedClassifierCV`: that wraps an
    estimator and re-runs cross-validation, whereas we already have out-of-sample
    scores from a model that is expensive to re-run. Fitting on the scores directly
    is both cheaper and easier to audit.
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0
        self.fitted: bool = False

    def fit(self, margins: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        margins = np.asarray(margins, dtype=float).reshape(-1, 1)
        y_true = np.asarray(y_true, dtype=int)
        if len(set(y_true.tolist())) < 2:
            raise ValueError("Cannot calibrate: the split contains a single class.")

        # No class_weight here. Platt scaling must learn the TRUE base rate of the
        # calibration split; re-balancing would teach it a prior that does not exist
        # in the data, producing probabilities that are wrong in a new way.
        model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        model.fit(margins, y_true)

        self.a = float(model.coef_[0][0])
        self.b = float(model.intercept_[0])
        self.fitted = True
        return self

    def transform(self, margins: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PlattCalibrator.fit must be called before transform.")
        margins = np.asarray(margins, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.a * margins + self.b)))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, data: dict) -> "PlattCalibrator":
        calibrator = cls()
        calibrator.a = float(data["a"])
        calibrator.b = float(data["b"])
        calibrator.fitted = bool(data.get("fitted", True))
        return calibrator


def probabilities_to_margin(probabilities: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Recover a logit margin from probabilities.

    Used when only probabilities were stored rather than raw logits. Clipped away
    from 0 and 1 so a saturated prediction does not become infinite.
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(probabilities / (1.0 - probabilities))


def reliability_table(report: CalibrationReport) -> str:
    """Markdown reliability diagram.

    A table rather than a plot: it needs no matplotlib dependency, it is readable in
    a terminal and in a pull request, and the numbers are exact rather than eyeballed
    off an axis.
    """
    lines = [
        "| confidence bin | n | mean predicted | observed | gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report.bins:
        if row["count"] == 0:
            lines.append(f"| {row['lower']:.1f}–{row['upper']:.1f} | 0 | — | — | — |")
            continue
        lines.append(
            f"| {row['lower']:.1f}–{row['upper']:.1f} | {row['count']} | "
            f"{row['mean_predicted']:.4f} | {row['observed_frequency']:.4f} | "
            f"{row['gap']:.4f} |"
        )
    return "\n".join(lines)
