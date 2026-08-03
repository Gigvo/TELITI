"""Calibration correctness — MVP_PLAN.md step 2.3.

Tested with synthetic distributions of known calibration, so the metrics are checked
against cases where the right answer is derivable rather than merely plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.calibration import (
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    probabilities_to_margin,
    reliability_table,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def perfectly_calibrated(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that mean exactly what they say: p == P(y=1)."""
    probabilities = rng.uniform(0.0, 1.0, n)
    labels = (rng.uniform(0.0, 1.0, n) < probabilities).astype(int)
    return labels, probabilities


# ===========================================================================
# Brier
# ===========================================================================


def test_brier_is_zero_for_perfect_confident_predictions():
    y = np.array([0, 1, 0, 1])
    assert brier_score(y, y.astype(float)) == pytest.approx(0.0)


def test_brier_is_one_for_confidently_wrong_predictions():
    y = np.array([0, 1, 0, 1])
    assert brier_score(y, 1.0 - y) == pytest.approx(1.0)


def test_brier_of_always_half_is_a_quarter():
    y = np.array([0, 1, 0, 1])
    assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)


# ===========================================================================
# ECE
# ===========================================================================


def test_ece_near_zero_on_well_calibrated_predictions(rng):
    labels, probabilities = perfectly_calibrated(20000, rng)
    report = expected_calibration_error(labels, probabilities)
    assert report.ece < 0.02


def test_ece_is_large_when_the_model_is_overconfident():
    """The failure mode calibration exists to catch: says 0.95, right 50% of the time."""
    labels = np.array([1] * 50 + [0] * 50)
    probabilities = np.full(100, 0.95)
    report = expected_calibration_error(labels, probabilities)
    assert report.ece > 0.4


def test_ece_detects_underconfidence_too():
    labels = np.ones(100, dtype=int)
    report = expected_calibration_error(labels, np.full(100, 0.1))
    assert report.ece > 0.8


def test_bins_cover_the_unit_interval_and_count_every_sample(rng):
    labels, probabilities = perfectly_calibrated(500, rng)
    report = expected_calibration_error(labels, probabilities, n_bins=10)
    assert len(report.bins) == 10
    assert sum(b["count"] for b in report.bins) == 500


def test_boundary_probabilities_are_not_dropped():
    """p == 0.0 and p == 1.0 must land in a bin, not vanish."""
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.0, 1.0, 0.0, 1.0])
    report = expected_calibration_error(labels, probabilities)
    assert sum(b["count"] for b in report.bins) == 4


def test_empty_bins_are_reported_not_hidden():
    """A model whose predictions cluster in two bins is saying something; hiding the
    empty bins conceals it."""
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.05, 0.95, 0.95])
    report = expected_calibration_error(labels, probabilities, n_bins=10)
    assert any(b["count"] == 0 for b in report.bins)
    assert len(report.bins) == 10


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="labels but"):
        expected_calibration_error(np.array([0, 1]), np.array([0.5]))


def test_max_calibration_error_is_the_worst_bin():
    labels = np.array([1] * 50 + [0] * 50)
    report = expected_calibration_error(labels, np.full(100, 0.95))
    assert report.max_calibration_error >= report.ece


# ===========================================================================
# Platt scaling
# ===========================================================================


def test_platt_fixes_a_miscalibrated_but_well_ranked_model(rng):
    """The exact situation this module exists for.

    Scores separate the classes cleanly (high PR-AUC) but are wildly overconfident.
    Ranking is untouched by calibration; only the numbers change.
    """
    n = 4000
    labels = rng.integers(0, 2, n)
    # Well-separated margins, then squashed into badly overconfident probabilities.
    margins = np.where(labels == 1, rng.normal(2.0, 1.0, n), rng.normal(-2.0, 1.0, n))
    overconfident = 1.0 / (1.0 + np.exp(-margins * 5.0))

    before = expected_calibration_error(labels, overconfident)
    calibrator = PlattCalibrator().fit(margins, labels)
    after = expected_calibration_error(labels, calibrator.transform(margins))

    assert after.ece < before.ece
    assert after.ece < 0.05


def test_calibration_preserves_ranking(rng):
    """Platt scaling is monotone, so PR-AUC must be unchanged.

    If this ever fails, the calibrator is doing something other than calibrating.
    """
    from sklearn.metrics import average_precision_score

    n = 2000
    labels = rng.integers(0, 2, n)
    margins = np.where(labels == 1, rng.normal(1.5, 1.0, n), rng.normal(-1.5, 1.0, n))

    calibrator = PlattCalibrator().fit(margins, labels)
    calibrated = calibrator.transform(margins)

    assert average_precision_score(labels, margins) == pytest.approx(
        average_precision_score(labels, calibrated), abs=1e-9
    )


def test_calibrator_output_is_a_valid_probability(rng):
    n = 500
    labels = rng.integers(0, 2, n)
    margins = rng.normal(0.0, 3.0, n)
    calibrated = PlattCalibrator().fit(margins, labels).transform(margins)
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_transform_before_fit_is_rejected():
    with pytest.raises(RuntimeError, match="fit must be called"):
        PlattCalibrator().transform(np.array([0.5]))


def test_single_class_calibration_is_rejected():
    """Calibrating on one class would produce a constant, not a calibration."""
    with pytest.raises(ValueError, match="single class"):
        PlattCalibrator().fit(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))


def test_calibrator_round_trips_through_a_dict(rng):
    n = 300
    labels = rng.integers(0, 2, n)
    margins = rng.normal(0.0, 2.0, n)
    original = PlattCalibrator().fit(margins, labels)
    restored = PlattCalibrator.from_dict(original.to_dict())

    np.testing.assert_allclose(original.transform(margins), restored.transform(margins))


# ===========================================================================
# Helpers
# ===========================================================================


def test_probabilities_to_margin_inverts_the_sigmoid():
    probabilities = np.array([0.1, 0.5, 0.9])
    recovered = 1.0 / (1.0 + np.exp(-probabilities_to_margin(probabilities)))
    np.testing.assert_allclose(recovered, probabilities, atol=1e-6)


def test_probabilities_to_margin_survives_saturation():
    """A saturated 0.0 or 1.0 must not become infinite."""
    margins = probabilities_to_margin(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(margins))


def test_reliability_table_renders_all_bins(rng):
    labels, probabilities = perfectly_calibrated(200, rng)
    table = reliability_table(expected_calibration_error(labels, probabilities, n_bins=5))
    assert table.count("\n") == 6  # header + separator + 5 bins
    assert "mean predicted" in table
