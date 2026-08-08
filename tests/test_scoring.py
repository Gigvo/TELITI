"""Scoring invariants and the synthetic-data guard — MVP_PLAN.md steps 1.5 / 3.1."""

from __future__ import annotations

import json

import pytest

from api.ingest import ingest
from api.rules.engine import default_engine
from api.scoring import compute_score, contribution_points, label_for_score
from api.schemas import RiskLabel
from ml.eval_set import EvalSetError, cohens_kappa, load_eval_set
from ml.feature_contract import (
    MAX_TOTAL_RULE_SHIFT,
    PER_RULE_CONTRIBUTION_CAP,
    RULE_FEATURE_ORDER,
    FeatureContractViolation,
)
from ml.rule_weights import RULE_WEIGHTS, assert_weights_sane, weight_vector


def evaluate(text: str):
    return default_engine().evaluate(ingest(text))


@pytest.fixture
def rules_enabled(monkeypatch):
    """Re-enable the rule layer for tests of the fusion MECHANISM.

    The layer ships disabled (`api/scoring.py::RULE_LAYER_ENABLED`) because it made
    the product measurably worse on real Indonesian data. The arithmetic still has to
    be correct, though — the flag may be flipped back, and a broken mechanism hiding
    behind a disabled flag is worse than a broken mechanism in plain sight.
    """
    import api.scoring as scoring

    monkeypatch.setattr(scoring, "RULE_LAYER_ENABLED", True)
    return scoring


# ===========================================================================
# Weights
# ===========================================================================


def test_every_feature_has_a_weight():
    assert set(RULE_WEIGHTS) == set(RULE_FEATURE_ORDER)


def test_weight_vector_follows_canonical_order():
    assert weight_vector() == [RULE_WEIGHTS[n] for n in RULE_FEATURE_ORDER]


def test_no_single_rule_exceeds_the_cap():
    """The concept paper 3.3 guarantee, enforced on the actual numbers."""
    for name, weight in RULE_WEIGHTS.items():
        assert 0.0 <= weight <= PER_RULE_CONTRIBUTION_CAP, name


def test_weights_respect_the_aggregate_ceiling():
    assert sum(RULE_WEIGHTS.values()) <= MAX_TOTAL_RULE_SHIFT + 1e-9


def test_payment_request_is_the_strongest_signal():
    """Domain ordering: asking a candidate for money is the defining scam marker."""
    assert RULE_WEIGHTS["payment_request_id"] == max(RULE_WEIGHTS.values())


def test_a_negative_weight_is_rejected(monkeypatch):
    import ml.rule_weights as rw

    monkeypatch.setitem(rw.RULE_WEIGHTS, "email_absent", -0.1)
    with pytest.raises(FeatureContractViolation, match="negative"):
        rw.assert_weights_sane()


def test_an_oversized_weight_is_rejected(monkeypatch):
    import ml.rule_weights as rw

    monkeypatch.setitem(rw.RULE_WEIGHTS, "email_absent", 0.9)
    with pytest.raises(FeatureContractViolation, match="dominate"):
        rw.assert_weights_sane()


def test_weights_are_sane_as_shipped():
    assert_weights_sane()


# ===========================================================================
# Score computation
# ===========================================================================


def test_score_matches_the_paper_formula(scam_text):
    """S = round((1 - p_final) * 100), concept paper 3.3."""
    breakdown = compute_score(0.8, evaluate(scam_text))
    assert breakdown.integrity_score == round((1 - breakdown.fused_probability) * 100)


def test_rules_only_push_toward_risk(rules_enabled, legit_text):
    """A clean ad gets no bonus. Its score is whatever the model said.

    A rule layer able to RAISE a score would let a well-formatted scam suppress a
    correct model warning.
    """
    breakdown = compute_score(0.3, evaluate(legit_text))
    assert breakdown.rule_shift_applied >= 0.0
    assert breakdown.fused_probability >= 0.3


def test_fired_rules_lower_the_score(rules_enabled, scam_text):
    with_rules = compute_score(0.5, evaluate(scam_text))
    assert with_rules.fused_probability > 0.5
    assert with_rules.integrity_score < 50


def test_clean_ad_score_equals_the_model_probability(legit_text):
    breakdown = compute_score(0.42, evaluate(legit_text))
    assert breakdown.contributions == {}
    assert breakdown.fused_probability == pytest.approx(0.42)
    assert breakdown.integrity_score == 58


def _all_rules_maxed():
    from api.rules.base import RuleOutcome
    from api.rules.engine import RuleEvaluation
    from api.schemas import RuleCategory

    return RuleEvaluation(
        outcomes={
            name: RuleOutcome(
                feature_id=name, severity=1.0, label_id="x", label_en="x",
                category=RuleCategory.LANGUAGE,
            )
            for name in RULE_FEATURE_ORDER
        }
    )


def test_worst_case_shift_reaches_but_never_passes_the_ceiling(rules_enabled):
    """Every rule firing at full severity must land exactly at the budget.

    Weights sum to MAX_TOTAL_RULE_SHIFT, so this case is not clipped — it is the
    guarantee holding by construction rather than by clamping.
    """
    breakdown = compute_score(0.5, _all_rules_maxed())
    assert breakdown.rule_shift_raw == pytest.approx(MAX_TOTAL_RULE_SHIFT)
    assert breakdown.rule_shift_applied == pytest.approx(MAX_TOTAL_RULE_SHIFT)
    assert not breakdown.was_clipped


def test_aggregate_shift_is_clipped_if_weights_ever_go_over_budget(rules_enabled, monkeypatch):
    """The clamp is the backstop for a future weight change slipping past review."""
    import ml.rule_weights as rw

    monkeypatch.setitem(rw.RULE_WEIGHTS, "payment_request_id", 0.30)
    breakdown = compute_score(0.5, _all_rules_maxed())
    assert breakdown.rule_shift_raw > MAX_TOTAL_RULE_SHIFT
    assert breakdown.rule_shift_applied == pytest.approx(MAX_TOTAL_RULE_SHIFT)
    assert breakdown.was_clipped


def test_probability_stays_in_range():
    for p in (0.0, 0.5, 0.99, 1.0):
        breakdown = compute_score(p, evaluate("Interview via Telegram, transfer biaya administrasi dulu ya."))
        assert 0.0 <= breakdown.fused_probability <= 1.0
        assert 0 <= breakdown.integrity_score <= 100


def test_invalid_model_probability_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        compute_score(1.5, evaluate("Lowongan admin online gaji besar sekali ya."))


def test_unavailable_signals_contribute_nothing():
    """A redacted corpus must not be scored as if it were clean."""
    redacted = evaluate(
        "Customer service representative wanted. Send your resume to "
        "#EMAIL_a1b2c3d4# or visit #URL_deadbeef# for more details."
    )
    assert compute_score(0.5, redacted).contributions == {}


def test_contribution_points_are_score_scale(rules_enabled):
    contributions = {"payment_request_id": 0.09}
    assert contribution_points(contributions) == {"payment_request_id": 9.0}


# ===========================================================================
# Advisory mode — the shipped configuration
# ===========================================================================
#
# Measured on the 195-item Indonesian holdout:
#   model only     PR-AUC 0.9258   5 false positives
#   model + rules  PR-AUC 0.8617  28 false positives
# The rule layer is therefore disabled. These tests pin that behaviour so it cannot
# be switched back on by accident.


def test_rules_do_not_move_the_score_when_disabled(scam_text):
    """The shipped default: the score is the model probability, untouched."""
    breakdown = compute_score(0.5, evaluate(scam_text))
    assert breakdown.rule_layer_enabled is False
    assert breakdown.rule_shift_applied == 0.0
    assert breakdown.fused_probability == pytest.approx(0.5)
    assert breakdown.integrity_score == 50


def test_rules_still_report_what_they_found(scam_text):
    """Disabled means 'does not move the score', not 'stops looking'.

    The findings remain available as advisory evidence — a user is still told the ad
    demands an up-front payment, they are just not told it changed the number.
    """
    breakdown = compute_score(0.5, evaluate(scam_text))
    assert breakdown.contributions, "rules should still evaluate and report"
    assert breakdown.rule_shift_raw > 0.0, "raw shift is still computed"


def test_advisory_contributions_are_reported_as_zero(scam_text):
    """A card reading "−9.5 pts" beside a score those points did not affect would be
    a lie told by the interface."""
    hits = evaluate(scam_text).to_rule_hits()
    assert hits, "expected the paper's scenario to fire rules"
    assert all(h.contribution == 0.0 for h in hits)
    assert contribution_points({"payment_request_id": 0.09}) == {"payment_request_id": 0.0}


def test_api_reports_the_advisory_state(client, scam_text):
    body = client.post("/api/v1/analyze", json={"text": scam_text}).json()
    assert body["rule_layer_enabled"] is False
    assert all(h["contribution"] == 0.0 for h in body["rule_hits"])


# ===========================================================================
# Labels
# ===========================================================================


#: Explicit bounds so these tests check the MAPPING LOGIC rather than whichever
#: numbers `ml/fit_thresholds.py` last produced. The fitted values move whenever the
#: model or the holdout changes; the logic must not.
_BOUNDS = {"tinggi_below": 40, "rendah_at_or_above": 70}


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0, RiskLabel.TINGGI), (39, RiskLabel.TINGGI), (40, RiskLabel.SEDANG),
     (69, RiskLabel.SEDANG), (70, RiskLabel.RENDAH), (100, RiskLabel.RENDAH)],
)
def test_label_boundaries(score, expected):
    assert label_for_score(score, _BOUNDS) == expected


def test_label_is_monotonic_in_score():
    order = {RiskLabel.TINGGI: 0, RiskLabel.SEDANG: 1, RiskLabel.RENDAH: 2}
    ranks = [order[label_for_score(s, _BOUNDS)] for s in range(101)]
    assert ranks == sorted(ranks)


def test_shipped_thresholds_are_fitted_not_placeholders():
    """§3.3 requires derived boundaries, not round numbers.

    If this fails, `artifacts/thresholds.json` is missing or malformed and the
    service has silently fallen back to the arbitrary placeholders the concept paper
    argues against.
    """
    from api.scoring import load_thresholds

    bounds, fitted = load_thresholds()
    assert fitted, "run: python ml/fit_thresholds.py"
    assert 0 <= bounds["tinggi_below"] < bounds["rendah_at_or_above"] <= 100


def test_fitted_thresholds_leave_a_usable_sedang_band():
    """A Sedang band one point wide would make the middle label meaningless.

    This is not hypothetical: with the EMSCAD calibrator the bands came out as
    Tinggi<98 / Sedang 98..98 / Rendah>=99, because the model's probabilities were
    all crushed near zero on Indonesian input.
    """
    from api.scoring import load_thresholds

    bounds, _ = load_thresholds()
    width = bounds["rendah_at_or_above"] - bounds["tinggi_below"]
    assert width >= 10, f"Sedang band is only {width} points wide"


def test_malformed_threshold_file_falls_back_visibly(tmp_path):
    """A broken file must not silently become arbitrary numbers."""
    from api.scoring import load_thresholds

    bad = tmp_path / "thresholds.json"
    bad.write_text("{not json", encoding="utf-8")
    bounds, fitted = load_thresholds(str(bad))
    assert fitted is False
    assert bounds == {"tinggi_below": 40, "rendah_at_or_above": 70}


def test_inverted_thresholds_are_rejected(tmp_path):
    from api.scoring import load_thresholds

    bad = tmp_path / "thresholds.json"
    bad.write_text(json.dumps({"tinggi_below": 90, "rendah_at_or_above": 20}), encoding="utf-8")
    assert load_thresholds(str(bad))[1] is False


# ===========================================================================
# Eval-set loading and the synthetic guard
# ===========================================================================


FIXTURE_LINE = {
    "id": "SYNTHETIC-0001",
    "text": "Lowongan admin online, gaji besar, hubungi via Telegram sekarang juga.",
    "label": 1,
    "source_url": "synthetic://fixture",
    "source_type": "synthetic",
    "channel": "telegram",
    "annotator_a": "fixture",
    "annotator_b": "fixture",
    "collected_at": "2026-07-31",
    "synthetic": True,
}

REAL_LINE = {
    "id": "id-holdout-0001",
    "text": "Lowongan Backend Engineer di PT Teknologi Nusantara, kirim lamaran ke hrd@teknologinusantara.co.id",
    "label": 0,
    "source_url": "https://example.co.id/karier",
    "source_type": "jobstreet",
    "channel": "job_board",
    "annotator_a": "ivan",
    "annotator_b": "revo",
    "label_a": 0,
    "label_b": 0,
    "collected_at": "2026-07-31",
}


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def test_synthetic_file_is_flagged(tmp_path):
    path = write_jsonl(tmp_path / "f.jsonl", [FIXTURE_LINE])
    assert load_eval_set(path).is_synthetic


def test_real_file_is_not_flagged(tmp_path):
    path = write_jsonl(tmp_path / "f.jsonl", [REAL_LINE])
    assert not load_eval_set(path).is_synthetic


def test_one_synthetic_row_taints_the_whole_file(tmp_path):
    """A mixed-provenance file is not partly trustworthy — no number from it is safe."""
    path = write_jsonl(tmp_path / "f.jsonl", [REAL_LINE, FIXTURE_LINE])
    assert load_eval_set(path).is_synthetic


def test_synthetic_data_refuses_to_produce_a_result(tmp_path):
    """The guard that stops a fabricated number reaching a slide."""
    eval_set = load_eval_set(write_jsonl(tmp_path / "f.jsonl", [FIXTURE_LINE]))
    with pytest.raises(EvalSetError, match="Refusing to"):
        eval_set.require_real("report metrics")


def test_synthetic_data_is_allowed_with_an_explicit_opt_in(tmp_path):
    eval_set = load_eval_set(write_jsonl(tmp_path / "f.jsonl", [FIXTURE_LINE]))
    eval_set.require_real("plumbing check", allow_synthetic=True)


def test_real_data_needs_no_opt_in(tmp_path):
    load_eval_set(write_jsonl(tmp_path / "f.jsonl", [REAL_LINE])).require_real("report")


def test_id_and_synthetic_flag_must_agree(tmp_path):
    bad = dict(FIXTURE_LINE, id="id-holdout-0009")
    with pytest.raises(EvalSetError, match="disagree"):
        load_eval_set(write_jsonl(tmp_path / "f.jsonl", [bad]))


def test_single_pass_labelling_is_accepted(tmp_path):
    """The dataset is labelled once from provenance, not by two annotators.

    Independent double-annotation was the original design; it was dropped when the
    holdout was collected, and `ml/validate_eval_set.py` documents why kappa is
    reported as n/a rather than faked. The loader must accept that shape.
    """
    single = {k: v for k, v in REAL_LINE.items() if k not in ("label_a", "label_b")}
    result = load_eval_set(write_jsonl(tmp_path / "f.jsonl", [single]))
    assert len(result) == 1
    assert result.items[0].label_a is None


def test_item_with_no_provenance_at_all_is_rejected(tmp_path):
    """A public link OR a note explaining a privately received message. Neither
    means the item cannot be traced back to anything."""
    orphan = {k: v for k, v in REAL_LINE.items() if k not in ("label_a", "label_b")}
    orphan["source_url"] = ""
    orphan.pop("notes", None)
    with pytest.raises(EvalSetError, match="provenance"):
        load_eval_set(write_jsonl(tmp_path / "f.jsonl", [orphan]))


def test_personally_received_message_needs_only_notes(tmp_path):
    """WhatsApp ads a team member received have no URL — these are among the most
    valuable items, being exactly the input the product is built for."""
    received = {k: v for k, v in REAL_LINE.items() if k not in ("label_a", "label_b")}
    received["source_url"] = None
    received["notes"] = "User-provided WhatsApp job ad, received personally."
    assert len(load_eval_set(write_jsonl(tmp_path / "f.jsonl", [received]))) == 1


def test_disagreement_requires_a_resolver(tmp_path):
    bad = dict(REAL_LINE, label_a=0, label_b=1)
    with pytest.raises(EvalSetError, match="resolved_by"):
        load_eval_set(write_jsonl(tmp_path / "f.jsonl", [bad]))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(EvalSetError, match="duplicate id"):
        load_eval_set(write_jsonl(tmp_path / "f.jsonl", [REAL_LINE, REAL_LINE]))


def test_duplicate_text_is_warned_about(tmp_path):
    """Scoring the same ad twice looks like two independent successes."""
    second = dict(REAL_LINE, id="id-holdout-0002")
    result = load_eval_set(write_jsonl(tmp_path / "f.jsonl", [REAL_LINE, second]))
    assert any("duplicates the text" in w for w in result.warnings)


def test_unknown_field_is_rejected(tmp_path):
    bad = dict(REAL_LINE, labl=1)
    with pytest.raises(EvalSetError, match="unknown field"):
        load_eval_set(write_jsonl(tmp_path / "f.jsonl", [bad]))


def test_missing_file_gives_a_useful_message(tmp_path):
    with pytest.raises(EvalSetError, match="not found"):
        load_eval_set(tmp_path / "nope.jsonl")


# ===========================================================================
# Cohen's kappa
# ===========================================================================


def test_kappa_is_one_on_perfect_agreement():
    assert cohens_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)


def test_kappa_is_zero_at_chance():
    assert cohens_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)


def test_kappa_is_negative_on_systematic_disagreement():
    assert cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0]) < 0


def test_kappa_undefined_when_one_class_used_throughout():
    """Two annotators who both label everything 'legitimate' agree 100% of the time
    and have demonstrated nothing. Raw agreement would report 1.0."""
    result = cohens_kappa([0, 0, 0, 0], [0, 0, 0, 0])
    assert result != result  # NaN
