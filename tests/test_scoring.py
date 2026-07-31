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


def test_rules_only_push_toward_risk(legit_text):
    """A clean ad gets no bonus. Its score is whatever the model said.

    A rule layer able to RAISE a score would let a well-formatted scam suppress a
    correct model warning.
    """
    breakdown = compute_score(0.3, evaluate(legit_text))
    assert breakdown.rule_shift_applied >= 0.0
    assert breakdown.fused_probability >= 0.3


def test_fired_rules_lower_the_score(scam_text):
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


def test_worst_case_shift_reaches_but_never_passes_the_ceiling():
    """Every rule firing at full severity must land exactly at the budget.

    Weights sum to MAX_TOTAL_RULE_SHIFT, so this case is not clipped — it is the
    guarantee holding by construction rather than by clamping.
    """
    breakdown = compute_score(0.5, _all_rules_maxed())
    assert breakdown.rule_shift_raw == pytest.approx(MAX_TOTAL_RULE_SHIFT)
    assert breakdown.rule_shift_applied == pytest.approx(MAX_TOTAL_RULE_SHIFT)
    assert not breakdown.was_clipped


def test_aggregate_shift_is_clipped_if_weights_ever_go_over_budget(monkeypatch):
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


def test_contribution_points_are_score_scale():
    contributions = {"payment_request_id": 0.09}
    assert contribution_points(contributions) == {"payment_request_id": 9.0}


# ===========================================================================
# Labels
# ===========================================================================


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0, RiskLabel.TINGGI), (39, RiskLabel.TINGGI), (40, RiskLabel.SEDANG),
     (69, RiskLabel.SEDANG), (70, RiskLabel.RENDAH), (100, RiskLabel.RENDAH)],
)
def test_label_boundaries(score, expected):
    assert label_for_score(score) == expected


def test_label_is_monotonic_in_score():
    order = {RiskLabel.TINGGI: 0, RiskLabel.SEDANG: 1, RiskLabel.RENDAH: 2}
    ranks = [order[label_for_score(s)] for s in range(101)]
    assert ranks == sorted(ranks)


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


def test_real_item_requires_independent_labels(tmp_path):
    bad = {k: v for k, v in REAL_LINE.items() if k not in ("label_a", "label_b")}
    with pytest.raises(EvalSetError, match="kappa"):
        load_eval_set(write_jsonl(tmp_path / "f.jsonl", [bad]))


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
