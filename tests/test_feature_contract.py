"""Tests for the feature-availability contract — MVP_PLAN.md section 1.1.

These exist because the failure they prevent is invisible: a model that leaks
EMSCAD-only metadata trains fine, scores beautifully, and then performs at chance
on real user input. Nothing errors. The only defence is a test that fails first.
"""

from __future__ import annotations

import pytest

from ml.feature_contract import (
    EMSCAD_COLUMNS,
    EMSCAD_DERIVABLE_FEATURES,
    INDONESIAN_FITTED_FEATURES,
    MAX_TOTAL_RULE_SHIFT,
    PER_RULE_CONTRIBUTION_CAP,
    PROFILE_STRUCTURED,
    PROFILE_TEXT_ONLY,
    RULE_FEATURE_ORDER,
    FeatureContractViolation,
    assert_features_partitioned,
    assert_no_forbidden_columns,
    assert_penalty_caps_sane,
    assert_rule_vector,
    assert_valid_profile,
    clip_rule_shift,
    text_document_columns,
)


# --- the leak we are actually afraid of -------------------------------------


@pytest.mark.parametrize(
    "column",
    ["has_company_logo", "has_questions", "telecommuting", "company_profile", "job_id"],
)
def test_platform_metadata_is_rejected_for_text_only(column):
    """None of these can be recovered from a pasted WhatsApp message."""
    with pytest.raises(FeatureContractViolation, match=column):
        assert_no_forbidden_columns(["title", "description", column], PROFILE_TEXT_ONLY)


def test_allowed_columns_pass():
    assert_no_forbidden_columns(["title", "description", "requirements"], PROFILE_TEXT_ONLY)


def test_location_is_rule_only():
    """Usable for UMK lookup, never as a model feature — EMSCAD geography does not
    transfer to Indonesia and the model would learn it as a fraud proxy."""
    with pytest.raises(FeatureContractViolation, match="location"):
        assert_no_forbidden_columns(["title", "location"], PROFILE_TEXT_ONLY)


def test_structured_profile_permits_platform_metadata():
    assert_no_forbidden_columns(
        ["title", "description", "company_profile", "has_company_logo"], PROFILE_STRUCTURED
    )


def test_text_document_fields_are_self_consistent():
    """The document builder must not assemble a field the profile forbids."""
    for profile in (PROFILE_TEXT_ONLY, PROFILE_STRUCTURED):
        assert_no_forbidden_columns(text_document_columns(profile), profile)


def test_text_document_fields_exist_in_emscad():
    for profile in (PROFILE_TEXT_ONLY, PROFILE_STRUCTURED):
        for column in text_document_columns(profile):
            assert column in EMSCAD_COLUMNS, f"{column!r} is not an EMSCAD column"


def test_unknown_profile_is_rejected():
    with pytest.raises(FeatureContractViolation):
        assert_valid_profile("whatever")


# --- the positional contract between training and serving -------------------


def test_rule_vector_order_is_enforced():
    assert_rule_vector(RULE_FEATURE_ORDER)

    scrambled = list(RULE_FEATURE_ORDER)
    scrambled[0], scrambled[1] = scrambled[1], scrambled[0]
    with pytest.raises(FeatureContractViolation, match="order mismatch"):
        assert_rule_vector(scrambled)

    with pytest.raises(FeatureContractViolation):
        assert_rule_vector(RULE_FEATURE_ORDER[:-1])


def test_rule_features_are_unique_and_partitioned():
    assert len(set(RULE_FEATURE_ORDER)) == len(RULE_FEATURE_ORDER), "duplicate rule feature"
    assert_features_partitioned()


def test_feature_order_is_independent_of_bucket_membership():
    """Reclassifying a feature must never permute the vector.

    RULE_FEATURE_ORDER is declared literally, not derived from the buckets, because
    the fusion model indexes it positionally. Moving a feature between weight-source
    buckets — which measurement forced us to do once already — must be a no-op here.
    """
    assert RULE_FEATURE_ORDER.index("email_free_provider") == 0
    assert RULE_FEATURE_ORDER.index("payment_request_id") == len(RULE_FEATURE_ORDER) - 1
    assert set(EMSCAD_DERIVABLE_FEATURES) | set(INDONESIAN_FITTED_FEATURES) == set(
        RULE_FEATURE_ORDER
    )


def test_only_qualification_conflict_is_learnable_from_emscad():
    """Records the measured finding — see eval/derivability_report.md.

    EMSCAD strips contact details, so every contact-derived feature must have its
    weight fitted on Indonesian data instead. If this assertion ever fails, the
    corpus changed and ml/verify_derivability.py must be re-run.
    """
    assert EMSCAD_DERIVABLE_FEATURES == ("qualification_conflict",)
    for feature in ("email_free_provider", "email_absent", "email_domain_mismatch",
                    "contact_messaging_only", "url_shortener"):
        assert feature in INDONESIAN_FITTED_FEATURES


# --- the bounded-contribution guarantee (concept paper 3.3) -----------------


def test_no_single_rule_can_dominate():
    assert_penalty_caps_sane()
    assert 0.0 < PER_RULE_CONTRIBUTION_CAP <= 0.5
    assert PER_RULE_CONTRIBUTION_CAP <= MAX_TOTAL_RULE_SHIFT


def test_an_oversized_per_rule_cap_is_rejected(monkeypatch):
    """The guard must actually fire — not just pass because we happen to be in budget."""
    import ml.feature_contract as fc

    monkeypatch.setattr(fc, "PER_RULE_CONTRIBUTION_CAP", 0.9)
    with pytest.raises(FeatureContractViolation, match="dominate"):
        fc.assert_penalty_caps_sane()


def test_aggregate_rule_shift_is_clipped():
    assert clip_rule_shift(0.9) == MAX_TOTAL_RULE_SHIFT
    assert clip_rule_shift(-0.9) == -MAX_TOTAL_RULE_SHIFT
    assert clip_rule_shift(0.1) == 0.1
