"""Tests for the feature-availability contract — MVP_PLAN.md section 1.1.

These exist because the failure they prevent is invisible: a model that leaks
EMSCAD-only metadata trains fine, scores beautifully, and then performs at chance
on real user input. Nothing errors. The only defence is a test that fails first.
"""

from __future__ import annotations

import pytest

from ml.feature_contract import (
    EMSCAD_COLUMNS,
    LEARNED_RULE_FEATURES,
    MAX_TOTAL_PENALTY,
    PENALTY_CAPS,
    PENALTY_RULE_FEATURES,
    PROFILE_STRUCTURED,
    PROFILE_TEXT_ONLY,
    RULE_FEATURE_ORDER,
    FeatureContractViolation,
    assert_no_forbidden_columns,
    assert_penalty_caps_sane,
    assert_rule_vector,
    assert_valid_profile,
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
    assert not set(LEARNED_RULE_FEATURES) & set(PENALTY_RULE_FEATURES), (
        "A feature cannot both have a learned weight and an additive penalty — "
        "it would be counted twice."
    )


# --- the bounded-penalty guarantee (concept paper 3.3) ----------------------


def test_penalty_caps_cannot_dominate_the_model():
    assert_penalty_caps_sane()
    # Same 1e-9 tolerance the implementation uses: 0.05 * 3 == 0.15000000000000002
    # in binary floating point, and a bare <= would fail on an exactly-at-budget config.
    assert sum(PENALTY_CAPS.values()) <= MAX_TOTAL_PENALTY + 1e-9
    assert set(PENALTY_CAPS) == set(PENALTY_RULE_FEATURES)


def test_over_budget_penalty_caps_are_rejected(monkeypatch):
    """The guard must actually fire — not just pass because we happen to be in budget."""
    import ml.feature_contract as fc

    monkeypatch.setattr(fc, "PENALTY_CAPS", {name: 0.5 for name in PENALTY_RULE_FEATURES})
    with pytest.raises(FeatureContractViolation, match="dominate"):
        fc.assert_penalty_caps_sane()


def test_every_penalty_cap_is_positive_and_small():
    for name, cap in PENALTY_CAPS.items():
        assert 0.0 < cap <= MAX_TOTAL_PENALTY, f"{name} has an implausible cap {cap}"
