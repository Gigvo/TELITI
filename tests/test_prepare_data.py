"""Tests for text cleaning and split preparation — MVP_PLAN.md step 1.2.

The split logic is tested on a synthetic frame rather than the real 50MB CSV so the
suite stays fast and runs without the dataset present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.feature_contract import EMSCAD_COLUMNS, LABEL_COLUMN, PROFILE_TEXT_ONLY
from ml.prepare_data import (
    MIN_DOCUMENT_WORDS,
    assert_splits_sound,
    build_frame,
    drop_degenerate,
    split_checksum,
    stratified_splits,
    validate_source,
)
from ml.text_cleaning import build_document, clean_text, desegment


# ===========================================================================
# Text cleaning
# ===========================================================================


def test_html_entities_are_decoded():
    assert clean_text("Salary &amp; benefits") == "Salary & benefits"
    assert "'" in clean_text("we&#39;re hiring")


def test_double_encoded_entities_are_decoded():
    """0.006% of EMSCAD descriptions are double-encoded."""
    assert "'" in clean_text("we&amp;#39;re hiring")


def test_glued_words_are_separated():
    """The dominant defect: tags stripped without a separator, 78% of documents."""
    assert desegment("Research InstituteOur passion") == "Research Institute Our passion"
    assert desegment("Account ExecutiveAs a member") == "Account Executive As a member"


@pytest.mark.parametrize(
    "term",
    ["JavaScript", "PostgreSQL", "iPhone", "PowerPoint", "MySQL", "LinkedIn", "WordPress"],
)
def test_real_camelcase_terms_survive_desegmentation(term):
    """JavaScript appears 917 times, PowerPoint 699, MySQL 592. Splitting them into
    'Java Script' would corrupt genuine technical vocabulary."""
    assert desegment(f"Experience with {term} required") == f"Experience with {term} required"


def test_desegmentation_is_case_insensitive_about_protection():
    assert "javascript" in desegment("Experience with javascript required").lower()


@pytest.mark.parametrize(
    "token",
    ["#EMAIL_a1b2c3d4#", "#URL_deadbeef#", "#PHONE_0f0f0f#", "#URL_75db76d58f7994c7#"],
)
def test_redaction_placeholders_survive_cleaning(token):
    """The placeholders MUST survive intact.

    They are the only marker telling the rule layer that contact details were
    stripped before we saw the text. Desegmentation's digit rule would otherwise
    rewrite "#EMAIL_a1b2c3d4#" as "#EMAIL_a 1b 2c 3d 4#", destroying the marker with
    no error — and email_absent would then report a redacted corpus as clean.
    """
    assert token in clean_text(f"Please send your resume to {token} for details.")
    assert token in desegment(f"Contact {token} today")


def test_redaction_placeholders_are_still_detected_after_cleaning():
    """End-to-end consequence of the bug above."""
    from api.ingest import ingest

    cleaned = clean_text("Send your resume to #EMAIL_a1b2c3d4# or visit #URL_deadbeef#.")
    assert ingest(cleaned).has_redaction_placeholders


def test_mojibake_apostrophe_is_repaired():
    """'Esri�s geographic' is a mangled curly apostrophe, pervasive in this copy."""
    assert clean_text("Esri�s technology") == "Esri's technology"


def test_stray_replacement_characters_are_removed():
    assert "�" not in clean_text("ESRI � Environmental Systems")


def test_html_tags_become_spaces_not_nothing():
    """Replacing tags with '' is exactly what produced the glued words upstream."""
    assert clean_text("<p>Institute</p><p>Our passion</p>") == "Institute Our passion"


def test_non_breaking_spaces_are_normalised():
    assert clean_text("world.\xa0 Privately held") == "world. Privately held"


def test_whitespace_is_collapsed_and_trimmed():
    assert clean_text("  too   many \t spaces  ") == "too many spaces"


def test_empty_input_is_handled():
    assert clean_text("") == ""
    assert clean_text(None or "") == ""


def test_build_document_joins_with_newlines():
    """Newline-joined so the XAI sentence splitter (step 3.4) has real boundaries."""
    doc = build_document(
        {"title": "Admin", "description": "Work from home.", "requirements": "None."},
        ("title", "description", "requirements"),
    )
    assert doc == "Admin\nWork from home.\nNone."


def test_build_document_skips_empty_fields():
    doc = build_document(
        {"title": "Admin", "description": "", "requirements": "None."},
        ("title", "description", "requirements"),
    )
    assert doc == "Admin\nNone."


# ===========================================================================
# Source validation (Gate 0.3)
# ===========================================================================


def _synthetic_emscad(n: int = 2000, fraud: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=fraud, replace=False)] = 1
    data = {column: [""] * n for column in EMSCAD_COLUMNS}
    data["job_id"] = list(range(1, n + 1))
    data["title"] = [f"Job title number {i}" for i in range(n)]
    data["description"] = ["A reasonably long job description with several words." for _ in range(n)]
    data["requirements"] = ["Some requirements listed here for the role." for _ in range(n)]
    data[LABEL_COLUMN] = labels
    return pd.DataFrame(data, columns=list(EMSCAD_COLUMNS))


def test_validate_source_rejects_a_wrong_row_count():
    with pytest.raises(ValueError, match="expected 17880 rows"):
        validate_source(_synthetic_emscad())


def test_validate_source_reports_problems_without_raising_when_not_strict():
    problems = validate_source(_synthetic_emscad(), strict=False)
    assert any("rows" in p for p in problems)


def test_validate_source_detects_missing_columns():
    df = _synthetic_emscad().drop(columns=["has_company_logo"])
    problems = validate_source(df, strict=False)
    assert any("column mismatch" in p for p in problems)


# ===========================================================================
# Frame construction and the feature contract
# ===========================================================================


def test_build_frame_emits_only_id_text_and_label():
    frame = build_frame(_synthetic_emscad(), PROFILE_TEXT_ONLY, desegment=True)
    assert set(frame.columns) == {"job_id", "text", LABEL_COLUMN, "n_words"}


def test_build_frame_carries_no_forbidden_column():
    """The guard from MVP_PLAN.md section 1.1: has_company_logo must never survive."""
    frame = build_frame(_synthetic_emscad(), PROFILE_TEXT_ONLY, desegment=True)
    for forbidden in ("has_company_logo", "has_questions", "telecommuting", "company_profile"):
        assert forbidden not in frame.columns


def test_degenerate_rows_are_dropped():
    df = _synthetic_emscad(n=100, fraud=10)
    df.loc[0, ["title", "description", "requirements"]] = ["Hi", "", ""]
    frame = build_frame(df, PROFILE_TEXT_ONLY, desegment=True)
    kept, dropped = drop_degenerate(frame)
    assert len(dropped) == 1
    assert (kept["n_words"] >= MIN_DOCUMENT_WORDS).all()


# ===========================================================================
# Splitting
# ===========================================================================


@pytest.fixture(scope="module")
def splits():
    frame = build_frame(_synthetic_emscad(n=4000, fraud=400), PROFILE_TEXT_ONLY, desegment=True)
    frame, _ = drop_degenerate(frame)
    return frame, stratified_splits(frame, seed=42)


def test_splits_have_the_intended_proportions(splits):
    frame, parts = splits
    total = len(frame)
    for name, expected in (("train", 0.70), ("val", 0.10), ("calib", 0.10), ("test", 0.10)):
        assert abs(len(parts[name]) / total - expected) < 0.01, name


def test_splits_are_disjoint_and_complete(splits):
    frame, parts = splits
    assert_splits_sound(parts, len(frame), float(frame[LABEL_COLUMN].mean()))


def test_overlapping_splits_are_rejected(splits):
    """assert_splits_sound must actually catch a leak, not just pass on clean input."""
    frame, parts = splits
    broken = dict(parts)
    broken["val"] = pd.concat([parts["val"], parts["train"].head(1)]).reset_index(drop=True)
    with pytest.raises(AssertionError, match="appears in both"):
        assert_splits_sound(broken, len(frame), float(frame[LABEL_COLUMN].mean()))


def test_stratification_preserves_the_fraud_rate(splits):
    frame, parts = splits
    base = frame[LABEL_COLUMN].mean()
    for name, part in parts.items():
        assert abs(part[LABEL_COLUMN].mean() - base) <= 0.005, name


def test_splitting_is_reproducible():
    frame = build_frame(_synthetic_emscad(n=2000, fraud=200), PROFILE_TEXT_ONLY, desegment=True)
    frame, _ = drop_degenerate(frame)
    first = stratified_splits(frame, seed=42)
    second = stratified_splits(frame, seed=42)
    for name in first:
        assert split_checksum(first[name]["job_id"]) == split_checksum(second[name]["job_id"])


def test_a_different_seed_produces_different_splits():
    frame = build_frame(_synthetic_emscad(n=2000, fraud=200), PROFILE_TEXT_ONLY, desegment=True)
    frame, _ = drop_degenerate(frame)
    assert split_checksum(stratified_splits(frame, 42)["train"]["job_id"]) != split_checksum(
        stratified_splits(frame, 7)["train"]["job_id"]
    )
