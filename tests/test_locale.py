"""Locale layer — English-first operation, Indonesian as a drop-in.

The product must be fully useful with English resources alone, because the
Indonesian evaluation data may never materialise. When Indonesian files DO appear,
they must activate with no code change.

The tests below cover both halves of that promise, plus the failure mode in between:
a missing resource must make its rule report itself UNASSESSED, never clean.
"""

from __future__ import annotations

import json

import pytest

from api.ingest import ingest
from api.locale import (
    FALLBACK_LOCALE,
    LOCALE_EN,
    LOCALE_ID,
    Locale,
    LocaleRegistry,
    detect_language,
    load_registry,
)
from api.rules.engine import default_engine
from api.rules.risk_phrases import find_phrases
from api.rules.salary import extract_monthly_salary
from api.scoring import compute_score

EN = load_registry().get(LOCALE_EN)
ID = load_registry().get(LOCALE_ID)


ENGLISH_SCAM = (
    "URGENT HIRING - Data Entry Clerk (Work From Home)\n"
    "Earn $5,000 per month, no experience necessary, start immediately!\n"
    "Limited slots available. A one-time administration fee of $50 is required "
    "to process your application.\n"
    "Interview conducted via Telegram. Send your CV and a photo of your ID to "
    "hiring.dept2024@gmail.com"
)

ENGLISH_LEGIT = (
    "Senior Backend Engineer - Acme Technologies\n"
    "Location: Seattle, Washington. Full-time, hybrid.\n"
    "Requirements: BS in Computer Science or equivalent, at least 4 years of "
    "experience building backend services in Python or Go.\n"
    "Salary range: $9,000 - $12,000 per month depending on experience.\n"
    "Apply through our careers page at https://careers.acmetechnologies.com"
)


# ===========================================================================
# Detection
# ===========================================================================


def test_english_text_detected_as_english():
    assert detect_language(ENGLISH_SCAM) == LOCALE_EN
    assert detect_language(ENGLISH_LEGIT) == LOCALE_EN


def test_indonesian_text_detected_as_indonesian(scam_text, legit_text):
    assert detect_language(scam_text) == LOCALE_ID
    assert detect_language(legit_text) == LOCALE_ID


def test_ambiguous_text_falls_back_to_english():
    """Ties go to the guaranteed-resourced locale."""
    assert detect_language("Admin 123") == FALLBACK_LOCALE


# ===========================================================================
# English must work standalone — the whole point of this layer
# ===========================================================================


def test_english_is_usable_out_of_the_box():
    assert EN.is_usable
    assert EN.has_lexicon
    assert EN.has_wages
    assert LOCALE_EN in load_registry().available()


def test_english_scam_triggers_the_high_weight_rules():
    """Before the locale layer, only 2 of 9 rules fired on this ad and all three
    highest-weighted rules were silent."""
    evaluation = default_engine(EN).evaluate(ingest(ENGLISH_SCAM))
    fired = {f for f, o in evaluation.outcomes.items() if o.fired}
    for expected in ("payment_request_id", "risk_phrase_score_id",
                     "salary_implausible_vs_umk", "email_free_provider",
                     "contact_messaging_only"):
        assert expected in fired, f"{expected} should fire on an English scam"


def test_english_scam_scores_as_high_risk():
    evaluation = default_engine(EN).evaluate(ingest(ENGLISH_SCAM))
    breakdown = compute_score(0.5, evaluation)
    assert breakdown.integrity_score < 40
    assert breakdown.risk_label.value == "Tinggi"


def test_english_legitimate_ad_stays_clean():
    """The false-positive guard, English edition."""
    evaluation = default_engine(EN).evaluate(ingest(ENGLISH_LEGIT))
    fired = {f for f, o in evaluation.outcomes.items() if o.fired}
    assert fired == set(), f"legitimate English ad fired: {fired}"


def test_english_payment_phrases():
    for phrase in ("administration fee", "processing fee", "security deposit",
                   "upfront payment", "wire transfer"):
        text = f"Great opportunity! A {phrase} of $50 is required before we can proceed."
        outcome = default_engine(EN).evaluate(ingest(text)).outcomes["payment_request_id"]
        assert outcome.fired, phrase


def test_english_benefit_framing_is_not_a_demand():
    """Same protection as Indonesian: the employer paying is a perk, not a scam."""
    text = "We provide full training at no cost. There is no application fee."
    outcome = default_engine(EN).evaluate(ingest(text)).outcomes["payment_request_id"]
    assert not outcome.fired


# ===========================================================================
# Currency and separator conventions — inverted between the locales
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Salary $5,000 per month", 5_000),
        ("Earn $5000 monthly", 5_000),
        ("Salary of $60,000 per year", 5_000),
        ("Pay $25 per hour", 4_333),
        ("Earn up to 5k per month", 5_000),
    ],
)
def test_english_salary_parsing(text, expected):
    parsed = extract_monthly_salary(text, EN)
    assert parsed is not None, text
    assert parsed[0] == expected


def test_separator_conventions_are_inverted():
    """The single most dangerous detail in the parser.

    Indonesian "9.000.000" is nine million (dot = thousands).
    English "9,000" is nine thousand (comma = thousands, dot = decimal).
    Reading either with the other convention breaks the rule completely.
    """
    assert extract_monthly_salary("Gaji Rp9.000.000 per bulan", ID)[0] == 9_000_000
    assert extract_monthly_salary("Salary $9,000 per month", EN)[0] == 9_000


def test_english_fee_is_not_read_as_salary():
    assert extract_monthly_salary("A processing fee of $50 is required", EN) is None


def test_per_annum_does_not_match_inside_per_bulan():
    """`pa\\b` without a leading boundary matches inside "per bulan" (Indonesian for
    "per month"), dividing every monthly salary by 12 and disabling the rule."""
    assert extract_monthly_salary("Gaji Rp15.000.000 per bulan", ID)[0] == 15_000_000


# ===========================================================================
# Missing resources must read as UNASSESSED, never clean
# ===========================================================================


def _empty_locale(code: str = "xx") -> Locale:
    return Locale(code=code, name="Empty", lexicon=None, wages=None)


def test_missing_lexicon_reports_unassessed_not_clean():
    """A missing lexicon looking like a clean ad is the silent failure this whole
    tri-state design exists to prevent."""
    evaluation = default_engine(_empty_locale()).evaluate(ingest(ENGLISH_SCAM))
    for feature in ("payment_request_id", "risk_phrase_score_id"):
        outcome = evaluation.outcomes[feature]
        assert outcome.available is False
        assert not outcome.fired


def test_missing_wage_table_reports_unassessed():
    outcome = default_engine(_empty_locale()).evaluate(ingest(ENGLISH_SCAM)).outcomes[
        "salary_implausible_vs_umk"
    ]
    assert outcome.available is False


def test_unassessed_rules_contribute_nothing_to_the_score():
    evaluation = default_engine(_empty_locale()).evaluate(ingest(ENGLISH_SCAM))
    breakdown = compute_score(0.5, evaluation)
    for feature in ("payment_request_id", "risk_phrase_score_id",
                    "salary_implausible_vs_umk"):
        assert feature not in breakdown.contributions


def test_unassessed_features_are_reported():
    evaluation = default_engine(_empty_locale()).evaluate(ingest(ENGLISH_SCAM))
    unavailable = set(evaluation.unavailable_features)
    assert {"payment_request_id", "risk_phrase_score_id",
            "salary_implausible_vs_umk"} <= unavailable


# ===========================================================================
# Registry behaviour and the Indonesian drop-in
# ===========================================================================


def test_registry_lists_only_usable_locales():
    registry = LocaleRegistry({
        LOCALE_EN: EN,
        "xx": _empty_locale(),
    })
    assert registry.available() == (LOCALE_EN,)


def test_resolve_honours_an_explicit_request():
    assert load_registry().resolve(ENGLISH_SCAM, LOCALE_ID).code == LOCALE_ID


def test_resolve_ignores_a_request_for_an_unusable_locale():
    """Asking for a locale with no resources must not produce an empty rule set."""
    registry = LocaleRegistry({LOCALE_EN: EN, "xx": _empty_locale()})
    assert registry.resolve(ENGLISH_SCAM, "xx").code == LOCALE_EN


def test_resolve_falls_back_when_detected_locale_is_unresourced(scam_text):
    """Indonesian text with no Indonesian resources scores with English rules.

    Degraded but honest — an empty rule set would be silently useless.
    """
    registry = LocaleRegistry({LOCALE_EN: EN, LOCALE_ID: _empty_locale(LOCALE_ID)})
    assert registry.resolve(scam_text).code == LOCALE_EN


def test_indonesian_activates_when_its_resources_are_present(scam_text):
    """The drop-in promise: with the files installed, Indonesian text routes to
    Indonesian rules automatically."""
    assert ID.is_usable
    assert load_registry().resolve(scam_text).code == LOCALE_ID


def test_unknown_locale_code_raises():
    with pytest.raises(KeyError, match="Unknown locale"):
        load_registry().get("klingon")


def test_missing_resources_are_reported_per_locale():
    registry = LocaleRegistry({"xx": _empty_locale()})
    assert registry.available() == ()


# ===========================================================================
# API surface
# ===========================================================================


def test_health_advertises_available_locales(client):
    body = client.get("/health").json()
    assert LOCALE_EN in body["locales_available"]
    assert "locale_resources" in body


def test_analyze_reports_the_locale_used(client):
    body = client.post("/api/v1/analyze", json={"text": ENGLISH_SCAM}).json()
    assert body["locale"] == LOCALE_EN
    assert body["locale_detected"] == LOCALE_EN


def test_analyze_accepts_an_explicit_locale(client, scam_text):
    body = client.post(
        "/api/v1/analyze", json={"text": scam_text, "locale": "id"}
    ).json()
    assert body["locale"] == LOCALE_ID


def test_analyze_surfaces_unassessed_rules(client):
    body = client.post("/api/v1/analyze", json={"text": ENGLISH_SCAM}).json()
    assert "unassessed_rules" in body


def test_english_scam_produces_rule_hits_through_the_api(client):
    body = client.post("/api/v1/analyze", json={"text": ENGLISH_SCAM}).json()
    fired = {h["rule_id"] for h in body["rule_hits"]}
    assert "payment_request_id" in fired
    assert "salary_implausible_vs_umk" in fired
