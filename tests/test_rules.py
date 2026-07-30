"""Golden cases for the rule layer — MVP_PLAN.md gate 1.4.

Every rule ships with negatives, not just positives. A rule that fires on real
Indonesian job ads is worse than no rule at all: concept paper section 3.6 commits us
to suppressing false positives against legitimate companies, and a rule that flags
everything destroys the score's meaning while looking like it works.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.ingest import acronym_of, ingest, registrable_label, url_host
from api.rules.contact_channel import ContactChannelRule
from api.rules.email_domain import EmailDomainRule
from api.rules.engine import RuleEngine, default_engine
from api.rules.base import Rule, RuleOutcome
from api.schemas import RuleCategory
from ml.feature_contract import RULE_FEATURE_ORDER, FeatureContractViolation


def severity(text: str, feature_id: str) -> float:
    outcome = default_engine().evaluate(ingest(text)).outcomes[feature_id]
    return outcome.severity if outcome.available else float("nan")


def outcome_for(text: str, feature_id: str) -> RuleOutcome:
    return default_engine().evaluate(ingest(text)).outcomes[feature_id]


# ===========================================================================
# Ingest helpers
# ===========================================================================


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("karier.teknologinusantara.co.id", "teknologinusantara"),
        ("gmail.com", "gmail"),
        ("mail.bca.co.id", "bca"),
        ("company.com", "company"),
        ("sub.domain.example.org", "example"),
    ],
)
def test_registrable_label(domain, expected):
    assert registrable_label(domain) == expected


def test_url_host_strips_scheme_path_and_query():
    assert url_host("https://karier.example.co.id/lowongan?id=3") == "karier.example.co.id"
    assert url_host("www.example.com") == "www.example.com"


def test_acronym_of():
    assert acronym_of("Bank Central Asia") == "bca"


def test_salary_digits_are_not_read_as_phone_numbers():
    """'Rp9.000.000' must not be extracted as an Indonesian mobile number."""
    result = ingest("Gaji Rp9.000.000 per bulan dan tunjangan Rp250.000 setiap minggu.")
    assert result.phones == ()


def test_real_phone_formats_are_extracted():
    for text in ("Hubungi 081234567890", "WA +62 812-3456-7890", "Telp 0812 3456 7890"):
        assert ingest(f"Lowongan kerja admin. {text} untuk info lebih lanjut.").phones


def test_email_inside_url_is_not_a_contact_address():
    result = ingest("Daftar di https://example.com/apply/contact@example.com sekarang juga.")
    assert result.emails == ()


def test_spans_address_the_raw_text():
    text = "Kirim CV ke rekrutmen@example.co.id atau kunjungi https://example.co.id/karier."
    result = ingest(text)
    for match in (*result.emails, *result.urls):
        assert text[match.start : match.end] == match.text


def test_url_trailing_full_stop_is_trimmed():
    result = ingest("Lamaran dikirim melalui https://example.co.id/karier.")
    assert result.urls[0].text == "https://example.co.id/karier"


def test_company_name_extraction():
    result = ingest("Lowongan di PT Teknologi Nusantara untuk posisi backend engineer.")
    assert result.companies[0].text == "PT Teknologi Nusantara"


def test_company_name_does_not_run_across_a_line_break():
    """`\\s+` between name words swallows the next line's first capitalised word,
    producing "PT Teknologi Nusantara Kualifikasi" — which then fails to match the
    company's own email domain and yields a spurious mismatch."""
    result = ingest("Software Engineer - PT Teknologi Nusantara\nKualifikasi: S1 Ilmu Komputer.")
    assert result.companies[0].text == "PT Teknologi Nusantara"


def test_company_name_survives_line_break_for_domain_matching():
    """The end-to-end consequence of the bug above."""
    text = (
        "Backend Engineer - PT Teknologi Nusantara\n"
        "Kualifikasi: S1 Ilmu Komputer.\n"
        "Kirim lamaran ke recruitment@teknologinusantara.co.id"
    )
    assert severity(text, "email_domain_mismatch") == 0.0


# ===========================================================================
# RULE: email_free_provider
# ===========================================================================


def test_free_provider_fires_on_gmail():
    outcome = outcome_for("Kirim CV dan KTP ke hrd.rekrutmen2024@gmail.com sekarang.", "email_free_provider")
    assert outcome.fired
    assert outcome.evidence == "hrd.rekrutmen2024@gmail.com"


def test_disposable_provider_outranks_ordinary_free_provider():
    disposable = severity("Kirim lamaran ke rekrut@mailinator.com hari ini juga.", "email_free_provider")
    free = severity("Kirim lamaran ke rekrut@gmail.com hari ini juga.", "email_free_provider")
    assert disposable > free


@pytest.mark.parametrize(
    "text",
    [
        "Lamaran dikirim ke recruitment@teknologinusantara.co.id paling lambat Jumat.",
        "Kirim berkas ke hrd@bca.co.id untuk posisi analis kredit.",
        "Hubungi careers@unilever.com untuk informasi lowongan magang.",
    ],
)
def test_free_provider_does_not_fire_on_corporate_domains(text):
    assert severity(text, "email_free_provider") == 0.0


def test_free_provider_span_is_correct():
    text = "Kirim CV ke hrd.rekrutmen2024@gmail.com sebelum tanggal 10."
    outcome = outcome_for(text, "email_free_provider")
    assert text[outcome.span.start : outcome.span.end] == outcome.evidence


# ===========================================================================
# RULE: email_absent
# ===========================================================================


def test_email_absent_does_not_fire_when_a_real_career_page_is_offered():
    """Applying via a company career page is how real companies hire.

    The absence of an email there is not weak evidence, it is no evidence. A small
    non-zero severity would put a warning card on a spotless posting.
    """
    assert severity(
        "Lamaran dikirim melalui halaman karier kami di https://karier.example.co.id",
        "email_absent",
    ) == 0.0


def test_email_absent_escalates_as_the_application_route_gets_thinner():
    no_route = severity(
        "Dibutuhkan admin online untuk perusahaan ternama, gaji besar, kerja dari rumah.",
        "email_absent",
    )
    weak_route = severity(
        "Dibutuhkan admin online, gaji besar. Daftar di https://bit.ly/3xYzAbc",
        "email_absent",
    )
    real_route = severity(
        "Lamaran dikirim melalui halaman karier kami di https://karier.example.co.id",
        "email_absent",
    )
    assert no_route > weak_route > real_route == 0.0


def test_email_absent_does_not_fire_when_an_email_exists():
    assert severity("Kirim CV ke hrd@example.co.id sebelum akhir bulan.", "email_absent") == 0.0


# ===========================================================================
# RULE: email_domain_mismatch
# ===========================================================================


def test_domain_mismatch_fires_when_domain_is_unrelated_to_the_named_company():
    outcome = outcome_for(
        "Lowongan di PT Teknologi Nusantara. Kirim lamaran ke hrd@rekrutmen-cepat123.com",
        "email_domain_mismatch",
    )
    assert outcome.fired


@pytest.mark.parametrize(
    "text",
    [
        # Exact match.
        "Lowongan di PT Teknologi Nusantara. Kirim ke hrd@teknologinusantara.co.id",
        # Subdomain of the company domain.
        "Lowongan di PT Teknologi Nusantara. Kirim ke hrd@karier.teknologinusantara.co.id",
        # Acronym: "Bank Central Asia" -> bca.
        "Lowongan di PT Bank Central Asia. Kirim ke hrd@bca.co.id",
        # A single distinctive word of the company name.
        "Lowongan di PT Teknologi Nusantara. Kirim ke hrd@nusantara.co.id",
    ],
)
def test_domain_mismatch_does_not_fire_on_legitimate_variations(text):
    """False 'mismatch' accuses a real company of impersonating itself."""
    assert severity(text, "email_domain_mismatch") == 0.0


def test_domain_mismatch_ignores_free_providers():
    """A Gmail address is already covered by email_free_provider; counting it here too
    would double-charge the same evidence."""
    assert severity(
        "Lowongan di PT Teknologi Nusantara. Kirim ke hrd.rekrutmen@gmail.com",
        "email_domain_mismatch",
    ) == 0.0


def test_domain_mismatch_needs_a_company_name_to_compare_against():
    assert severity("Kirim lamaran ke hrd@rekrutmen-cepat123.com hari ini.", "email_domain_mismatch") == 0.0


# ===========================================================================
# RULE: contact_messaging_only
# ===========================================================================


def test_telegram_outranks_whatsapp():
    """The central calibration of this rule: WhatsApp is ordinary business comms in
    Indonesia, Telegram is not. Getting this backwards floods legitimate ads with
    false positives."""
    telegram = severity("Dibutuhkan admin. Interview via Telegram, hubungi sekarang.", "contact_messaging_only")
    whatsapp = severity("Dibutuhkan admin. Hubungi via WhatsApp, chat sekarang.", "contact_messaging_only")
    assert telegram > whatsapp > 0.0


def test_messaging_does_not_fire_when_a_company_email_is_also_offered():
    assert severity(
        "Dibutuhkan admin di PT Maju Jaya. Hubungi via WhatsApp atau kirim CV ke "
        "hrd@majujaya.co.id untuk proses lamaran.",
        "contact_messaging_only",
    ) == 0.0


def test_messaging_does_not_fire_when_a_career_page_is_also_offered():
    assert severity(
        "Dibutuhkan admin. Hubungi via WhatsApp, atau lamar di https://karier.majujaya.co.id",
        "contact_messaging_only",
    ) == 0.0


def test_free_provider_email_does_not_count_as_a_formal_route():
    """'Kirim CV ke hrd@gmail.com, interview via Telegram' is the canonical scam
    shape — a Gmail address must not be allowed to suppress this rule."""
    assert severity(
        "Interview via Telegram. Kirim CV ke hrd.rekrutmen2024@gmail.com",
        "contact_messaging_only",
    ) > 0.0


def test_wa_me_link_is_detected():
    assert severity(
        "Lowongan admin online. Daftar sekarang https://wa.me/6281234567890",
        "contact_messaging_only",
    ) > 0.0


def test_chat_hiring_phrase_raises_severity():
    base = severity("Dibutuhkan admin. Hubungi via WhatsApp untuk info.", "contact_messaging_only")
    boosted = severity(
        "Dibutuhkan admin. Hubungi via WhatsApp, langsung kerja tanpa interview.",
        "contact_messaging_only",
    )
    assert boosted > base


def test_messaging_does_not_fire_on_an_ordinary_ad():
    assert severity(
        "Software Engineer di PT Teknologi Nusantara. Kirim lamaran ke "
        "recruitment@teknologinusantara.co.id sebelum 30 Juni.",
        "contact_messaging_only",
    ) == 0.0


# ===========================================================================
# RULE: url_shortener
# ===========================================================================


@pytest.mark.parametrize(
    "url",
    ["https://bit.ly/3xYzAbc", "https://s.id/loker2025", "https://cutt.ly/abc123", "https://tinyurl.com/loker"],
)
def test_shortener_fires_on_known_shorteners(url):
    assert severity(f"Daftar lowongan admin online sekarang juga di {url} ya.", "url_shortener") > 0.0


def test_shortener_is_worse_when_it_is_the_only_link():
    alone = severity("Daftar sekarang di https://bit.ly/3xYzAbc untuk posisi admin.", "url_shortener")
    accompanied = severity(
        "Daftar di https://bit.ly/3xYzAbc atau lihat https://example.co.id/karier untuk detail.",
        "url_shortener",
    )
    assert alone > accompanied


def test_link_aggregators_are_not_treated_as_shorteners():
    """Indonesian SMEs and campus career centres use Linktree for real postings."""
    assert severity(
        "Info lowongan lengkap ada di https://linktr.ee/kariernusantara ya.", "url_shortener"
    ) == 0.0


def test_shortener_does_not_fire_on_an_ordinary_company_url():
    assert severity(
        "Lamaran dikirim melalui https://karier.teknologinusantara.co.id/backend",
        "url_shortener",
    ) == 0.0


# ===========================================================================
# Availability — the EMSCAD redaction problem
# ===========================================================================


EMSCAD_STYLE_TEXT = (
    "We are looking for a customer service representative. "
    "Please send your resume to #EMAIL_a1b2c3d4# or visit #URL_deadbeef# for details."
)


def test_redacted_corpus_marks_contact_signals_unavailable():
    """On EMSCAD the contact details were stripped before we saw them.

    Reporting 'no email -> clean' there would teach the fusion model a property of
    the dataset rather than of job scams. Every contact-derived feature must report
    itself unassessed instead.
    """
    evaluation = default_engine().evaluate(ingest(EMSCAD_STYLE_TEXT))
    for feature_id in ("email_free_provider", "email_absent", "email_domain_mismatch",
                       "contact_messaging_only", "url_shortener"):
        assert evaluation.outcomes[feature_id].available is False


def test_unavailable_is_distinguishable_from_clean():
    """The whole point: 0.0 in the feature vector is ambiguous, the mask is not."""
    redacted = default_engine().evaluate(ingest(EMSCAD_STYLE_TEXT))
    clean = default_engine().evaluate(
        ingest("Kirim lamaran ke recruitment@teknologinusantara.co.id sebelum 30 Juni.")
    )

    index = RULE_FEATURE_ORDER.index("email_free_provider")
    assert redacted.feature_vector()[index] == clean.feature_vector()[index] == 0.0
    assert redacted.availability_mask()[index] is np.False_
    assert clean.availability_mask()[index] is np.True_


def test_unavailable_features_are_reported():
    assert "email_absent" in default_engine().evaluate(ingest(EMSCAD_STYLE_TEXT)).unavailable_features


def test_unavailable_outcomes_never_produce_hits():
    assert default_engine().evaluate(ingest(EMSCAD_STYLE_TEXT)).to_rule_hits() == []


# ===========================================================================
# Engine contract
# ===========================================================================


def test_feature_vector_matches_the_canonical_order():
    vector = default_engine().evaluate(ingest("Kirim CV ke hrd@gmail.com sekarang juga.")).feature_vector()
    assert vector.shape == (len(RULE_FEATURE_ORDER),)
    assert vector.dtype == np.float64


def test_pending_features_are_exactly_the_day_2_rules():
    """Becomes empty when step 2.4 lands. This test is the completion check."""
    assert set(default_engine().pending_features) == {
        "qualification_conflict",
        "salary_implausible_vs_umk",
        "risk_phrase_score_id",
        "payment_request_id",
    }


def test_engine_rejects_two_rules_owning_one_feature():
    with pytest.raises(FeatureContractViolation, match="owned by both"):
        RuleEngine([EmailDomainRule(), EmailDomainRule()])


def test_engine_rejects_unknown_feature_ids():
    class BogusRule(Rule):
        feature_ids = ("not_a_real_feature",)

        def evaluate(self, ctx):
            return []

    with pytest.raises(FeatureContractViolation, match="unknown feature"):
        RuleEngine([BogusRule()])


def test_engine_rejects_a_rule_that_skips_a_declared_feature():
    """A missing slot must never be silently read as a clean one."""

    class LazyRule(Rule):
        feature_ids = ("qualification_conflict", "payment_request_id")

        def evaluate(self, ctx):
            return [
                RuleOutcome(
                    feature_id="qualification_conflict",
                    severity=0.5,
                    label_id="x",
                    label_en="x",
                    category=RuleCategory.QUALIFICATION,
                )
            ]

    with pytest.raises(FeatureContractViolation, match="returned"):
        RuleEngine([LazyRule()]).evaluate(ingest("Lowongan admin online gaji besar sekali."))


def test_rule_hits_are_sorted_by_contribution():
    hits = default_engine().evaluate(
        ingest("Interview via Telegram. Kirim CV ke hrd.rekrutmen2024@gmail.com")
    ).to_rule_hits()
    assert len(hits) >= 2
    assert [h.contribution for h in hits] == sorted((h.contribution for h in hits), reverse=True)


def test_severity_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        RuleOutcome(
            feature_id="email_absent", severity=1.5, label_id="x", label_en="x",
            category=RuleCategory.COMPANY,
        )


def test_unavailable_outcome_cannot_carry_severity():
    with pytest.raises(ValueError, match="not assessed"):
        RuleOutcome(
            feature_id="email_absent", severity=0.4, label_id="x", label_en="x",
            category=RuleCategory.COMPANY, available=False,
        )


# ===========================================================================
# End-to-end: the concept paper's own scenario (section 3.4)
# ===========================================================================


def test_paper_scenario_triggers_the_expected_rules(scam_text):
    evaluation = default_engine().evaluate(ingest(scam_text))
    fired = {fid for fid, o in evaluation.outcomes.items() if o.fired}
    assert "email_free_provider" in fired, "hrd.rekrutmen2024@gmail.com"
    assert "contact_messaging_only" in fired, "interview via Telegram"


def test_legitimate_ad_triggers_no_contact_rules(legit_text):
    """The false-positive guard. A real posting with a corporate email and a career
    page must come out of the contact rules completely clean."""
    evaluation = default_engine().evaluate(ingest(legit_text))
    fired = {fid for fid, o in evaluation.outcomes.items() if o.fired}
    assert fired == set(), f"legitimate ad fired: {fired}"
