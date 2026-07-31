"""Golden cases for the step 2.4 rules — MVP_PLAN.md gate 2.4.

Salary, qualification conflict, and the Indonesian risk-phrase lexicon.

As in tests/test_rules.py, every rule ships with negatives. These three are the most
false-positive-prone in the system: salary because legitimate high-paying jobs exist,
qualification because contradictory requirements are usually just sloppy copywriting,
and risk phrases because ordinary marketing language overlaps heavily with scam
language.
"""

from __future__ import annotations

import pytest

from api.ingest import ingest
from api.locale import LOCALE_ID, load_registry
from api.rules.engine import default_engine
from api.rules.qualification import _find_experience_years, _find_no_experience
from api.rules.risk_phrases import find_phrases as _find_phrases
from api.rules.risk_phrases import saturating_score
from api.rules.salary import extract_monthly_salary as _extract_monthly_salary
from api.rules.salary import looks_entry_level as _looks_entry_level

#: These cases are all Indonesian. Pin the locale so a detection wobble on a short
#: fragment cannot make a rule test flap — detection is exercised separately in
#: tests/test_locale.py.
ID = load_registry().get(LOCALE_ID)


def severity(text: str, feature_id: str) -> float:
    outcome = default_engine(ID).evaluate(ingest(text)).outcomes[feature_id]
    return outcome.severity if outcome.available else float("nan")


def outcome_for(text: str, feature_id: str):
    return default_engine(ID).evaluate(ingest(text)).outcomes[feature_id]


# Locale-defaulting wrappers, so the cases below stay readable.
def extract_monthly_salary(text: str, locale=ID):
    return _extract_monthly_salary(text, locale)


def looks_entry_level(text: str, locale=ID) -> bool:
    return _looks_entry_level(text, locale)


def find_phrases(text: str, locale=ID):
    return _find_phrases(text, locale)


def regional_minimum(text: str):
    return ID.wages.lookup(text)


# ===========================================================================
# Salary parsing
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Gaji Rp9.000.000 per bulan", 9_000_000),
        ("Gaji Rp 9.000.000,- per bulan", 9_000_000),
        ("Gaji 9jt per bulan", 9_000_000),
        ("Gaji 9 juta per bulan", 9_000_000),
        ("Penghasilan Rp4,5jt per bulan", 4_500_000),
        ("Gaji Rp750.000 per minggu", 3_000_000),
        ("Upah Rp350.000 per hari", 7_700_000),
    ],
)
def test_salary_formats_are_parsed(text, expected):
    parsed = extract_monthly_salary(text)
    assert parsed is not None, text
    assert parsed[0] == expected


def test_salary_range_takes_the_lower_bound():
    """The top of a range is marketing; judge the ad by its own floor."""
    parsed = extract_monthly_salary("Gaji Rp5.000.000 - Rp12.000.000 sesuai pengalaman")
    assert parsed is not None
    assert parsed[0] == 5_000_000


def test_indonesian_decimal_convention():
    """Comma is the DECIMAL separator, dot is the THOUSAND separator — the opposite
    of English. Getting this backwards makes the rule fire on everything."""
    assert extract_monthly_salary("Gaji 4,5 juta")[0] == 4_500_000
    assert extract_monthly_salary("Gaji Rp4.500.000")[0] == 4_500_000


def test_administration_fee_is_not_read_as_salary():
    """'Wajib membayar biaya administrasi Rp250.000' is a COST to the applicant.

    Reading it as compensation would both miss the scam and mis-state the salary.
    """
    assert extract_monthly_salary("Wajib membayar biaya administrasi sebesar Rp250.000") is None


def test_numbers_without_salary_context_are_ignored():
    assert extract_monthly_salary("Kantor kami di Jalan Merdeka No. 45.000 karyawan") is None
    assert extract_monthly_salary("Perusahaan berdiri sejak 1.999") is None


def test_no_salary_mentioned_parses_to_nothing():
    assert extract_monthly_salary("Dibutuhkan admin, kirim CV ke hrd@example.co.id") is None


# ===========================================================================
# UMK lookup
# ===========================================================================


def test_city_resolves_to_its_province():
    value, region = regional_minimum("Lokasi penempatan: Yogyakarta")
    assert value > 0
    assert "yogya" in region.lower()


def test_jakarta_has_the_highest_minimum():
    jakarta, _ = regional_minimum("Lokasi: Jakarta Selatan")
    jateng, _ = regional_minimum("Lokasi: Semarang")
    assert jakarta > jateng


def test_unknown_location_falls_back_to_national_median():
    value, region = regional_minimum("Lowongan admin online, kerja dari rumah")
    assert value > 0
    assert "nasional" in region.lower()


# ===========================================================================
# RULE: salary_implausible_vs_umk
# ===========================================================================


def test_paper_scenario_fires(scam_text):
    """Concept paper 3.4: 'admin online, gaji Rp9 juta, tanpa pengalaman'."""
    assert severity(scam_text, "salary_implausible_vs_umk") > 0.0


def test_entry_level_with_absurd_salary_fires():
    assert severity(
        "Dibutuhkan admin online di Yogyakarta, gaji Rp15.000.000 per bulan, "
        "tanpa pengalaman, langsung kerja.",
        "salary_implausible_vs_umk",
    ) > 0.0


@pytest.mark.parametrize(
    "text",
    [
        # Senior role, high salary — entirely ordinary.
        "Senior Backend Engineer di Jakarta. Pengalaman minimal 5 tahun. "
        "Gaji Rp25.000.000 - Rp35.000.000 per bulan.",
        # Entry level at a plausible wage.
        "Dibutuhkan admin di Semarang, gaji Rp3.000.000 per bulan, fresh graduate dipersilakan.",
        # Manager: seniority marker removes the entry-level assumption.
        "Marketing Manager Jakarta, gaji Rp20.000.000, tanpa pengalaman di bidang FMCG "
        "tidak masalah.",
        # No salary quoted at all.
        "Dibutuhkan staff admin, gaji sesuai standar perusahaan. Kirim CV ke hrd@example.co.id",
    ],
)
def test_legitimate_salaries_do_not_fire(text):
    """Legitimate high-paying jobs exist. Flagging them hits exactly the employers
    section 3.6 tells us to protect."""
    assert severity(text, "salary_implausible_vs_umk") == 0.0


def test_severity_rises_with_the_multiple():
    modest = severity(
        "Admin online Yogyakarta, gaji Rp6.000.000, tanpa pengalaman.",
        "salary_implausible_vs_umk",
    )
    extreme = severity(
        "Admin online Yogyakarta, gaji Rp30.000.000, tanpa pengalaman.",
        "salary_implausible_vs_umk",
    )
    assert extreme > modest


def test_evidence_explains_the_comparison():
    outcome = outcome_for(
        "Admin online Yogyakarta, gaji Rp15.000.000, tanpa pengalaman.",
        "salary_implausible_vs_umk",
    )
    assert "upah minimum" in outcome.evidence.lower()
    assert "x" in outcome.evidence


def test_entry_level_detection():
    assert looks_entry_level("Dibutuhkan admin online, tanpa pengalaman")
    assert looks_entry_level("Fresh graduate dipersilakan melamar")
    assert not looks_entry_level("Senior Software Engineer, pengalaman minimal 5 tahun")
    assert not looks_entry_level("Marketing Manager dengan pengalaman minimal 3 tahun")


# ===========================================================================
# RULE: qualification_conflict
# ===========================================================================


def test_indonesian_conflict_fires():
    assert severity(
        "Loker CS online. Dibutuhkan fresh graduate dengan pengalaman minimal 5 tahun.",
        "qualification_conflict",
    ) > 0.0


def test_english_conflict_fires():
    """Bilingual by necessity: this is the ONE feature EMSCAD can teach, and EMSCAD
    is English. Indonesian-only patterns would forfeit that."""
    assert severity(
        "Entry level position available. Candidates must have at least 5 years "
        "of relevant experience in customer service.",
        "qualification_conflict",
    ) > 0.0


@pytest.mark.parametrize(
    "text",
    [
        "Fresh graduate dipersilakan melamar. Pelatihan disediakan perusahaan.",
        "Senior Engineer, pengalaman minimal 5 tahun membangun sistem terdistribusi.",
        "Fresh graduate welcome, minimal 1 tahun pengalaman magang menjadi nilai tambah.",
        "Dibutuhkan admin, tanpa pengalaman, akan diberikan training selama 2 minggu.",
    ],
)
def test_coherent_requirements_do_not_fire(text):
    """Contradictory requirements are usually sloppy copywriting, not fraud — and a
    1-year internship ask alongside 'fresh graduate' is perfectly coherent."""
    assert severity(text, "qualification_conflict") == 0.0


def test_wider_gaps_are_more_severe():
    small = severity(
        "Fresh graduate dengan pengalaman minimal 2 tahun.", "qualification_conflict"
    )
    large = severity(
        "Fresh graduate dengan pengalaman minimal 8 tahun.", "qualification_conflict"
    )
    assert large > small


def test_severity_is_capped_because_it_is_often_just_carelessness():
    assert severity(
        "Fresh graduate dengan pengalaman minimal 20 tahun.", "qualification_conflict"
    ) <= 0.85


def test_negated_mention_is_ignored():
    assert _find_no_experience("Kandidat bukan fresh graduate diutamakan") is None


def test_experience_year_extraction():
    assert _find_experience_years("pengalaman minimal 5 tahun")[0] == 5
    assert _find_experience_years("3 tahun pengalaman di bidang serupa")[0] == 3
    assert _find_experience_years("at least 4 years experience")[0] == 4
    assert _find_experience_years("5+ years of relevant experience")[0] == 5
    assert _find_experience_years("pengalaman minimal 1 tahun") is None  # below threshold


# ===========================================================================
# RULE: payment_request_id
# ===========================================================================


@pytest.mark.parametrize(
    "phrase",
    ["biaya administrasi", "biaya pelatihan", "uang jaminan", "biaya pendaftaran", "wajib transfer"],
)
def test_payment_request_fires(phrase):
    assert severity(
        f"Lowongan admin online. Pelamar {phrase} sebesar Rp250.000 sebelum mulai bekerja.",
        "payment_request_id",
    ) > 0.0


@pytest.mark.parametrize(
    "text",
    [
        "Perusahaan menanggung seluruh biaya pelatihan karyawan baru.",
        "Backend Engineer di PT Teknologi Nusantara. Kirim CV ke hrd@teknologinusantara.co.id",
        "Benefit: asuransi kesehatan, tunjangan transportasi, dan bonus tahunan.",
    ],
)
def test_payment_request_does_not_fire_on_ordinary_ads(text):
    assert severity(text, "payment_request_id") == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "Perusahaan menanggung seluruh biaya pelatihan karyawan baru.",
        "Biaya pelatihan ditanggung perusahaan sepenuhnya.",
        "Tidak dipungut biaya administrasi dalam proses rekrutmen ini.",
        "Seluruh biaya akomodasi digratiskan untuk peserta terpilih.",
        "Proses rekrutmen kami gratis, tanpa biaya pendaftaran apa pun.",
    ],
)
def test_employer_paid_costs_are_a_benefit_not_a_demand(text):
    """"Perusahaan menanggung biaya pelatihan" is the OPPOSITE of a scam signal.

    Substring matching alone cannot tell who pays, so the payer is read from the
    surrounding context. Firing here would flag good employers for advertising a
    perk — and would have fired on a real anti-fraud disclaimer
    ("tidak dipungut biaya"), which is the single most ironic false positive
    available to this system.
    """
    assert severity(text, "payment_request_id") == 0.0


def test_benefit_framing_does_not_suppress_a_real_demand():
    """The negation window must not swallow a genuine demand nearby."""
    assert severity(
        "Pelatihan gratis, namun pelamar wajib membayar biaya administrasi Rp250.000.",
        "payment_request_id",
    ) > 0.0


def test_payment_is_separate_from_the_aggregate():
    """The payment group must not also inflate risk_phrase_score_id — that would
    count the same evidence twice."""
    evaluation = default_engine().evaluate(
        ingest("Wajib membayar biaya administrasi Rp250.000 untuk proses berkas lamaran.")
    )
    assert evaluation.outcomes["payment_request_id"].fired
    assert "biaya administrasi" not in evaluation.outcomes["risk_phrase_score_id"].evidence


# ===========================================================================
# RULE: risk_phrase_score_id
# ===========================================================================


def test_risk_phrases_fire_on_scam_language():
    assert severity(
        "Penghasilan tak terbatas, kerja santai cukup pakai hp, kuota terbatas!",
        "risk_phrase_score_id",
    ) > 0.0


def test_clean_professional_ad_does_not_fire(legit_text):
    assert severity(legit_text, "risk_phrase_score_id") == 0.0


def test_weak_phrases_cannot_accumulate_to_certainty():
    """The saturating shape is what stops ordinary marketing language from
    manufacturing a maximum-severity verdict on a real small business."""
    mild = severity(
        "Dibutuhkan admin, kerja dari rumah, segera hubungi kami untuk info lebih lanjut.",
        "risk_phrase_score_id",
    )
    assert mild < 0.6


def test_severity_rises_with_corroboration():
    one = severity("Lowongan admin, kerja dari rumah saja.", "risk_phrase_score_id")
    many = severity(
        "Penghasilan tak terbatas, modal hp saja, kerja santai, tanpa interview, "
        "kuota terbatas, langsung diterima!",
        "risk_phrase_score_id",
    )
    assert many > one


def test_saturating_score_properties():
    assert saturating_score([]) == 0.0
    assert saturating_score([0.5, 0.5]) == pytest.approx(0.75)
    # Five weak phrases must not reach certainty.
    assert saturating_score([0.2] * 5) < 0.7
    assert saturating_score([0.9, 0.9, 0.9]) < 1.0


def test_word_boundaries_prevent_substring_matches():
    """'deposit' must not match inside 'depositor'."""
    assert not any(m.text.lower() == "deposit" for m in find_phrases("Kami mencari depositor."))


def test_lexicon_loads_and_has_content():
    matches = find_phrases("biaya administrasi dan kuota terbatas")
    assert {m.group for m in matches} >= {"payment", "urgency"}


# ===========================================================================
# Engine completion — gate 2.4
# ===========================================================================


def test_all_feature_slots_are_now_implemented():
    """The completion check for step 2.4. This is what 'done' means."""
    assert default_engine().pending_features == ()


def test_full_pipeline_on_the_paper_scenario(scam_text):
    evaluation = default_engine().evaluate(ingest(scam_text))
    fired = {fid for fid, o in evaluation.outcomes.items() if o.fired}
    for expected in (
        "email_free_provider",
        "contact_messaging_only",
        "payment_request_id",
        "salary_implausible_vs_umk",
        "risk_phrase_score_id",
    ):
        assert expected in fired, f"{expected} should fire on the paper scenario"


def test_legitimate_ad_stays_completely_clean(legit_text):
    """The false-positive guard, now across all nine rules."""
    evaluation = default_engine().evaluate(ingest(legit_text))
    fired = {fid for fid, o in evaluation.outcomes.items() if o.fired}
    assert fired == set(), f"legitimate ad fired: {fired}"
