"""Contract tests for the TELITI API — MVP_PLAN.md gate 1.1.

These test the CONTRACT, not the stub's fake intelligence. They must keep passing
unchanged when the real model replaces the stub in step 2.5; if one of them has to
be edited then, the contract broke and the frontend broke with it.
"""

from __future__ import annotations

import pytest

from api.constants import MAX_TEXT_LENGTH, MIN_TEXT_LENGTH
from api.schemas import AnalyzeResponse, RiskLabel

ENDPOINT = "/api/v1/analyze"


# --- health -----------------------------------------------------------------


def test_health_reports_the_real_model_is_loaded(client):
    """Inverted at step 2.5, when the stub was replaced by the trained model.

    Until then this asserted `model_loaded is False`, which was how anyone could tell
    a real demo from a fake one. Now the same flag proves the opposite: if this fails,
    the artefacts are missing and the service is degraded.
    """
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True, (
        "artifacts/scam_model missing — copy it from the training machine"
    )
    assert body["status"] == "ok"
    assert body["thresholds_loaded"] is True, "run: python ml/fit_thresholds.py"
    assert not body["model_version"].startswith("stub")


def test_openapi_schema_is_served(client):
    """The frontend generates its TypeScript types from this."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert ENDPOINT in response.json()["paths"]


# --- response shape ---------------------------------------------------------


def test_analyze_returns_valid_response(client, scam_text):
    response = client.post(ENDPOINT, json={"text": scam_text})
    assert response.status_code == 200
    # Re-validating through the model catches any field the handler forgot.
    parsed = AnalyzeResponse.model_validate(response.json())
    assert 0 <= parsed.integrity_score <= 100
    assert parsed.risk_label in set(RiskLabel)
    assert parsed.summary
    assert parsed.disclaimer, "Ethics requirement, concept paper 3.6 — must never be empty"
    assert parsed.privacy_note
    assert parsed.request_id
    assert parsed.latency_ms >= 0


def test_score_and_probability_agree(client, scam_text):
    """S = round((1 - p_final) * 100). If these ever drift apart, scoring is broken."""
    parsed = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": scam_text}).json())
    assert parsed.integrity_score == round((1.0 - parsed.fused_probability) * 100)


def test_risk_label_is_consistent_with_score(client, scam_text, legit_text):
    """Label must be a pure function of the score — never independently assigned."""
    seen: dict[int, str] = {}
    for text in (scam_text, legit_text, scam_text.upper(), legit_text.replace("\n", " ")):
        parsed = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": text}).json())
        previous = seen.setdefault(parsed.integrity_score, parsed.risk_label.value)
        assert previous == parsed.risk_label.value


def test_analyze_is_deterministic(client, scam_text):
    """Same input, same score. The frontend snapshot-tests against this."""
    first = client.post(ENDPOINT, json={"text": scam_text}).json()
    second = client.post(ENDPOINT, json={"text": scam_text}).json()
    assert first["integrity_score"] == second["integrity_score"]
    assert first["fused_probability"] == second["fused_probability"]
    assert first["request_id"] != second["request_id"], "request_id must be per-request"


# --- evidence ---------------------------------------------------------------


def test_spans_index_into_the_original_text(client, scam_text):
    """Spans must address the RAW submitted text.

    The frontend highlights by slicing the user's own string. If offsets point into
    our cleaned/normalised text instead, highlights land on the wrong words — which
    looks like the model is nonsense even when the score is right.
    """
    parsed = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": scam_text}).json())

    for evidence in parsed.sentence_evidence:
        assert evidence.span is not None
        assert scam_text[evidence.span.start : evidence.span.end] == evidence.text

    # A rule's `evidence` is a human-readable explanation and may be a comparison
    # rather than a quotation ("Gaji ... setara 4.0x upah minimum ..."). Only the
    # SPAN must address the raw text, because that is what the frontend slices.
    for hit in parsed.rule_hits:
        if hit.span is not None:
            assert 0 <= hit.span.start < hit.span.end <= len(scam_text)
            assert scam_text[hit.span.start : hit.span.end].strip()


def test_rule_hits_are_well_formed(client, scam_text):
    parsed = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": scam_text}).json())
    assert parsed.rule_hits, "The canonical scam text must trigger at least one rule"
    for hit in parsed.rule_hits:
        assert hit.rule_id
        assert hit.label_id and hit.label_en, "Both languages required for the UI"
        assert 0.0 <= hit.severity <= 1.0
        assert hit.evidence


def test_sentence_evidence_is_ranked_by_magnitude(client, scam_text):
    parsed = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": scam_text}).json())
    deltas = [abs(e.delta) for e in parsed.sentence_evidence]
    assert deltas == sorted(deltas, reverse=True)


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "x" * (MIN_TEXT_LENGTH - 1)},
        {"text": "x" * (MAX_TEXT_LENGTH + 1)},
        {"text": ""},
        {},
        {"text": "x" * 100, "profile": "not_a_profile"},
        {"text": "x" * 100, "source_channel": "carrier_pigeon"},
        # extra="forbid" — a typo'd field must fail loudly, not be ignored.
        {"text": "x" * 100, "unexpected_field": True},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post(ENDPOINT, json=payload).status_code == 422


def test_valid_optional_fields_are_accepted(client, scam_text):
    response = client.post(
        ENDPOINT, json={"text": scam_text, "source_channel": "whatsapp", "profile": "text_only"}
    )
    assert response.status_code == 200


def test_unicode_and_emoji_do_not_break_spans(client):
    """Indonesian ads on WhatsApp are full of emoji. Offsets must survive them."""
    text = (
        "🔥 LOWONGAN KERJA URGENT 🔥\n"
        "Admin online, gaji Rp9.000.000, tanpa pengalaman 💰\n"
        "Wajib transfer biaya administrasi Rp250.000 ✅\n"
        "Hubungi kami sekarang juga, kuota terbatas!"
    )
    parsed = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": text}).json())
    for evidence in parsed.sentence_evidence:
        assert text[evidence.span.start : evidence.span.end] == evidence.text


# ===========================================================================
# Real model inference — MVP_PLAN.md step 2.5
# ===========================================================================


def test_score_comes_from_the_model_not_a_hash(client, scam_text, legit_text):
    """The stub derived scores from a SHA-256 of the text, so any two different
    strings produced unrelated numbers. A real model must separate a blatant scam
    from a clean posting."""
    scam = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": scam_text}).json())
    legit = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": legit_text}).json())
    assert scam.integrity_score < legit.integrity_score, (
        f"scam scored {scam.integrity_score}, legitimate ad scored "
        f"{legit.integrity_score} — the model is not separating them"
    )


def test_paper_scenario_is_flagged_high_risk(client, scam_text):
    """The concept paper's own §3.4 example. This is the demo; it must not regress."""
    body = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": scam_text}).json())
    assert body.risk_label.value == "Tinggi"


def test_legitimate_posting_is_not_flagged_high_risk(client, legit_text):
    """The false-positive guard §3.6 treats as the expensive error."""
    body = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": legit_text}).json())
    assert body.risk_label.value != "Tinggi"


def test_model_probability_is_calibrated_not_saturated(client, scam_text):
    """With the EMSCAD calibrator every Indonesian ad scored 93-100, because the
    model carried a 4.8% base rate into a domain where scams are far more common.
    A deployment-calibrated probability must use the range."""
    body = AnalyzeResponse.model_validate(client.post(ENDPOINT, json={"text": scam_text}).json())
    assert 0.0 <= body.model_probability <= 1.0
    assert body.integrity_score < 90, (
        f"score {body.integrity_score} suggests the EMSCAD calibrator is in use; "
        f"run python ml/fit_thresholds.py to produce calibrator_deployment.json"
    )


def test_identical_text_scores_identically(client, scam_text):
    """Inference must be deterministic — the model is in eval mode, no dropout."""
    first = client.post(ENDPOINT, json={"text": scam_text}).json()
    second = client.post(ENDPOINT, json={"text": scam_text}).json()
    assert first["integrity_score"] == second["integrity_score"]
    assert first["model_probability"] == second["model_probability"]


def test_sentence_evidence_is_no_longer_approximate(client, scam_text):
    """Inverted at step 3.4.

    Until then this asserted True, because the evidence came from a keyword list and
    the UI had to say so. It is now leave-one-out occlusion — the model's own
    reasoning — so the flag must be False and the caveat must disappear from the UI.
    """
    body = client.post(ENDPOINT, json={"text": scam_text}).json()
    assert body["sentence_evidence_approximate"] is False


def test_unavailable_model_returns_503_not_a_fake_score(monkeypatch, scam_text):
    """If the artefacts fail to load, the service must refuse to answer rather than
    inventing a number. A 503 is recoverable; a fabricated score is not."""
    from fastapi.testclient import TestClient

    import api.main as main

    class _Unloaded:
        is_loaded = False
        info = type("I", (), {"loaded": False, "version": "unloaded",
                              "calibrator": "none", "max_length": 256,
                              "device": "none", "error": "simulated"})()

    monkeypatch.setattr(main, "scam_model", _Unloaded())
    with TestClient(main.app) as unloaded_client:
        response = unloaded_client.post(ENDPOINT, json={"text": scam_text})
    assert response.status_code == 503
    assert "not available" in response.json()["detail"].lower()
