"""Appeal / correction intake — concept paper §3.6.

Two invariants carry weight here and both are asserted rather than assumed: reports
must never reach training data, and the fact that filing one STORES the
advertisement must be visible to the person filing it.
"""

from __future__ import annotations

import json

import pytest

from api.feedback import load_reports, store_report, summarise

ENDPOINT = "/api/v1/report"

LEGIT_AD = (
    "Software Engineer (Backend) - PT Teknologi Nusantara\n"
    "Kualifikasi: S1 Ilmu Komputer, pengalaman minimal 2 tahun.\n"
    "Lamaran dikirim melalui https://karier.teknologinusantara.co.id"
)


@pytest.fixture
def reports_file(tmp_path, monkeypatch):
    """Point storage at a temp file so tests never touch the real queue."""
    path = tmp_path / "corrections.jsonl"
    import api.feedback as feedback

    monkeypatch.setattr(feedback, "REPORTS_PATH", path)
    return path


# ===========================================================================
# Storage
# ===========================================================================


def test_report_is_appended_and_readable(reports_file):
    stored = store_report(correction="false_positive", text=LEGIT_AD, path=reports_file)
    assert stored.report_id.startswith("rep-")

    reports = load_reports(reports_file)
    assert len(reports) == 1
    assert reports[0]["text"] == LEGIT_AD
    assert reports[0]["correction"] == "false_positive"


def test_storage_is_append_only(reports_file):
    """A second report must not overwrite the first."""
    for i in range(3):
        store_report(correction="false_positive", text=f"{LEGIT_AD} #{i}", path=reports_file)
    assert len(load_reports(reports_file)) == 3


def test_report_records_which_model_produced_the_disputed_score(reports_file):
    """Without this a report cannot be interpreted later — a complaint about a model
    two retrains ago may say nothing about the one now deployed."""
    store_report(
        correction="false_positive", text=LEGIT_AD,
        model_version="mdistilbert-256-pr0.8669", path=reports_file,
    )
    assert load_reports(reports_file)[0]["model_version"] == "mdistilbert-256-pr0.8669"


def test_reports_start_unreviewed(reports_file):
    """`reviewed` is set by a human afterwards, so the file records what was decided
    rather than only what was claimed."""
    store_report(correction="false_negative", text=LEGIT_AD, path=reports_file)
    assert load_reports(reports_file)[0]["reviewed"] is False


def test_overlong_comment_is_truncated_not_rejected(reports_file):
    """Losing the whole report over a long comment would be the wrong trade."""
    stored = store_report(
        correction="other", text=LEGIT_AD, comment="x" * 9000, path=reports_file
    )
    assert len(stored.comment) <= 2000


def test_blank_contact_is_stored_as_null(reports_file):
    stored = store_report(
        correction="other", text=LEGIT_AD, contact="   ", path=reports_file
    )
    assert stored.contact is None


def test_missing_file_reads_as_empty(tmp_path):
    assert load_reports(tmp_path / "nothing.jsonl") == []


def test_one_corrupt_line_does_not_hide_the_rest(reports_file):
    """A malformed line must lose one report, not the whole queue."""
    store_report(correction="false_positive", text=LEGIT_AD, path=reports_file)
    with reports_file.open("a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
    store_report(correction="false_negative", text=LEGIT_AD, path=reports_file)

    assert len(load_reports(reports_file)) == 2


def test_summary_counts_by_correction_type(reports_file):
    store_report(correction="false_positive", text=LEGIT_AD, path=reports_file)
    store_report(correction="false_positive", text=LEGIT_AD, path=reports_file)
    store_report(correction="false_negative", text=LEGIT_AD, path=reports_file)

    summary = summarise(reports_file)
    assert summary["total"] == 3
    assert summary["unreviewed"] == 3
    assert summary["by_correction"] == {"false_negative": 1, "false_positive": 2}


# ===========================================================================
# The endpoint
# ===========================================================================


def test_appeal_is_accepted(client, reports_file):
    response = client.post(
        ENDPOINT,
        json={
            "correction": "false_positive",
            "text": LEGIT_AD,
            "reported_score": 12,
            "reported_label": "Tinggi",
            "comment": "Ini perusahaan kami, lowongan ini asli.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["report_id"].startswith("rep-")
    assert body["received_at"]


def test_response_states_that_the_text_was_stored(client, reports_file):
    """Analysis persists nothing; filing a report does. The difference must be
    visible rather than assumed."""
    body = client.post(
        ENDPOINT, json={"correction": "false_positive", "text": LEGIT_AD}
    ).json()
    assert body["stored_text"] is True


def test_response_states_reports_are_not_used_for_training(client, reports_file):
    """The guarantee that stops this endpoint being a poisoning vector.

    Retraining on submitted labels would let anyone move any score in either
    direction — including a scammer clearing their own advertisement. Concept paper
    Tahap 3 requires reporter agreement and reviewer confirmation first.
    """
    body = client.post(
        ENDPOINT, json={"correction": "false_negative", "text": LEGIT_AD}
    ).json()
    assert body["used_for_training"] is False


def test_reports_are_not_written_anywhere_training_reads(client, reports_file, tmp_path):
    """Structural version of the guarantee: the quarantine file must not be one of
    the paths the training pipeline consumes."""
    from ml.eval_set import DEFAULT_HOLDOUT
    import api.feedback as feedback

    client.post(ENDPOINT, json={"correction": "false_negative", "text": LEGIT_AD})

    assert feedback.REPORTS_PATH.resolve() != DEFAULT_HOLDOUT.resolve()
    assert "processed" not in str(feedback.REPORTS_PATH)
    assert load_reports(reports_file), "report should still have been stored"


def test_submitted_text_is_sanitised(client, reports_file):
    """Same treatment as analysis: control characters and bidirectional overrides
    must not survive into the stored record."""
    client.post(
        ENDPOINT,
        json={"correction": "other", "text": LEGIT_AD + "\x00‮ injected"},
    )
    stored = load_reports(reports_file)[0]["text"]
    assert "\x00" not in stored
    assert "‮" not in stored


def test_disputed_analysis_can_be_linked_back(client, reports_file, scam_text):
    """A report is far more useful when it points at the analysis that caused it."""
    analysis = client.post("/api/v1/analyze", json={"text": scam_text}).json()
    client.post(
        ENDPOINT,
        json={
            "correction": "false_positive",
            "text": scam_text,
            "request_id": analysis["request_id"],
            "reported_score": analysis["integrity_score"],
        },
    )
    stored = load_reports(reports_file)[0]
    assert stored["request_id"] == analysis["request_id"]
    assert stored["model_version"] == analysis["model_version"]


@pytest.mark.parametrize(
    "payload",
    [
        {"correction": "false_positive"},                       # no text
        {"text": LEGIT_AD},                                     # no correction type
        {"correction": "not_a_type", "text": LEGIT_AD},
        {"correction": "other", "text": "too short"},
        {"correction": "other", "text": LEGIT_AD, "reported_score": 150},
        {"correction": "other", "text": LEGIT_AD, "unexpected": True},
    ],
)
def test_malformed_reports_are_rejected(client, reports_file, payload):
    assert client.post(ENDPOINT, json=payload).status_code == 422


def test_contact_is_optional(client, reports_file):
    """Requiring identification before someone may object would defeat the point."""
    assert (
        client.post(ENDPOINT, json={"correction": "other", "text": LEGIT_AD}).status_code
        == 200
    )
    assert load_reports(reports_file)[0]["contact"] is None


def test_report_endpoint_is_documented(client):
    """It is an ethics commitment (§3.6), so it belongs in the public contract."""
    assert ENDPOINT in client.get("/openapi.json").json()["paths"]
