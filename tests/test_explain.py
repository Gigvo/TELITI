"""Sentence-level occlusion — MVP_PLAN.md step 3.4.

The scoring function is injected, so these run without loading a transformer and
the expected influence of each sentence is known in advance.
"""

from __future__ import annotations

import pytest

from api.explain import (
    MIN_SENTENCE_CHARS,
    occlusion_evidence,
    split_sentences,
)

ENDPOINT = "/api/v1/analyze"


# ===========================================================================
# Sentence splitting
# ===========================================================================


def test_splits_on_sentence_punctuation():
    assert [s.text for s in split_sentences("Dibutuhkan admin. Gaji besar sekali.")] == [
        "Dibutuhkan admin.",
        "Gaji besar sekali.",
    ]


def test_splits_on_line_breaks():
    """Job ads are mostly line-broken rather than punctuated."""
    text = "LOWONGAN ADMIN\nGaji Rp5 juta\nInterview via Telegram"
    assert len(split_sentences(text)) == 3


def test_indonesian_thousand_separators_do_not_split():
    """'Rp9.000.000' is one number, not three sentences."""
    sentences = split_sentences("Gaji Rp9.000.000 per bulan untuk posisi ini.")
    assert len(sentences) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Pengalaman min. 2 tahun di bidang serupa. Kirim CV.", 2),
        ("Lowongan di PT. Maju Jaya. Hubungi kami sekarang.", 2),
        ("Kantor di Jl. Merdeka No. 45. Datang langsung ya.", 2),
    ],
)
def test_abbreviations_do_not_end_a_sentence(text, expected):
    """Without this, "min. 2 tahun" becomes two sentences and occlusion ends up
    measuring the effect of removing "min." on its own.

    The guard must include the DOT — `(?<!\\bpt\\.)`, not `(?<!\\bpt)`. Checking only
    the letters inspects the characters before the whitespace rather than before the
    period, and "PT. Maju Jaya" still split.
    """
    assert len(split_sentences(text)) == expected


def test_spans_index_into_the_original_text():
    """The frontend highlights by slicing the user's own string."""
    text = "Dibutuhkan admin online.\nGaji Rp9.000.000 per bulan.\nHubungi via Telegram."
    for sentence in split_sentences(text):
        assert text[sentence.start : sentence.end] == sentence.text


def test_empty_and_whitespace_text_yield_nothing():
    assert split_sentences("") == []
    assert split_sentences("   \n\n  ") == []


# ===========================================================================
# Occlusion
# ===========================================================================


SCAM_AD = (
    "Dibutuhkan admin online untuk perusahaan kami.\n"
    "Wajib membayar biaya administrasi Rp250.000.\n"
    "Kirim lamaran ke hrd@contoh.co.id sebelum akhir bulan."
)


def scorer_reacting_to(trigger: str, base: float = 3.0, drop: float = 2.0):
    """Fake model: any variant missing `trigger` scores `base - drop`.

    Lets a test assert which sentence the explanation blames, with the right answer
    fixed in advance.
    """

    def score(texts: list[str]) -> list[float]:
        return [base if trigger in text else base - drop for text in texts]

    return score


def test_identifies_the_sentence_the_model_reacts_to():
    evidence = occlusion_evidence(
        SCAM_AD, base_margin=3.0, margin_batch=scorer_reacting_to("biaya administrasi")
    )
    assert evidence
    top = evidence[0]
    assert "biaya administrasi" in top.text
    assert top.delta == pytest.approx(2.0)
    assert top.polarity.value == "risk"


def test_sign_convention_positive_means_the_sentence_carried_risk():
    """delta = margin(with) − margin(without). Positive means the ad looked safer
    once the sentence was removed."""
    evidence = occlusion_evidence(
        SCAM_AD, base_margin=3.0, margin_batch=scorer_reacting_to("biaya administrasi")
    )
    risky = [e for e in evidence if "biaya administrasi" in e.text][0]
    assert risky.delta > 0
    assert risky.polarity.value == "risk"


def test_sentence_that_makes_an_ad_look_safer_is_marked_safe():
    """Removing a legitimising sentence should RAISE the scam margin, giving a
    negative delta."""

    def score(texts: list[str]) -> list[float]:
        # Without the corporate email the ad looks worse.
        return [1.0 if "hrd@contoh.co.id" in t else 4.0 for t in texts]

    evidence = occlusion_evidence(SCAM_AD, base_margin=1.0, margin_batch=score)
    email = [e for e in evidence if "hrd@contoh.co.id" in e.text][0]
    assert email.delta < 0
    assert email.polarity.value == "safe"


def test_results_are_ranked_by_absolute_influence():
    evidence = occlusion_evidence(
        SCAM_AD, base_margin=3.0, margin_batch=scorer_reacting_to("biaya administrasi")
    )
    magnitudes = [abs(e.delta) for e in evidence]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_single_sentence_yields_no_evidence():
    """Removing the only sentence leaves an empty document, whose score says nothing
    about that sentence."""
    assert occlusion_evidence("Dibutuhkan admin online sekarang.", 2.0, lambda t: [0.0]) == []


def test_very_short_fragments_are_not_scored():
    """A bullet marker is not a claim worth a forward pass."""
    text = "-\n*\nDibutuhkan admin online untuk perusahaan kami di Jakarta."
    scored = occlusion_evidence(text, 2.0, lambda t: [0.0] * len(t))
    assert all(len(e.text) >= MIN_SENTENCE_CHARS for e in scored)


def test_sentence_cap_is_respected_and_calls_are_batched():
    """Every sentence costs a forward pass, so the cap bounds latency. It must also
    be ONE batched call, not one call per sentence."""
    text = "\n".join(f"Kalimat nomor {i} tentang lowongan kerja ini." for i in range(40))
    calls: list[int] = []

    def score(texts: list[str]) -> list[float]:
        calls.append(len(texts))
        return [1.0] * len(texts)

    occlusion_evidence(text, 2.0, score, max_sentences=10)
    assert len(calls) == 1, "expected a single batched call"
    assert calls[0] == 10


def test_skipped_sentences_are_reported_as_neutral_not_dropped():
    """An omitted sentence is indistinguishable from one that was checked and found
    harmless. Sentences past the cap are reported with delta 0 instead."""
    text = "\n".join(f"Kalimat nomor {i} tentang lowongan kerja ini." for i in range(12))
    evidence = occlusion_evidence(text, 2.0, lambda t: [2.0] * len(t),
                                  max_sentences=3, top_k=50)
    assert len(evidence) == 12
    assert all(e.delta == 0.0 for e in evidence[3:])


def test_removal_does_not_fuse_neighbouring_words():
    """Concatenating the remainder would produce "adminGaji" — the same defect that
    required a whole cleaning module for EMSCAD."""
    from api.explain import _text_without

    text = "Dibutuhkan admin online.\nGaji Rp9.000.000.\nHubungi kami."
    sentences = split_sentences(text)
    remainder = _text_without(text, sentences[1])
    assert "online.Hubungi" not in remainder
    assert "Gaji" not in remainder


def test_top_k_limits_the_returned_count():
    text = "\n".join(f"Kalimat nomor {i} tentang lowongan kerja ini." for i in range(10))
    assert len(occlusion_evidence(text, 2.0, lambda t: [1.0] * len(t), top_k=3)) == 3


# ===========================================================================
# Through the API
# ===========================================================================


def test_api_reports_evidence_as_exact_not_approximate(client, scam_text):
    """The keyword heuristic is gone; this is the model's own reasoning now."""
    body = client.post(ENDPOINT, json={"text": scam_text}).json()
    assert body["sentence_evidence_approximate"] is False
    assert body["sentence_evidence"], "expected occlusion evidence"


def test_api_evidence_spans_slice_the_analysed_text(client, scam_text):
    body = client.post(ENDPOINT, json={"text": scam_text}).json()
    analysed = body["analysed_text"]
    for evidence in body["sentence_evidence"]:
        span = evidence["span"]
        assert analysed[span["start"] : span["end"]] == evidence["text"]


def test_api_evidence_deltas_are_not_saturated(client, scam_text):
    """Probability-space occlusion collapsed to ~0.001 on a confident prediction,
    making the ranking noise. Margin space must keep real separation."""
    body = client.post(ENDPOINT, json={"text": scam_text}).json()
    top = max(abs(e["delta"]) for e in body["sentence_evidence"])
    assert top > 0.05, f"largest influence was {top}, suspiciously flat"
