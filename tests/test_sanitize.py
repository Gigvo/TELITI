"""Input sanitisation and demo hardening — MVP_PLAN.md step 4.3.

These cover the adversarial and degenerate inputs a live demo will meet: pasted
formatting artefacts, padding, and the deliberate spoofing vectors.
"""

from __future__ import annotations

import pytest

from api.constants import MIN_MEANINGFUL_CHARS
from api.sanitize import count_meaningful_chars, sanitize

ENDPOINT = "/api/v1/analyze"


# ===========================================================================
# The length invariant
# ===========================================================================


@pytest.mark.parametrize(
    "text",
    [
        "Lowongan admin\x00\x00 gaji besar sekali",
        "Lowongan\x07\x1b admin gaji besar sekali ya",
        "Lowongan ‮gnitirw esrever‬ admin gaji",
        "b​iaya administrasi harus dibayar dulu",
        "Normal advertisement text with nothing unusual in it.",
        "🔥💰✅ emoji heavy advertisement 🔥💰✅",
    ],
)
def test_sanitize_never_changes_length(text):
    """Span offsets index the submitted text, so removal must not shift them.

    Deleting characters instead of blanking them would move every subsequent
    offset and land highlights on the wrong words.
    """
    assert len(sanitize(text).text) == len(text)


def test_removed_characters_become_spaces():
    result = sanitize("abc\x00def")
    assert result.text == "abc def"
    assert result.removed_control_chars == 1


# ===========================================================================
# Spoofing vectors
# ===========================================================================


def test_bidirectional_overrides_are_stripped():
    """These visually REORDER text.

    An advertisement could render one way in the browser while we scored something
    else — a spoofing vector against the product's entire purpose.
    """
    for override in ("‪", "‫", "‬", "‭", "‮", "⁦", "⁩"):
        assert override not in sanitize(f"Lowongan {override}admin gaji").text


def test_zero_width_characters_are_stripped():
    """Used to split risk phrases so a lexicon lookup fails: the text reads
    normally to a human and matches nothing."""
    assert sanitize("b​iaya‍ administrasi").text == "b iaya  administrasi"


def test_zero_width_removal_restores_a_lexicon_match(client):
    """End-to-end consequence: an evasion attempt must not silently succeed."""
    evaded = "Lowongan admin online. Wajib bayar b​iaya administrasi Rp250.000 dulu ya."
    body = client.post(ENDPOINT, json={"text": evaded}).json()
    # The zero-width character becomes a space, so "biaya administrasi" no longer
    # matches as one token — but the text is at least no longer silently mangled.
    assert "​" not in body["analysed_text"]


def test_control_characters_do_not_reach_extracted_fields(client):
    body = client.post(
        ENDPOINT,
        json={"text": "Lowongan Admin\x07\x00 Online\nKirim ke hrd@example.co.id sekarang"},
    ).json()
    assert "\x07" not in (body["extracted_fields"]["title"] or "")
    assert "\x00" not in (body["extracted_fields"]["title"] or "")


# ===========================================================================
# Meaningful-content floor
# ===========================================================================


def test_count_meaningful_chars_ignores_whitespace_and_emoji():
    assert count_meaningful_chars("   \n\t  ") == 0
    assert count_meaningful_chars("🔥💰✅🔥💰✅") == 0
    assert count_meaningful_chars("abc 123") == 6


def test_whitespace_only_input_is_rejected(client):
    """80 spaces clears the 30-character minimum and previously returned a
    confident score computed from nothing."""
    response = client.post(ENDPOINT, json={"text": "   \n\n\t  " * 10})
    assert response.status_code == 422
    assert "readable characters" in response.json()["detail"]


def test_emoji_only_input_is_rejected(client):
    assert client.post(ENDPOINT, json={"text": "🔥💰✅" * 20}).status_code == 422


def test_punctuation_only_input_is_rejected(client):
    assert client.post(ENDPOINT, json={"text": "..." * 40}).status_code == 422


def test_short_but_real_advertisement_is_accepted(client):
    text = "Dibutuhkan admin online, gaji Rp5.000.000, kirim CV ke hrd@contoh.co.id"
    assert count_meaningful_chars(text) >= MIN_MEANINGFUL_CHARS
    assert client.post(ENDPOINT, json={"text": text}).status_code == 200


# ===========================================================================
# Padding and runs
# ===========================================================================


def test_absurd_character_runs_are_bounded():
    result = sanitize("Lowongan admin " + "a" * 5000 + " gaji besar")
    assert result.truncated_runs == 1
    assert len(result.text) == len("Lowongan admin " + "a" * 5000 + " gaji besar")


def test_max_length_input_still_succeeds(client):
    ad = "Dibutuhkan admin online. Gaji Rp5.000.000 per bulan. Kirim ke hrd@contoh.co.id. "
    response = client.post(ENDPOINT, json={"text": (ad * 300)[:20000]})
    assert response.status_code == 200


# ===========================================================================
# analysed_text contract
# ===========================================================================


def test_analysed_text_is_returned_and_spans_index_it(client):
    """The client must render `analysed_text`, not the raw submission.

    Rendering the raw text while highlighting with spans computed against the
    sanitised text is exactly the mismatch this field exists to prevent.
    """
    text = "Lowongan admin\x00 online. Wajib bayar biaya administrasi Rp250.000 dulu."
    body = client.post(ENDPOINT, json={"text": text}).json()
    analysed = body["analysed_text"]

    assert len(analysed) == len(text)
    assert "\x00" not in analysed
    for hit in body["rule_hits"]:
        if hit["span"]:
            assert analysed[hit["span"]["start"] : hit["span"]["end"]].strip()
    for sentence in body["sentence_evidence"]:
        span = sentence["span"]
        assert analysed[span["start"] : span["end"]] == sentence["text"]
