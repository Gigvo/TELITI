"""API-level constants for TELITI.

Kept separate from `ml/feature_contract.py`: that module governs what the *model*
may see, this one governs what the *service* accepts and returns.
"""

from __future__ import annotations

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

#: Identifies which artefacts produced a score. Every response carries it so a
#: screenshot from a demo can always be traced back to a specific model build.
#: The `stub-` prefix means NO REAL MODEL IS LOADED — see api/main.py.
STUB_MODEL_VERSION = "stub-0.0.0"

# --- Input limits -----------------------------------------------------------
# Below MIN there is nothing to analyse and the score would be noise; above MAX
# we are being used as a document scanner, not a job-ad checker.
MIN_TEXT_LENGTH = 30
MAX_TEXT_LENGTH = 20_000

#: Minimum count of characters that actually carry meaning (letters and digits).
#:
#: Length alone is not enough: 80 spaces clears MIN_TEXT_LENGTH and produced a
#: confident-looking score computed from nothing. A job ad with fewer than this
#: many real characters cannot be assessed, and saying so is better than
#: returning a number.
MIN_MEANINGFUL_CHARS = 20

#: Characters that must never survive ingestion.
#:
#: - C0/C1 control codes: corrupt extracted fields and downstream logs.
#: - U+202A-U+202E, U+2066-U+2069: bidirectional overrides. These visually
#:   REORDER text, so an advertisement can render differently in the browser than
#:   the text we scored. That is a spoofing vector against the whole product:
#:   the user reads one thing and we analysed another.
#: - U+FEFF, U+200B-U+200D: zero-width characters used to break up risk phrases
#:   so a lexicon match fails ("b​iaya administrasi").
CONTROL_CHAR_PATTERN = (
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    r"​-‍‪-‮⁦-⁩﻿]"
)

#: Cap on how many sentences get the leave-one-out treatment in the XAI module.
#: Each one is a forward pass; this is what keeps us inside the <1s budget.
MAX_SENTENCES_FOR_OCCLUSION = 60
TOP_K_SENTENCE_EVIDENCE = 5

# --- Risk labels ------------------------------------------------------------
RISK_RENDAH = "Rendah"
RISK_SEDANG = "Sedang"
RISK_TINGGI = "Tinggi"

#: PLACEHOLDER ONLY. Real boundaries are derived on Day 3 (step 3.2) from a
#: precision target on the fraud class and written to artifacts/thresholds.json.
#: These round numbers exist so the stub can return something; the concept paper
#: (section 3.3) is explicit that shipped thresholds must not be chosen arbitrarily.
PLACEHOLDER_THRESHOLDS = {"tinggi_below": 40, "rendah_at_or_above": 70}

# --- Ethics -----------------------------------------------------------------
#: Concept paper section 3.6: "skor disajikan sebagai indikator risiko, bukan vonis".
#: This is a product requirement, not UI decoration. The frontend must display it.
DISCLAIMER_ID = (
    "Skor ini adalah indikator risiko, bukan vonis. TELITI dapat keliru. "
    "Gunakan hasil ini sebagai bahan pertimbangan, bukan sebagai kesimpulan akhir "
    "mengenai keabsahan lowongan atau perusahaan."
)

DISCLAIMER_EN = (
    "This score is a risk indicator, not a verdict. TELITI can be wrong. "
    "Treat this as one input to your own judgement, not as a final conclusion "
    "about the legitimacy of a posting or a company."
)

#: Concept paper section 3.6: no analysed text is persisted.
PRIVACY_NOTE_ID = "Teks yang Anda analisis tidak disimpan oleh sistem."
PRIVACY_NOTE_EN = "The text you analyse is not stored by this system."


def disclaimer_for(locale_code: str) -> str:
    return DISCLAIMER_ID if locale_code == "id" else DISCLAIMER_EN


def privacy_note_for(locale_code: str) -> str:
    return PRIVACY_NOTE_ID if locale_code == "id" else PRIVACY_NOTE_EN
