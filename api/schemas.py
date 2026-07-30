"""TELITI API contract — MVP_PLAN.md step 1.1.

⚠️ THIS FILE IS FROZEN AS OF DAY 1.

Three people build against these shapes in parallel: the frontend renders them, the
backend fills them, the ML side feeds them. Changing a field name here costs everyone
a rebase, so additive changes only (new optional fields), and announce anything else.

Every model sets `protected_namespaces=()` because Pydantic v2 reserves the `model_`
prefix by default and would emit warnings for `model_probability` / `model_version`.
Those names are worth keeping — they say exactly what they are.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from api.constants import (
    DISCLAIMER_ID,
    MAX_TEXT_LENGTH,
    MIN_TEXT_LENGTH,
    PRIVACY_NOTE_ID,
    RISK_RENDAH,
    RISK_SEDANG,
    RISK_TINGGI,
)

_BASE = ConfigDict(protected_namespaces=(), extra="forbid")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskLabel(str, Enum):
    """Concept paper section 3.1: Rendah / Sedang / Tinggi."""

    RENDAH = RISK_RENDAH
    SEDANG = RISK_SEDANG
    TINGGI = RISK_TINGGI


class SourceChannel(str, Enum):
    """Where the user found the ad.

    Not a model input for the MVP — it is recorded for analytics and to let the
    rule layer adjust phrasing ("kontak hanya via Telegram" is more alarming when
    the ad already arrived through Telegram). Optional on purpose: never make the
    user answer a question before they can get a score.
    """

    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    INSTAGRAM = "instagram"
    JOB_BOARD = "job_board"
    WEB = "web"
    OTHER = "other"


class InferenceProfile(str, Enum):
    """Mirrors ml.feature_contract.PROFILES. See MVP_PLAN.md section 1.1."""

    TEXT_ONLY = "text_only"
    STRUCTURED = "structured"


class Polarity(str, Enum):
    RISK = "risk"
    SAFE = "safe"


class RuleCategory(str, Enum):
    CONTACT = "contact"
    COMPANY = "company"
    COMPENSATION = "compensation"
    QUALIFICATION = "qualification"
    LANGUAGE = "language"


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


class Span(BaseModel):
    """Character offsets into the ORIGINAL submitted text.

    Offsets must index the raw input, not the cleaned/normalised text, so the
    frontend can highlight in place without re-running our cleaning logic.
    """

    model_config = _BASE

    start: int = Field(ge=0)
    end: int = Field(ge=0)


class SentenceEvidence(BaseModel):
    """One sentence, scored by leave-one-out occlusion (MVP_PLAN.md section 1.4).

    `delta` is the change in p(scam) when this sentence is REMOVED. A large
    positive delta means removing it made the ad look safer, i.e. the sentence
    was carrying risk.
    """

    model_config = _BASE

    text: str
    delta: float = Field(description="p(scam) with sentence − p(scam) without it.")
    polarity: Polarity
    span: Optional[Span] = None


class RuleHit(BaseModel):
    """One deterministic rule that fired.

    `contribution` is in SCORE POINTS (0-100 scale) and signed the way a user reads
    it: positive means it pushed the integrity score DOWN. Learned rules get theirs
    from the fusion model's coefficients; penalty rules get theirs from their capped
    additive term.
    """

    model_config = _BASE

    rule_id: str = Field(description="Stable id, e.g. 'email_free_provider'.")
    category: RuleCategory
    label_id: str = Field(description="Human-readable label, Indonesian.")
    label_en: str = Field(description="Human-readable label, English.")
    severity: float = Field(ge=0.0, le=1.0)
    contribution: float = Field(description="Score points removed by this rule.")
    evidence: str = Field(description="The exact text fragment that triggered it.")
    span: Optional[Span] = None


class ExtractedFields(BaseModel):
    """What the ingest layer pulled out of the raw text (MVP_PLAN.md step 3.1 of the paper).

    Shown to the user so they can sanity-check our parsing — if we misread the
    salary, the salary rule's verdict should be visibly discountable.
    """

    model_config = _BASE

    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    salary_raw: Optional[str] = Field(default=None, description="Salary exactly as written.")
    salary_idr_monthly: Optional[int] = Field(
        default=None, description="Normalised to IDR/month; null if unparseable."
    )
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request / response
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    model_config = _BASE

    text: str = Field(
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="Raw job-ad text, exactly as the user copied it.",
    )
    source_channel: Optional[SourceChannel] = None
    profile: InferenceProfile = InferenceProfile.TEXT_ONLY


class AnalyzeResponse(BaseModel):
    model_config = _BASE

    integrity_score: int = Field(
        ge=0, le=100, description="0-100. Higher is safer. S = round((1 - p_final) * 100)."
    )
    risk_label: RiskLabel
    model_probability: float = Field(
        ge=0.0, le=1.0, description="Calibrated p(scam) from the text model ALONE, pre-fusion."
    )
    fused_probability: float = Field(
        ge=0.0, le=1.0, description="p_final after fusion and bounded penalties."
    )

    summary: str = Field(description="Narrative explanation in Indonesian.")
    sentence_evidence: list[SentenceEvidence] = Field(default_factory=list)
    rule_hits: list[RuleHit] = Field(default_factory=list)
    extracted_fields: ExtractedFields = Field(default_factory=ExtractedFields)

    disclaimer: str = DISCLAIMER_ID
    privacy_note: str = PRIVACY_NOTE_ID

    request_id: str
    model_version: str
    latency_ms: int = Field(ge=0)


class HealthResponse(BaseModel):
    model_config = _BASE

    status: str
    model_version: str
    model_loaded: bool = Field(
        description="False while running on the Day-1 stub. The demo must not "
        "be run with this False."
    )
    thresholds_loaded: bool


class ErrorResponse(BaseModel):
    model_config = _BASE

    detail: str
    request_id: Optional[str] = None
