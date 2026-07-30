"""TELITI API — Day 1 stub.

Serves the frozen contract in `api/schemas.py` with a FAKE scorer so that the
frontend and the rule layer can be built in parallel with model training.

Replacing the stub is MVP_PLAN.md step 2.5 (real model) and step 4.1 (fusion).
Until then `/health` reports `model_loaded: false`, and that is the single check
that tells you whether what you are looking at is real.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.constants import (
    API_PREFIX,
    PLACEHOLDER_THRESHOLDS,
    STUB_MODEL_VERSION,
    TOP_K_SENTENCE_EVIDENCE,
)
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    ExtractedFields,
    HealthResponse,
    Polarity,
    RiskLabel,
    RuleCategory,
    RuleHit,
    SentenceEvidence,
    Span,
)

log = logging.getLogger("teliti")

#: Flipped to True in step 2.5 when real artefacts are loaded.
MODEL_LOADED = False
THRESHOLDS_LOADED = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_LOADED:
        log.warning(
            "TELITI is running the DAY-1 STUB. Scores are synthetic and mean nothing. "
            "Do not screenshot this for the pitch."
        )
    yield


app = FastAPI(
    title="TELITI API",
    description=(
        "Menilai integritas lowongan kerja untuk melindungi pencari kerja dari "
        "job scam dan ghost job."
    ),
    version="0.1.0-mvp",
    lifespan=lifespan,
)

# Vite dev server. Tighten before any public deployment (Day 7).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a stack trace to the client; never lose it from the logs either."""
    request_id = str(uuid.uuid4())
    log.exception("Unhandled error (request_id=%s)", request_id)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Terjadi kesalahan internal. Silakan coba lagi.", request_id=request_id
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Stub scorer
# ---------------------------------------------------------------------------
#
# Deterministic (same text -> same score) so the frontend can write snapshot tests,
# and varied (different text -> different score) so it exercises all three risk
# labels during development. A handful of keyword nudges make the canonical demo
# texts land on the right side, which is the only reason the keywords exist.
# There is no intelligence here whatsoever.

_STUB_RISK_KEYWORDS = (
    "biaya administrasi",
    "biaya pelatihan",
    "transfer",
    "uang jaminan",
    "telegram",
    "tanpa pengalaman",
    "kuota terbatas",
    "gaji besar",
    "langsung kerja",
)

# Split on sentence punctuation followed by space, OR on a line break directly.
# Job ads are mostly line-broken rather than punctuated, so `\n` must be a
# separator in its own right — requiring trailing whitespace after it silently
# merges every unpunctuated line into its neighbour.
# Note this still splits "Rp9.000.000" safely: those dots have no whitespace after.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PHONE_RE = re.compile(r"(?:\+62|62|0)8[1-9][0-9]{6,11}")


def _stub_probability(text: str) -> float:
    """Hash-derived base probability, nudged by keyword presence."""
    digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
    base = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF  # uniform in [0, 1]

    lowered = text.lower()
    hits = sum(1 for kw in _STUB_RISK_KEYWORDS if kw in lowered)

    # Pull toward 1.0 as keywords accumulate, but never fully saturate.
    probability = base * 0.5 + min(hits, 4) * 0.15
    return max(0.0, min(0.99, probability))


def _label_for(score: int) -> RiskLabel:
    if score < PLACEHOLDER_THRESHOLDS["tinggi_below"]:
        return RiskLabel.TINGGI
    if score >= PLACEHOLDER_THRESHOLDS["rendah_at_or_above"]:
        return RiskLabel.RENDAH
    return RiskLabel.SEDANG


def _split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Naive splitter — replaced by the Indonesian-aware one in step 3.4."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in _SENTENCE_SPLIT.split(text):
        stripped = piece.strip()
        if not stripped:
            continue
        start = text.find(stripped, cursor)
        if start == -1:
            continue
        out.append((stripped, start, start + len(stripped)))
        cursor = start + len(stripped)
    return out


def _stub_sentence_evidence(text: str) -> list[SentenceEvidence]:
    scored: list[SentenceEvidence] = []
    for sentence, start, end in _split_sentences_with_spans(text):
        lowered = sentence.lower()
        hits = sum(1 for kw in _STUB_RISK_KEYWORDS if kw in lowered)
        delta = 0.12 * hits if hits else -0.01
        scored.append(
            SentenceEvidence(
                text=sentence,
                delta=round(delta, 4),
                polarity=Polarity.RISK if delta > 0 else Polarity.SAFE,
                span=Span(start=start, end=end),
            )
        )
    scored.sort(key=lambda s: abs(s.delta), reverse=True)
    return scored[:TOP_K_SENTENCE_EVIDENCE]


def _stub_rule_hits(text: str) -> list[RuleHit]:
    """Two illustrative rules so the frontend has real-shaped cards to render.

    The real implementations land in api/rules/ on Days 1-2 (steps 1.4 and 2.4).
    """
    hits: list[RuleHit] = []
    lowered = text.lower()

    free_providers = ("@gmail.com", "@yahoo.com", "@outlook.com", "@hotmail.com")
    for email in _EMAIL_RE.findall(text):
        if any(email.lower().endswith(p) for p in free_providers):
            start = text.find(email)
            hits.append(
                RuleHit(
                    rule_id="email_free_provider",
                    category=RuleCategory.COMPANY,
                    label_id="Kontak memakai email gratis, bukan domain perusahaan",
                    label_en="Contact uses a free email provider, not a company domain",
                    severity=0.6,
                    contribution=12.0,
                    evidence=email,
                    span=Span(start=start, end=start + len(email)),
                )
            )
            break

    for phrase in ("biaya administrasi", "biaya pelatihan", "uang jaminan"):
        if phrase in lowered:
            start = lowered.find(phrase)
            hits.append(
                RuleHit(
                    rule_id="payment_request_id",
                    category=RuleCategory.LANGUAGE,
                    label_id="Meminta pembayaran di awal proses rekrutmen",
                    label_en="Requests an up-front payment during recruitment",
                    severity=0.9,
                    contribution=15.0,
                    evidence=text[start : start + len(phrase)],
                    span=Span(start=start, end=start + len(phrase)),
                )
            )
            break

    return hits


def _stub_extracted_fields(text: str) -> ExtractedFields:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return ExtractedFields(
        title=lines[0][:120] if lines else None,
        emails=_EMAIL_RE.findall(text),
        urls=_URL_RE.findall(text),
        phones=_PHONE_RE.findall(text),
    )


def _stub_summary(score: int, label: RiskLabel, rule_hits: list[RuleHit]) -> str:
    if rule_hits:
        reasons = "; ".join(h.label_id.lower() for h in rule_hits)
        return (
            f"[STUB] Skor integritas {score}/100 (risiko {label.value}). "
            f"Indikasi yang ditemukan: {reasons}."
        )
    return (
        f"[STUB] Skor integritas {score}/100 (risiko {label.value}). "
        f"Tidak ada aturan deterministik yang terpicu."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=STUB_MODEL_VERSION,
        model_loaded=MODEL_LOADED,
        thresholds_loaded=THRESHOLDS_LOADED,
    )


@app.post(
    f"{API_PREFIX}/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Menilai integritas satu lowongan kerja",
)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    started = time.perf_counter()

    probability = _stub_probability(payload.text)
    score = round((1.0 - probability) * 100)
    label = _label_for(score)
    rule_hits = _stub_rule_hits(payload.text)

    return AnalyzeResponse(
        integrity_score=score,
        risk_label=label,
        model_probability=round(probability, 4),
        fused_probability=round(probability, 4),
        summary=_stub_summary(score, label, rule_hits),
        sentence_evidence=_stub_sentence_evidence(payload.text),
        rule_hits=rule_hits,
        extracted_fields=_stub_extracted_fields(payload.text),
        request_id=str(uuid.uuid4()),
        model_version=STUB_MODEL_VERSION,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
