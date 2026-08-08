"""TELITI API — MVP_PLAN.md steps 1.1 / 2.5.

Serves the frozen contract in `api/schemas.py`. Every part of the scoring path is
real: ingest, the rule layer, the fine-tuned transformer, deployment calibration and
the fitted risk thresholds.

`/health` reports `model_loaded`. When it is false the artefacts failed to load and
analysis requests return 503 — the service starts and says so rather than refusing
to start or, worse, returning invented numbers.

One piece remains approximate: `sentence_evidence` still uses keyword matching
rather than the occlusion-based explanation planned for step 3.4. It is marked
`approximate: true` in the response.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.constants import (
    API_PREFIX,
    TOP_K_SENTENCE_EVIDENCE,
    disclaimer_for,
    privacy_note_for,
)
from api.ingest import ingest
from api.locale import detect_language, load_registry
from api.model import MODEL_DIR, scam_model
from api.rules.engine import default_engine
from api.sanitize import sanitize
from api.scoring import RULE_LAYER_ENABLED, compute_score, load_thresholds
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    Polarity,
    RiskLabel,
    RuleHit,
    SentenceEvidence,
    Span,
)

log = logging.getLogger("teliti")

#: The rule layer is REAL as of step 1.4; the text model as of step 2.5. Nothing in
#: the scoring path is synthetic any more — `MODEL_LOADED` is now derived from
#: whether the artefacts actually loaded rather than being a hand-set flag.
_RULE_ENGINE = default_engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load eagerly at start-up so a broken artefact surfaces immediately in the logs
    # rather than on a user's first request during a demo.
    info = scam_model.info
    if info.loaded:
        log.info(
            "Text model ready: %s on %s, calibrator=%s, max_length=%d",
            info.version, info.device, info.calibrator, info.max_length,
        )
    else:
        log.error(
            "TEXT MODEL NOT LOADED (%s). The service will return 503 for analysis "
            "requests. Check that %s exists.",
            info.error, MODEL_DIR,
        )
    if _RULE_ENGINE.pending_features:
        log.info("Rule slots awaiting step 2.4: %s", ", ".join(_RULE_ENGINE.pending_features))

    registry = load_registry()
    log.info("Locales available: %s", ", ".join(registry.available()) or "none")
    for code, locale in sorted(registry.locales.items()):
        if locale.missing:
            log.warning(
                "Locale %r missing %s — its rules will report themselves unassessed. "
                "Drop the file(s) into data/reference/ to enable it.",
                code, ", ".join(locale.missing),
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
# Sentence evidence — interim, pending step 3.4
# ---------------------------------------------------------------------------
#
# The concept paper (§3.1) promises "which sentence is suspicious". Doing that
# properly means occlusion: remove each sentence, re-score, and rank by the change
# in p(scam). That is step 3.4.
#
# Until then this ranks sentences by keyword presence. It is a heuristic, NOT model
# output, so the response marks it `approximate: true` — a user must not believe the
# model pointed at a sentence when a word list did.

_RISK_KEYWORDS = (
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


def _keyword_sentence_evidence(text: str) -> list[SentenceEvidence]:
    scored: list[SentenceEvidence] = []
    for sentence, start, end in _split_sentences_with_spans(text):
        lowered = sentence.lower()
        hits = sum(1 for kw in _RISK_KEYWORDS if kw in lowered)
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


def _summary(score: int, label: RiskLabel, rule_hits: list[RuleHit], locale_code: str) -> str:
    """Narrative explanation, in the language of the ad.

    Replaced by the XAI composer in step 3.4, which will cite the specific sentences
    the model reacted to rather than only the rules that fired.
    """
    if locale_code == "id":
        if rule_hits:
            reasons = "; ".join(h.label_id.lower() for h in rule_hits)
            return (
                f"Skor integritas {score}/100 (risiko {label.value}). "
                f"Indikasi yang ditemukan: {reasons}."
            )
        return (
            f"Skor integritas {score}/100 (risiko {label.value}). "
            f"Tidak ada aturan deterministik yang terpicu."
        )

    english_label = {"Rendah": "Low", "Sedang": "Medium", "Tinggi": "High"}[label.value]
    if rule_hits:
        reasons = "; ".join(h.label_en.lower() for h in rule_hits)
        return (
            f"Integrity score {score}/100 ({english_label} risk). "
            f"Findings: {reasons}."
        )
    return (
        f"Integrity score {score}/100 ({english_label} risk). "
        f"No deterministic rules were triggered."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    registry = load_registry()
    _, thresholds_fitted = load_thresholds()
    info = scam_model.info
    return HealthResponse(
        status="ok" if info.loaded else "degraded",
        model_version=info.version,
        model_loaded=info.loaded,
        thresholds_loaded=thresholds_fitted,
        locales_available=list(registry.available()),
        locale_resources={
            code: list(locale.missing) for code, locale in sorted(registry.locales.items())
        },
    )


def _mount_frontend() -> None:
    """Serve the built React bundle from this process, if it is present.

    Lets the Docker image be a single container on a single port: no CORS, no
    reverse proxy, nothing extra to configure on an unfamiliar demo network.

    In development this directory does not exist and nothing is mounted — the Vite
    dev server handles the UI and proxies here, which keeps hot reload working.

    Mounted LAST so it never shadows /api or /health.
    """
    static_dir = os.environ.get("TELITI_STATIC_DIR", "web/dist")
    index = Path(static_dir) / "index.html"
    if not index.is_file():
        log.info("No frontend bundle at %s; serving API only.", static_dir)
        return

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    log.info("Serving frontend from %s", static_dir)


@app.post(
    f"{API_PREFIX}/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Menilai integritas satu lowongan kerja",
)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    started = time.perf_counter()

    # Sanitise before anything reads the text (step 4.3). Length-preserving, so
    # span offsets still address what the client will render.
    cleaned = sanitize(payload.text)
    if not cleaned.is_analysable:
        raise HTTPException(
            status_code=422,
            detail=(
                f"This text contains only {cleaned.meaningful_chars} readable "
                f"characters. Paste the full job advertisement so there is "
                f"something to analyse."
            ),
        )

    text = cleaned.text

    # Real: locale resolution + ingest + rule layer (steps 1.4 / 2.4).
    locale = load_registry().resolve(text, payload.locale)
    engine = default_engine(locale)

    ingested = ingest(text)
    evaluation = engine.evaluate(ingested)
    rule_hits = evaluation.to_rule_hits()

    # Real inference (step 2.5). The rule layer is advisory, so `compute_score`
    # returns the calibrated model probability unchanged — but it is still the one
    # place that applies the fitted thresholds and builds the breakdown, so the
    # score and its label can never disagree.
    if not scam_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "The scoring model is not available. This is a server-side problem, "
                "not a problem with the text you submitted."
            ),
        )

    probability = scam_model.probability(text)
    breakdown = compute_score(probability, evaluation)
    score = breakdown.integrity_score
    label = breakdown.risk_label

    return AnalyzeResponse(
        integrity_score=score,
        risk_label=label,
        model_probability=round(breakdown.model_probability, 4),
        fused_probability=round(breakdown.fused_probability, 4),
        summary=_summary(score, label, rule_hits, locale.code),
        sentence_evidence=_keyword_sentence_evidence(text),
        sentence_evidence_approximate=True,
        rule_layer_enabled=RULE_LAYER_ENABLED,
        analysed_text=text,
        rule_hits=rule_hits,
        extracted_fields=ingested.fields,
        locale=locale.code,
        locale_detected=detect_language(payload.text),
        unassessed_rules=list(evaluation.unavailable_features),
        disclaimer=disclaimer_for(locale.code),
        privacy_note=privacy_note_for(locale.code),
        request_id=str(uuid.uuid4()),
        model_version=scam_model.info.version,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


# Registered LAST: StaticFiles at "/" would otherwise shadow /api and /health.
_mount_frontend()
