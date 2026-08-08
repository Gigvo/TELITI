"""TELITI API — MVP_PLAN.md steps 1.1 / 2.5.

Serves the frozen contract in `api/schemas.py`. Every part of the scoring path is
real: ingest, the rule layer, the fine-tuned transformer, deployment calibration and
the fitted risk thresholds.

`/health` reports `model_loaded`. When it is false the artefacts failed to load and
analysis requests return 503 — the service starts and says so rather than refusing
to start or, worse, returning invented numbers.

Sentence evidence comes from leave-one-out occlusion (step 3.4): each sentence is
removed, the remainder re-scored, and the change in the model's logit margin reported
as that sentence's influence. It is the model's own reasoning, not a keyword list.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
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
    disclaimer_for,
    privacy_note_for,
)
from api.explain import occlusion_evidence
from api.feedback import store_report
from api.fetch_url import UrlFetchError, fetch_job_ad
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
    ReportRequest,
    ReportResponse,
    RiskLabel,
    RuleHit,
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

# Allowed browser origins.
#
# The Vite dev server is always permitted so local development needs no setup.
# Deployed frontends are added through TELITI_ALLOWED_ORIGINS, comma-separated:
#
#     TELITI_ALLOWED_ORIGINS=https://teliti.vercel.app,https://teliti-git-dev.vercel.app
#
# An allowlist rather than "*": the API is a different origin from the frontend
# once deployed, and a wildcard would let any site on the internet drive it using
# a visitor's browser. `allow_credentials` stays False — no cookies are involved,
# and the combination of credentials with a broad origin list is the classic way
# CORS gets misconfigured into a vulnerability.
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_EXTRA_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("TELITI_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS + _EXTRA_ORIGINS,
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


@app.post(
    f"{API_PREFIX}/report",
    response_model=ReportResponse,
    tags=["analysis"],
    summary="Ajukan banding atau koreksi label",
)
async def report(payload: ReportRequest) -> ReportResponse:
    """Accept an appeal against a score — concept paper §3.6.

    The case this exists for is a legitimate company scored *Tinggi*. §3.6 names
    false positives against real businesses as the error to suppress, and a system
    that can be wrong with no way to say so is worse than one that admits it.

    Two properties the response states rather than implies:

    - **The advertisement is stored.** Analysis persists nothing; filing a report is
      a separate, deliberate act, so the difference is reported back explicitly.
    - **Nothing here reaches the model.** Reports are quarantined for human review.
      Retraining on submitted labels would let anyone move any score in either
      direction, including a scammer clearing their own advertisement — which is
      exactly the poisoning risk the paper's Tahap 3 warns about.
    """
    cleaned = sanitize(payload.text)

    stored = store_report(
        correction=payload.correction.value,
        text=cleaned.text,
        reported_score=payload.reported_score,
        reported_label=payload.reported_label.value if payload.reported_label else None,
        request_id=payload.request_id,
        model_version=scam_model.info.version,
        comment=payload.comment,
        contact=payload.contact,
    )

    log.info(
        "correction filed: %s (%s) for request_id=%s",
        stored.report_id, stored.correction, payload.request_id,
    )

    return ReportResponse(
        report_id=stored.report_id,
        received_at=stored.received_at,
        message=(
            "Terima kasih. Laporan Anda disimpan untuk ditinjau manusia dan tidak "
            "digunakan untuk melatih ulang model."
        ),
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

    # Resolve the input. Exactly one of text/url is present — the request model
    # enforces that, so this cannot silently prefer one over the other.
    source_url: str | None = None
    if payload.url:
        try:
            page = fetch_job_ad(payload.url)
        except UrlFetchError as exc:
            log.info("URL fetch refused (%s): %s", exc.reason, payload.url[:200])
            # 422, not 502: from the client's perspective the submitted input could
            # not be used, which is the same class of problem as unusable text.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raw_text = page.text
        source_url = page.final_url
    else:
        raw_text = payload.text

    # Sanitise before anything reads the text (step 4.3). Length-preserving, so
    # span offsets still address what the client will render.
    cleaned = sanitize(raw_text)
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

    # One forward pass for the score, then one batched pass over the leave-one-out
    # variants for the explanation (step 3.4).
    base_margin = scam_model.margins([text])[0]
    probability = scam_model.calibrate_margin(base_margin)
    breakdown = compute_score(probability, evaluation)
    sentence_evidence = occlusion_evidence(text, base_margin, scam_model.margins)
    score = breakdown.integrity_score
    label = breakdown.risk_label

    return AnalyzeResponse(
        integrity_score=score,
        risk_label=label,
        model_probability=round(breakdown.model_probability, 4),
        fused_probability=round(breakdown.fused_probability, 4),
        summary=_summary(score, label, rule_hits, locale.code),
        sentence_evidence=sentence_evidence,
        sentence_evidence_approximate=False,
        rule_layer_enabled=RULE_LAYER_ENABLED,
        source_url=source_url,
        analysed_text=text,
        rule_hits=rule_hits,
        extracted_fields=ingested.fields,
        locale=locale.code,
        # `text`, not `payload.text` — on the URL path the latter is None, and the
        # language of a fetched page comes from what was fetched.
        locale_detected=detect_language(text),
        unassessed_rules=list(evaluation.unavailable_features),
        disclaimer=disclaimer_for(locale.code),
        privacy_note=privacy_note_for(locale.code),
        request_id=str(uuid.uuid4()),
        model_version=scam_model.info.version,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


# Registered LAST: StaticFiles at "/" would otherwise shadow /api and /health.
_mount_frontend()
