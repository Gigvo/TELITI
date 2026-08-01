# TELITI

**Teknologi Evaluasi Lowongan dan Integritas** — an AI platform that scores the integrity of a job
advertisement *before* a jobseeker sends their CV and personal data, to protect them from job scams
and ghost jobs.

Paste a job ad → get an **Integrity Score (0–100)**, a risk label (Rendah / Sedang / Tinggi), and a
plain-language explanation of *why*.

> The score is a risk indicator, not a verdict. See `api/constants.py::DISCLAIMER_ID`.

- **Plan:** [`MVP_PLAN.md`](MVP_PLAN.md) — day-by-day steps, gates, and fallbacks.
- **Concept paper:** `SateManaAjaDah_ConceptPaper.pdf`
- **Status:** Day 1. API contract frozen; scoring is a **stub** (`/health` reports
  `model_loaded: false`).

---

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Run the API:

```bash
.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000/docs> for interactive API docs.

Run the tests:

```bash
.venv/Scripts/python.exe -m pytest
```

### Frontend

Requires Node 20+. In a second terminal:

```bash
cd web && npm install && npm run dev
```

Then open <http://localhost:5173>. The Vite dev server proxies `/api` and `/health`
to FastAPI on port 8000, so the browser sees a single origin and there is no CORS
preflight in development.

```bash
cd web && npm test          # component and behaviour tests
cd web && npm run build     # typecheck + production bundle
```

Both servers must be running: the UI shows an **API unreachable** banner otherwise,
and a **model is stubbed** banner until a real model is loaded.

### Training box only

The transformer is fine-tuned on a separate machine with an **RTX 5050 (Blackwell, sm_120)**.
Install PyTorch there **first and separately** — a plain `pip install torch` installs cleanly,
reports `cuda.is_available() == True`, and then dies at the first real matmul because the wheel
contains no sm_120 kernels:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify before training anything (this is Gate 0.2 in the plan — an availability check alone is not
sufficient proof):

```bash
python -c "import torch; print(torch.cuda.get_device_capability(0)); print((torch.randn(4096,4096,device='cuda')@torch.randn(4096,4096,device='cuda')).sum().item())"
```

---

## Architecture

Four layers, per concept paper §3.1:

| Layer | Module | What it does |
|---|---|---|
| Ingest | `api/ingest.py` | Clean text, extract title / salary / contact / location |
| Text analysis | `ml/train_transformer.py` → ONNX | Fine-tuned multilingual transformer → calibrated p(scam) |
| Metadata rules | `api/rules/` | Deterministic Indonesian checks: email domain, salary vs UMK, qualification conflict, risk phrases |
| Scoring + XAI | `api/scoring.py`, `api/explain.py` | Logistic-regression fusion + bounded penalties → Integrity Score, with sentence and rule evidence |

### Two files to read before writing any code

1. **`ml/feature_contract.py`** — defines what the model is allowed to see. EMSCAD ships metadata
   (`has_company_logo`, `has_questions`, …) that does not exist when a user pastes a WhatsApp
   message. A model that leans on those columns scores beautifully offline and performs at chance
   in production, and *nothing errors*. The contract is enforced with assertions for that reason.
2. **`api/schemas.py`** — the frozen API contract. Three people build against it in parallel.
   Additive changes only; announce anything else.

---

## Repository layout

```
data/raw/          EMSCAD (fake_job_postings.csv) — gitignored, download separately
data/reference/    UMK figures, Indonesian risk-phrase lexicon — committed
eval/              ~200-item manually annotated Indonesian holdout — committed, NEVER trained on
ml/                Data prep, training, calibration, fusion, ONNX export, evaluation
api/               FastAPI service, rule engine, scoring, explanations
web/               React + Vite + TypeScript frontend
tests/             Rule golden cases, scoring invariants, API contract tests
artifacts/         Trained model, calibrator, fusion model, thresholds — gitignored
```

## Scope

**In the MVP:** job-scam detection, Integrity Score, rule layer, XAI explanations, web app.

**Not in the MVP** (concept paper Table 1, Roadmap Tahap 3 — do not imply otherwise when demoing):
ghost-job detection, browser extension, B2B moderation API, crowdsourced labelling.
