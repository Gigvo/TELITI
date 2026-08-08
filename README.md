# TELITI

**Teknologi Evaluasi Lowongan dan Integritas** — an AI platform that scores the integrity of a job
advertisement *before* a jobseeker sends their CV and personal data, to protect them from job scams
and ghost jobs.

Paste a job ad, or give it a link, and get an **Integrity Score (0–100)**, a risk label
(Rendah / Sedang / Tinggi), and a plain-language explanation of *why* — with the specific sentences
that moved the score highlighted in the original text.

> The score is a risk indicator, not a verdict. See `api/constants.py::DISCLAIMER_ID`.

| | |
|---|---|
| **Status** | Working end to end. Real model, real explanations, 414 backend + 33 frontend tests passing. |
| **Plan** | [`MVP_PLAN.md`](MVP_PLAN.md) — day-by-day steps, gates, and fallbacks |
| **Model weights** | Hosted on Hugging Face — downloaded automatically on first run. See [`docs/PUBLISH_MODEL.md`](docs/PUBLISH_MODEL.md) |
| **Concept paper** | `SateManaAjaDah_ConceptPaper.pdf` |

---

## Quickstart

Requires **Python 3.11+** (developed on 3.13) and **Node 20+** (developed on 25).

### 1. Backend

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000
```

That is all. **The model weights download automatically on first start** (~516 MB, once — they are
cached afterwards) from the Hugging Face repository in [`api/artifacts.py`](api/artifacts.py). If
you already have `artifacts/scam_model/` locally, that is used instead and nothing is downloaded.

Open <http://127.0.0.1:8000/docs> for interactive API docs, and check
<http://127.0.0.1:8000/health> — `model_loaded` must be `true`.

### 2. Frontend

In a second terminal:

```bash
cd web && npm install && npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` and `/health` to FastAPI on port
8000, so the browser sees a single origin and there is no CORS preflight in development.

Both servers must be running: the UI shows an **API unreachable** banner otherwise.

### 3. Or just use Docker

One container serves both the API and the built frontend — no proxy, no CORS, no second terminal:

```bash
docker build -t teliti .
docker run --rm -p 8000:8000 teliti
```

Open <http://localhost:8000>. This is also exactly what gets deployed.

---

## Testing

```bash
.venv/Scripts/python.exe -m pytest          # 414 backend tests
cd web && npm test                          # 33 component/behaviour tests
cd web && npm run build                     # typecheck + production bundle
cd web && npm run lint
```

Against a running server — local, Docker, or deployed:

```bash
.venv/Scripts/python.exe scripts/smoke_test.py http://localhost:8000
```

18 end-to-end checks. This is the one that catches a **model-less deployment**: a container with no
weights starts cleanly, serves the UI, answers `/health`, and 503s every analysis. Nothing else
notices.

---

## What it does

| Layer | Module | What it does |
|---|---|---|
| Ingest | `api/ingest.py`, `api/fetch_url.py` | Clean text; extract title / salary / contact / location. Fetch and extract from a URL, behind an SSRF guard. |
| Text analysis | `ml/train_transformer.py` | Fine-tuned mDistilBERT → calibrated p(scam) |
| Metadata rules | `api/rules/` | Deterministic Indonesian checks: email domain, salary vs UMK, qualification conflict, risk phrases. **Advisory only — see below.** |
| Scoring | `api/scoring.py` | Calibrated probability → Integrity Score and risk label |
| Explanation | `api/explain.py` | Leave-one-out sentence occlusion in logit-margin space |
| Appeals | `api/feedback.py` | Append-only store for "this result is wrong" reports |

### API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/analyze` | Score an ad. Accepts `text` **or** `url`. |
| `POST /api/v1/report` | File an appeal against a result. |
| `GET /health` | Model, calibrator, threshold and locale status. |

`api/schemas.py` is the frozen contract — several people build against it in parallel. Additive
changes only; announce anything else.

---

## Results, stated honestly

Two evaluations on two different datasets. **The numbers are not comparable to each other**, because
PR-AUC scales with how common the positive class is.

**EMSCAD validation** (English, 1,788 items, 4.8% scams):

| Metric | Value |
|---|---|
| PR-AUC | 0.8669 |
| TF-IDF baseline | 0.8769 |
| ROC-AUC | 0.9767 |
| Gate 2.1 (needed ≥ 0.88) | **FAILED** |

The transformer did **not** beat a TF-IDF baseline on English EMSCAD. Say this out loud rather than
letting the Indonesian number imply otherwise.

**Indonesian holdout** (195 items, 36.4% scams) — `eval/indonesian_results.md`:

| Configuration | PR-AUC | False positives |
|---|---|---|
| **model only** | **0.9258** | 5 |
| model + rules | 0.8617 | 28 |
| rules only | 0.4167 | 93 |

### Why the rule layer is disabled

`RULE_LAYER_ENABLED = False` in `api/scoring.py`. Adding the rules made things **worse** — PR-AUC
fell from 0.9258 to 0.8617 and false positives went from 5 to 28. Concept paper §3.6 names false
positives against real businesses as the expensive error, so a change that multiplies them by five
does not ship.

The rules still run and are still shown as context. They just do not move the score. This is a
deliberate, measured decision, not an unfinished feature.

---

## Limitations to state when demoing

Do not let a demo imply more than the system does:

- Trained on **EMSCAD, an English corpus**, then recalibrated for Indonesian deployment. Indonesian
  performance is estimated from a 195-item holdout — indicative, not established.
- The transformer **did not beat the TF-IDF baseline** on EMSCAD validation.
- The rule layer is **disabled** for scoring.
- Appeals are stored for human review and are **not** used to retrain the model. The UI says so;
  keep saying so.

**Not in the MVP** (concept paper Table 1, Roadmap Tahap 3 — do not imply otherwise): ghost-job
detection, browser extension, B2B moderation API, crowdsourced labelling.

---

## Two files to read before writing any code

1. **`ml/feature_contract.py`** — defines what the model is allowed to see. EMSCAD ships metadata
   (`has_company_logo`, `has_questions`, …) that does not exist when a user pastes a WhatsApp
   message. A model that leans on those columns scores beautifully offline and performs at chance in
   production, and *nothing errors*. The contract is enforced with assertions for that reason.
2. **`api/schemas.py`** — the frozen API contract, as above.

---

## Repository layout

```
api/               FastAPI service, rule engine, scoring, explanations, URL fetch, appeals
ml/                Data prep, training, calibration, threshold fitting, evaluation
web/               React + Vite + TypeScript frontend
tests/             Rule golden cases, scoring invariants, API contract tests
scripts/           smoke_test.py (18 end-to-end checks), publish_to_hf.py
eval/              195-item Indonesian holdout + result reports — committed, NEVER trained on
data/raw/          EMSCAD (fake_job_postings.csv) — gitignored, download separately
data/reference/    UMK figures, Indonesian risk-phrase lexicon — committed
data/reports/      Filed appeals — gitignored
artifacts/         Trained model, calibrators, thresholds — gitignored, fetched from Hugging Face
docs/              Paper revision guide, model publishing, dataset spec, GPU handoff, locales
```

### Artifacts

Gitignored because the weights are 516 MB. **You do not need to obtain them manually** — they are
downloaded from Hugging Face on first run and cached. `api/artifacts.py` prefers a local copy when
one exists.

```
artifacts/
  scam_model/
    model.safetensors          516 MB
    config.json
    tokenizer.json
    tokenizer_config.json
    training_summary.json
    checkpoint/                1.6 GB — resume state only, NOT needed to serve or published
  calibrator.json              EMSCAD-fitted Platt scaling
  calibrator_deployment.json   Indonesian-recalibrated — this is the one used
  thresholds.json              Risk-label boundaries
```

`calibrator_deployment.json` matters. With the EMSCAD calibrator every Indonesian ad scored 93–100,
because the model carried a 4.8% base rate into a domain where scams are far more common. If scores
look implausibly high, that is why — regenerate with `python ml/fit_thresholds.py`.

---

## Running it in one container

The `Dockerfile` builds the React bundle and serves it from the same FastAPI process, so the whole
product runs from one image on one port — no proxy, no CORS, no second terminal:

```bash
docker build -t teliti .
docker run --rm -p 8000:8000 teliti
```

Open <http://localhost:8000>.

The image bakes in whatever is in `artifacts/` at build time. Building without them produces an
image that starts, serves the UI, and returns 503 from every analysis — run `scripts/smoke_test.py`
against it to catch that, since nothing else will.

Give the container **at least 2 GB RAM**; below that it is OOM-killed during start-up while loading
the model.

---

## Training

Only needed to reproduce the model. The transformer is fine-tuned on a machine with an
**RTX 5050 (Blackwell, sm_120)**. Install PyTorch there **first and separately** — a plain
`pip install torch` installs cleanly, reports `cuda.is_available() == True`, and then dies at the
first real matmul because the wheel contains no sm_120 kernels:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify before training anything — an availability check alone is not sufficient proof (Gate 0.2):

```bash
python -c "import torch; print(torch.cuda.get_device_capability(0)); print((torch.randn(4096,4096,device='cuda')@torch.randn(4096,4096,device='cuda')).sum().item())"
```

Then:

```bash
python ml/prepare_data.py        # EMSCAD → data/processed
python ml/train_transformer.py   # → artifacts/scam_model
python ml/fit_thresholds.py      # → calibrator_deployment.json, thresholds.json
```

See [`docs/GPU_HANDOFF.md`](docs/GPU_HANDOFF.md) for running this on someone else's machine, and
[`docs/INDONESIAN_DATASET_SPEC.md`](docs/INDONESIAN_DATASET_SPEC.md) for what the Indonesian dataset
must look like.
