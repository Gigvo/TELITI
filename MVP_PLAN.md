# TELITI — MVP Development Plan (Tahap 1, compressed)

**Target:** shippable, demo-able MVP by **end of Day 4**. Days 5–8 = hardening, Indonesian
adaptation, and evaluation report.
**Team:** 3 people. **Training box:** RTX 5050 8GB + i5-13450HX. **Serving box:** CPU-only, 16GB.
**Source of truth:** `SateManaAjaDah_ConceptPaper.pdf` §3.1–§3.6.

---

## 0. The one-paragraph summary of what we are building

A FastAPI service that takes a raw job-ad text, runs it through (a) a fine-tuned multilingual
transformer and (b) a deterministic Indonesian rule layer, fuses both into a calibrated
`Integrity Score 0–100` via a logistic-regression meta-model, and returns the score plus
per-sentence and per-rule evidence. A React dashboard renders that evidence. Nothing else.

---

## 1. Critical design decisions (read this before Day 1)

These are the decisions where the "obvious" reading of the paper is the wrong engineering call.
Each one is justified by *time-to-result* or *correctness*, because we have 4 days.

### 1.1 Feature-availability contract — the most important rule in this project

EMSCAD ships metadata columns that are strongly predictive but **do not exist when a user pastes
a WhatsApp message**: `has_company_logo`, `has_questions`, `telecommuting`, `department`,
`industry`, `function`, `required_education`.

If the meta-model learns to lean on `has_company_logo`, offline metrics will look excellent and
the live product will be garbage. This is the single most likely way this project fails.

**Rule:** every feature fed to the model must be derivable from the pasted text alone.

We define two inference profiles and freeze them on Day 1:

| Profile | Input | Allowed features |
|---|---|---|
| `text_only` (**MVP ships this**) | pasted chat/ad text | title, description body, salary string, contact string, everything the rule layer extracts from text |
| `structured` (stretch, Day 7+) | job-board URL scrape | `text_only` + company profile presence, logo, application questions |

Day 1 deliverable is a hard-coded `ALLOWED_FEATURES[profile]` list, and the training script must
raise an error if a column outside that list reaches the model. Not a convention — an assertion.

### 1.2 Model choice: multilingual, not English-then-translate

The paper offers DistilBERT (English) with `translate-train` as fallback. Translating 17,880 docs
before we have a working pipeline is a day we don't have.

**Primary model: `distilbert-base-multilingual-cased` (mDistilBERT).** 6 layers, 134M params,
trained on 104 languages including Indonesian. Fine-tune once on English EMSCAD, get usable
zero-shot Indonesian transfer for free. Trains in ~12–18 min on the 5050.

**Escalation ladder** (only climb it if the gate below fails):
1. mDistilBERT zero-shot → measure on Indonesian eval set.
2. If Indonesian recall < 0.75: **translate-train-lite** (Day 5) — machine-translate only a
   stratified subset (all 866 fraud + ~3,500 sampled real ≈ 4.4k docs) with NLLB-200-distilled-600M
   locally on the 5050 (~30–40 min), append to the English set, retrain. Full-corpus translation
   is not needed and not worth it.
3. If still failing: swap backbone to `xlm-roberta-base` (better cross-lingual, ~2.5× slower) or
   IndoBERT on the translated-only corpus.

Do **not** start at step 3. Each rung costs a training run; the ladder exists so we only pay for
rungs we need.

### 1.3 Class imbalance: SMOTE is for the baseline only

EMSCAD is 4.84% fraud (866 / 17,880). The paper says "SMOTE or class weighting".

SMOTE interpolates between feature vectors. That is meaningful for TF-IDF vectors and **meaningless
for token sequences** — you cannot average two sentences into a valid sentence. Applying SMOTE to
transformer inputs is a common and silent mistake.

- **TF-IDF baselines:** SMOTE is fine (and we'll report it in the ablation).
- **Transformer:** weighted cross-entropy (`pos_weight ≈ 19.6`) + threshold tuning on the
  precision-recall curve. No resampling.

### 1.4 XAI: occlusion in production, LIME/SHAP in the report

LIME on a transformer needs ~1,000–5,000 forward passes per explanation. That is 10–60s on CPU.
Our latency target is <1s. LIME cannot be in the request path.

- **Production explanation = sentence-level occlusion.** Split the ad into sentences, re-run the
  model with each sentence removed, and rank sentences by Δp. A 20-sentence ad = 20 batched
  forward passes ≈ 150–300ms on quantized ONNX. This answers *exactly* the question the paper
  poses ("kalimat mana yang mencurigakan") and is more readable than token-level highlights.
- **Meta-model / rule contributions = exact.** The fusion model is logistic regression, so each
  rule's contribution is literally `coef_[i] * x[i]`. No approximation needed, zero cost.
- **LIME + SHAP still get built** (Day 6) as an offline analysis notebook, to validate that
  occlusion agrees with them and to produce the figures for the paper. Keep the paper's claim
  honest: LIME/SHAP are used in the methodology, occlusion is the deployed approximation.

### 1.5 Serving: ONNX INT8 on CPU

Export the fine-tuned model to ONNX and apply dynamic INT8 quantization
(`optimum[onnxruntime]`). Expect ~3–4× speedup and ~4× smaller model vs PyTorch CPU, with <1%
F1 drop. Verify the drop, don't assume it. This is what makes the <1s target achievable and
matches the paper's §3.6 ops description.

---

## 2. Repository layout (create on Day 1, don't reorganize later)

```
TELITI/
├── data/
│   ├── raw/                     # fake_job_postings.csv (EMSCAD), gitignored
│   ├── processed/               # train/val/calib/test splits, gitignored
│   └── reference/
│       ├── umk_2025.json        # UMK per province/city — committed
│       └── risk_phrases_id.yaml # Indonesian risk lexicon — committed
├── eval/
│   └── indonesian_holdout.jsonl # ~200 manually annotated items — committed
├── ml/
│   ├── prepare_data.py          # load, clean, split, enforce feature contract
│   ├── train_baseline.py        # TF-IDF × {LogReg, LinearSVC, XGBoost}
│   ├── train_transformer.py     # mDistilBERT fine-tune
│   ├── calibrate.py             # Platt scaling + reliability diagram
│   ├── train_fusion.py          # LR meta-model (stacking)
│   ├── export_onnx.py           # ONNX + INT8 quantization + parity check
│   └── evaluate.py              # all metrics, EMSCAD + Indonesian holdout
├── api/
│   ├── main.py                  # FastAPI app
│   ├── schemas.py               # Pydantic request/response — THE contract
│   ├── pipeline.py              # ingest → text → rules → fuse → explain
│   ├── ingest.py                # cleaning + field extraction
│   ├── rules/
│   │   ├── engine.py            # runs all rules, returns RuleHit[]
│   │   ├── email_domain.py
│   │   ├── salary_sanity.py     # vs UMK
│   │   ├── qualification.py     # fresh-grad + 5yr-experience conflict
│   │   └── risk_phrases.py
│   ├── scoring.py               # meta-model + bounded penalties + thresholds
│   └── explain.py               # sentence occlusion + rule contributions
├── web/                         # React + Vite + TypeScript + Tailwind
├── tests/
│   ├── test_rules.py            # golden cases per rule
│   ├── test_scoring.py          # monotonicity, bounds, threshold invariants
│   └── test_api.py              # contract tests
├── artifacts/                   # model.onnx, calibrator.pkl, fusion.pkl (gitignored)
├── notebooks/                   # exploration + LIME/SHAP analysis (Day 6)
├── requirements.txt
└── MVP_PLAN.md
```

---

## 3. Team split (parallelism is what makes 4 days possible)

| Role | Owns | Person |
|---|---|---|
| **A — ML** | `ml/`, model training, calibration, fusion, metrics | _assign_ |
| **B — Backend** | `api/`, rules, ONNX serving, XAI module, tests | _assign_ |
| **C — Data + Frontend** | `eval/`, `data/reference/`, `web/` | _assign_ |

The Indonesian eval set (C) is the longest-pole *human* task and gates our headline metric.
It starts Hour 1 of Day 1 and runs in background all week. Do not let it slip to Day 4.

**Day 1, Hour 1, all three together:** freeze `api/schemas.py`. Once the JSON contract exists,
B can build against a stub model and C can build against a mocked API. Nobody blocks.

---

## 4. Day-by-day plan

Each step has a **Gate** — a concrete, checkable pass condition. Do not start the next step until
the gate is green. If a gate fails, the fallback is written next to it.

---

### DAY 0 — Environment (2–3 hours, do this the evening before Day 1)

**0.1 Serving box (this machine, CPU)**
```bash
python -m venv .venv && source .venv/Scripts/activate
pip install fastapi uvicorn pydantic onnxruntime scikit-learn pandas numpy pyyaml pytest
```

**0.2 Training box (RTX 5050) — this is the step that will bite you**

The RTX 5050 is Blackwell (compute capability sm_120). PyTorch wheels built for CUDA ≤12.6 do not
contain sm_120 kernels and will fail at runtime with "no kernel image is available for execution on
the device" — *after* appearing to install fine.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets accelerate evaluate scikit-learn optimum[onnxruntime]
```

**Gate 0.2:** the following prints `True` and a non-empty capability ≥ `(12, 0)`:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```
Then run one real matmul on GPU (`torch.randn(4096,4096,device='cuda') @ ...`) — `is_available()`
returning True is not sufficient proof that kernels exist for sm_120.
*Fallback if this fails:* PyTorch nightly cu128, or train on Colab T4 as originally planned.

**0.3 Dataset**
Download EMSCAD (`fake_job_postings.csv`, ~50MB) from the Kaggle "Real / Fake Job Posting
Prediction" dataset → `data/raw/`.

**Gate 0.3:** `df.shape == (17880, 18)` and `df.fraudulent.sum() == 866`.

**0.4** `git init`, create the tree from §2, commit the skeleton.

---

### DAY 1 — Contract, data, baseline, rule layer skeleton

#### Step 1.1 — Freeze the API contract (all three, 60–90 min) ⚠️ blocking

Write `api/schemas.py` with the full request/response shape. Proposed:

```
POST /api/v1/analyze
  request  { text: str, source_channel?: "whatsapp"|"web"|"other", profile: "text_only" }
  response {
    integrity_score: int,              # 0-100
    risk_label: "Rendah"|"Sedang"|"Tinggi",
    model_probability: float,          # calibrated p(scam) from text model
    summary: str,                      # human-readable narrative
    sentence_evidence: [               # from occlusion, top-5
      { text, delta, polarity: "risk"|"safe" }
    ],
    rule_hits: [
      { rule_id, label_id, label_en, severity, contribution, evidence }
    ],
    extracted_fields: { title?, salary?, contact?, location? },
    latency_ms: int,
    model_version: str
  }
```

Include a `disclaimer` field carrying the paper's ethical position: this is a risk indicator, not
a verdict. It is a product requirement (§3.6), not decoration.

**Gate 1.1:** `uvicorn` serves `/docs`, and `POST /api/v1/analyze` returns a hard-coded fake
response matching the schema. Commit. **From this moment B and C are unblocked.**

#### Step 1.2 — Data preparation (A, 2 hours)

`ml/prepare_data.py`:
- Concatenate text fields into one document: `title + "\n" + description + "\n" + requirements`.
  Keep `company_profile` and `benefits` out of the `text_only` document — a WhatsApp ad has neither.
- Strip HTML, normalize whitespace/URLs, lowercase only for the TF-IDF branch (transformer keeps case).
- Split **stratified 70/10/10/10** → `train / val / calib / test`. Four splits, deliberately:
  `val` tunes hyperparameters, `calib` fits Platt scaling **and** trains the fusion meta-model,
  `test` is touched exactly once at the end. Reusing val for calibration produces overconfident
  probabilities — that quietly breaks the whole "calibrated score" claim.
- Assert the feature contract from §1.1.
- Fixed `random_state=42`, save split indices to disk.

**Gate 1.2:** each split has the fraud ratio within ±0.5pp of 4.84%; splits are disjoint (assert
on index intersection); rerunning the script reproduces byte-identical files.

#### Step 1.3 — Baseline models (A, 2 hours)

`ml/train_baseline.py`: TF-IDF (word 1–2gram, `min_df=3`, `max_features=50k`) × {LogisticRegression
`class_weight=balanced`, LinearSVC, XGBoost `scale_pos_weight=19.6`}, plus one SMOTE variant.

Report on `val`: **PR-AUC (primary), F1-fraud, precision/recall at best-F1 threshold, ROC-AUC.**
Accuracy is reported but explicitly labelled as uninformative — a model predicting "all real" scores
95.2%.

**Gate 1.3:** best baseline **PR-AUC ≥ 0.80** on val. This is our floor and our fallback model. If
the transformer somehow underperforms this, we ship this instead and say so honestly.

*Why bother:* it's 20 minutes of compute, it de-risks Day 2 entirely, and it's a required row in
the ablation table.

#### Step 1.4 — Rule engine skeleton + first two rules (B, 3 hours)

`api/rules/engine.py` with a uniform interface: every rule returns
`RuleHit(rule_id, fired: bool, severity: float 0-1, evidence: str, span: (start,end)|None)`.

Implement first:
- `email_domain` — free provider (gmail/yahoo/outlook/proton...) vs corporate; no email at all;
  domain/company-name mismatch.
- `contact_channel` — WhatsApp/Telegram link or "hubungi via Telegram" as the *only* contact route.

**Gate 1.4:** `pytest tests/test_rules.py` green, with ≥5 golden cases per rule including
negatives (a legitimate ad must not fire it). Every rule ships with its test in the same commit.

#### Step 1.5 — Indonesian eval set kickoff (C, 3 hours + background all week)

Target **200 items**: ~60–70 confirmed scams, ~130–140 legitimate. Sources:
- Scam: cases published by Bareskrim/Kominfo/Jobstreet warnings, Instagram `@lowongan_palsu`-type
  watchdog accounts, media reporting, r/indonesia + Twitter/X threads with screenshots, Kaskus.
- Legitimate: Jobstreet, Glints, Kalibrr, Karir.com, university career centers, real company
  Instagram posts.

Rules for the set:
- Store as JSONL: `{id, text, label, source_url, annotator, notes, channel}`.
- **Two annotators per item**, disagreements resolved by the third. Record Cohen's κ.
- **Never used for training.** This is the held-out set that produces our headline number.
- While annotating, log every suspicious phrase into `risk_phrases_id.yaml`. The lexicon is a
  *byproduct* of annotation, which is why annotation starts on Day 1.

**Gate 1.5 (end of Day 1):** ≥50 items collected, schema validated, κ computed on the first 30.

#### Step 1.6 — Frontend shell (C, 2 hours)

`npm create vite@latest web -- --template react-ts`, add Tailwind, build the single-screen layout:
textarea → Analyze button → score gauge + risk badge + evidence list. Wire to the stub endpoint
from 1.1. Configure the Vite dev proxy to FastAPI now, not later.

**Gate 1.6:** paste text → see the stubbed score rendered in the browser.

---

### DAY 2 — Transformer, calibration, rule layer complete

#### Step 2.1 — Fine-tune mDistilBERT (A, 3–4 hours incl. debugging)

`ml/train_transformer.py`. Config to start from:

```
model            distilbert-base-multilingual-cased
max_length       256          # covers ~90% of EMSCAD ads; 512 doubles cost for ~2pp
batch_size       32           # fits 8GB with fp16; drop to 16 if OOM
lr               2e-5, linear warmup 10%
epochs           3            # 4-5 overfits at this imbalance
loss             weighted CE, pos_weight = 19.6
precision        fp16 (torch.amp)
eval             every 200 steps on val, select best PR-AUC (NOT best loss)
seed             42
```

Expect ~12–18 min/run on the 5050. Budget for 3 runs.

**Gate 2.1:** val **PR-AUC ≥ 0.88** and **recall-fraud ≥ 0.85** at the operating threshold.
*Fallback ladder if missed:* (a) max_length 384; (b) 4 epochs with lr 3e-5; (c) unfreeze-all with
lower lr; (d) accept the baseline from 1.3 and reallocate the day. Do not exceed 3 extra runs —
past that, the time is better spent on the Indonesian gap, which is the real risk.

**Sanity check before trusting any number:** manually inspect 20 top-confidence fraud predictions.
If the model is keying on an artifact (e.g. a specific boilerplate string present only in fraud
rows), you'll see it immediately. Ten minutes here has saved entire projects.

#### Step 2.2 — Zero-shot Indonesian probe (A, 30 min)

Run the Day-1 partial Indonesian set (~50–80 items) through the model. **This number will be
worse than EMSCAD — that is expected and it is exactly what the paper commits to reporting
openly (§3.2).** We need it now, on Day 2, because it decides whether Day 5 does translate-train.

**Gate 2.2:** number recorded in `eval/results.md` with the item count. No pass/fail — it's a
decision input. Decision rule: **recall < 0.75 → translate-train-lite is scheduled for Day 5.**

#### Step 2.3 — Probability calibration (A, 1.5 hours)

`ml/calibrate.py`: fit Platt scaling (`LogisticRegression` on raw logits) on the `calib` split.
Report **Brier score** and **Expected Calibration Error (10 bins)** before/after, and save a
reliability diagram PNG.

**Gate 2.3:** ECE ≤ 0.05 after calibration and strictly better than before. Without this, "Integrity
Score 73" is a number with no meaning, and the calibration claim in the paper is unsupported.

#### Step 2.4 — Remaining rules (B, 4 hours)

- `salary_sanity` — parse Indonesian salary strings ("Rp9jt", "9 juta/bulan", "8-12jt",
  "Rp 9.000.000"), compare against `umk_2025.json` for the detected region (default: national
  median if no region found). Fire on `salary > 3× UMK` for an entry-level/no-experience role.
  Also fire on *absurdly vague* ("gaji besar", "penghasilan tak terbatas").
- `qualification_conflict` — "fresh graduate"/"tanpa pengalaman" co-occurring with "pengalaman
  min. N tahun"; also degree-vs-role mismatch.
- `risk_phrases` — YAML lexicon with weights: "biaya administrasi", "biaya pelatihan", "transfer",
  "interview via Telegram", "data KTP", "uang jaminan", "kuota terbatas", "langsung kerja",
  urgency markers. Weighted, not binary — one weak phrase must not fire a red flag.

**Gate 2.4:** all rules tested; run the rule layer alone over the Indonesian eval set and confirm
**rule-only precision ≥ 0.70** on items where ≥1 rule fires. If a rule has precision < 0.5 in
isolation, its weight gets capped hard in §2.5 or it gets cut.

#### Step 2.5 — Real inference in the API (B, 2 hours)

Replace the stub: load the PyTorch checkpoint (ONNX comes Day 3), run text model + rule layer,
return real `model_probability` and `rule_hits`. Fusion not yet wired — use a documented
placeholder combination.

**Gate 2.5:** end-to-end `curl` on a real scam text returns a plausible score in <5s.

---

### DAY 3 — Fusion, thresholds, XAI, ONNX

#### Step 3.1 — Fusion meta-model (A, 3 hours) — the core of §3.3

`ml/train_fusion.py`. On the `calib` split only:

```
features  = [ calibrated_p_text ] + [ rule feature vector, EMSCAD-derivable subset ]
model     = LogisticRegression(class_weight='balanced')
output    = p_fusi
```

Only rules that EMSCAD can express (email domain, contact channel, qualification conflict) get
*learned* weights. The Indonesia-only signals (UMK salary sanity, Indonesian risk phrases) have no
EMSCAD support, so per the paper they are applied as **bounded additive penalties**:

```
p_final = clip( p_fusi + Σ min(penalty_i, cap_i), 0, 1 )
   with  Σ cap_i ≤ 0.15     # no deterministic rule can dominate the model
```

Then: `S = round((1 - p_final) × 100)`.

**Gate 3.1:** fusion PR-AUC on `calib` > text-model-alone PR-AUC (if not, the rules add nothing —
report that honestly and simplify). All meta-model coefficients on risk features must be
**positive** — a negative coefficient on "free email domain" means the model learned something
backwards from EMSCAD's quirks, and that feature must be dropped, not shipped.

#### Step 3.2 — Threshold selection (A, 1 hour)

Set Rendah/Sedang/Tinggi cutoffs from a **precision target on the fraud class**, per §3.3 — not
round numbers.

- `Tinggi` boundary: highest score at which fraud precision ≥ 0.85 on val.
- `Rendah` boundary: lowest score at which fraud recall ≥ 0.95 (i.e. below this we're confident
  it's safe).
- Everything between = `Sedang`.

**Gate 3.2:** thresholds written to `artifacts/thresholds.json` with the precision/recall figures
that produced them recorded alongside. Anyone must be able to justify each number.

#### Step 3.3 — ONNX export + quantization (A, 1.5 hours)

`ml/export_onnx.py` via `optimum.onnxruntime`, then dynamic INT8.

**Gate 3.3:** (a) ONNX FP32 outputs match PyTorch within 1e-4; (b) INT8 F1 drop < 1pp vs FP32
on test — **measure it**; (c) p95 single-item latency on the CPU box < 400ms, leaving headroom
for occlusion.
*Fallback:* if INT8 degrades too much, ship FP32 ONNX and reduce occlusion to top-10 sentences.

#### Step 3.4 — XAI module (B, 3 hours)

`api/explain.py`:
- Sentence split (Indonesian-aware: handle "Rp9.000.000" and "min." without splitting on them).
- Batch all leave-one-sentence-out variants in **one** ONNX call.
- Rank by Δp, return top-5 with polarity.
- Rule contributions = `coef × value`, exact, plus a template-generated Indonesian sentence per
  fired rule.
- Compose `summary` from the top rule hits + top sentence.

**Gate 3.4:** on the WhatsApp scenario from §3.4 of the paper ("admin online, gaji Rp9 juta, tanpa
pengalaman"), the top-ranked sentence is the one a human would pick, and total latency < 1s.
This exact case goes into `tests/` as a regression test — it's the demo, so it must never break.

#### Step 3.5 — Frontend real integration (C, 4 hours)

Score gauge (color-coded by risk label), sentence evidence with the risky sentences highlighted
inline in the original text, rule cards with plain-Indonesian explanations, extracted fields panel,
loading and error states, the disclaimer banner.

**Gate 3.5:** full flow works against the live API on 3 different real ads.

---

### DAY 4 — Integration, evaluation, demo-ready 🎯

#### Step 4.1 — Wire fusion into the API (B, 2 hours)
Load `fusion.pkl` + `thresholds.json` + INT8 ONNX. Full pipeline live.
**Gate:** `pytest tests/` fully green, including the §3.4 regression test.

#### Step 4.2 — Final evaluation (A, 3 hours)
Touch `test` split **once**. Run `ml/evaluate.py` producing `eval/results.md`:

| Report | Content |
|---|---|
| EMSCAD test | PR-AUC, F1, precision, recall, confusion matrix, ROC |
| Indonesian holdout (zero-shot) | same metrics + item count + κ — **the headline number** |
| Calibration | Brier, ECE, reliability diagram |
| Ablation | text-only / rules-only / fused; TF-IDF baseline vs transformer |
| Latency | p50/p95/p99 on CPU, with and without occlusion |
| Error analysis | 10 worst false positives + 10 worst false negatives, with commentary |

**Gate 4.2:** every number in the eventual paper/pitch traces to a line in this file. No number
gets quoted anywhere that isn't here.

#### Step 4.3 — Demo hardening (B+C, 3 hours)
Input length caps, empty/garbage input handling, model-load-failure fallback, CORS, request
timeouts, a `/health` endpoint, and 6–8 curated demo texts as one-click examples in the UI.

**Gate 4.3:** two team members independently run the demo end-to-end on a clean checkout using
only the README. If either gets stuck, the README is wrong — fix it now, not on demo day.

#### Step 4.4 — Freeze (30 min)
Tag `v0.1-mvp`. **From here, everything else is additive.** If Day 5–8 goes badly, we still have
a working product.

---

### DAYS 5–8 — Hardening (priority-ordered; take from the top, drop from the bottom)

**Day 5 — Close the Indonesian gap** (the highest-value remaining work)
- If Gate 2.2 said recall < 0.75: translate-train-lite. NLLB-200-distilled-600M on the 5050,
  translate 866 fraud + 3,500 sampled real, append to English train set, retrain, re-evaluate.
- Expand the Indonesian eval set toward the full 200. Recompute the headline number.
- Refit the risk-phrase lexicon from newly annotated items.

**Day 6 — Evidence quality**
- LIME + SHAP analysis notebook; measure rank correlation between LIME and our occlusion ranking.
  If they agree, the paper's XAI claim is substantiated and the production shortcut is justified.
- False-positive campaign: run 100 *legitimate* Indonesian ads from real companies. Tune thresholds
  to push FP down — §3.6 names this explicitly as an ethics requirement, and a system that flags
  real companies is worse than useless.

**Day 7 — Product surface**
- `structured` profile: URL ingestion for Jobstreet/Glints (scraping ~2 sites, not a general crawler).
- Feedback/appeal endpoint (`POST /api/v1/report`) — the §3.6 correction mechanism. Store reports;
  do **not** auto-train on them (data poisoning, §Tahap 3).
- Deploy: Docker + a small VPS or HF Spaces so the demo isn't running off someone's laptop.
- Confirm no analyzed text is persisted — the §3.6 privacy claim must be true in code, not just prose.

**Day 8 — Buffer & delivery**
- Demo rehearsal, slides, screen recording as backup for live-demo failure.
- Update the concept paper with real measured numbers.
- Write the honest limitations section: English-trained model, small eval set, ghost job is not
  implemented (roadmap Tahap 3 — **do not imply otherwise in the demo**).

---

## 5. Out of scope for the MVP — say this out loud when demoing

Per Table 1 of the paper, these are Roadmap Tahap 3 and we will not fake them:
- Ghost job detection (needs posting-age / repost history — impossible from a pasted text).
- Browser extension.
- B2B moderation API and crowdsourced labeling flywheel.

The `structured` profile in Day 7 is the only stretch item, and only if Days 5–6 finish clean.

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| sm_120 / CUDA 12.8 wheel problems | Med | Blocks all training | Gate 0.2 on Day 0, Colab T4 fallback |
| Zero-shot Indonesian transfer is poor | **High** | Headline metric weak | Rule layer is natively Indonesian; translate-train-lite on Day 5; report honestly |
| Meta-model leaks EMSCAD-only metadata | Med | Offline great, live broken | §1.1 assertion enforced in code from Day 1 |
| Indonesian eval set slips | **High** | No headline number | Starts Day 1 hour 1; 200 is a target, 120 well-annotated beats 200 sloppy |
| False positives on real companies | Med | Ethical + reputational | Day 6 FP campaign; thresholds from precision targets |
| Latency > 1s with occlusion | Low | Demo feels sluggish | INT8 + batched occlusion + cap sentence count |
| Scope creep into ghost job / extension | Med | Nothing finishes | §5 is a contract, Day-4 tag is the freeze point |

---

## 7. Definition of done for the MVP

1. Paste an Indonesian job ad → Integrity Score 0–100 + Rendah/Sedang/Tinggi + sentence evidence
   + rule evidence, in under 1 second, in a browser.
2. Every score traces to a calibrated probability and a set of bounded, inspectable rule terms.
3. `eval/results.md` reports EMSCAD **and** Indonesian held-out performance, including the
   unflattering numbers.
4. `pytest tests/` green, including the paper's §3.4 scenario as a regression test.
5. A clean checkout + README gets a teammate to a running demo without asking anyone.
