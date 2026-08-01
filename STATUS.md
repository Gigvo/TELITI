# TELITI — build status

**Updated:** 2026-08-01 · **Tests:** 297 backend + 19 frontend = **316 passing**
**Tag:** none yet (freeze at Day 4)

Deadline is 4–8 days. Days 1–4 are the critical path to a demo; everything after is
additive. See [MVP_PLAN.md](MVP_PLAN.md) for the full plan and gates.

> **Locale layer in place.** The product works fully on **English** ads
> ([docs/LOCALES.md](docs/LOCALES.md)). Indonesian is a drop-in: put the two
> reference files in `data/reference/` and it activates on restart. The English
> path is not a fallback — it is a complete product.

---

## Progress at a glance

| Phase | Status |
|---|---|
| Day 0 — environment | ▓▓▓▓▓▓▓░░░ done except GPU box |
| Day 1 — contract, data, baseline, rules, UI | ▓▓▓▓▓▓▓▓▓░ done except annotation |
| Day 2 — transformer, calibration | ▓▓▓░░░░░░░ rules done, model blocked on GPU |
| Day 3 — fusion, XAI, ONNX | ▓▓▓▓░░░░░░ scoring + UI done, rest blocked |
| Day 4 — integration, evaluation | ░░░░░░░░░░ not started |

---

## ✅ Done

### Day 0 — environment
- [x] **0.1** CPU/serving deps installed (`.venv`)
- [x] **0.3** EMSCAD downloaded, **Gate 0.3 PASS** — 17,880 × 18, 866 fraud (4.84%)
- [x] **0.4** Repo skeleton, git initialised

### Day 1
- [x] **1.1** API contract frozen — `api/schemas.py`, **Gate 1.1 PASS**
- [x] **1.2** Data prep — 4-way stratified split, **Gate 1.2 PASS**
  - train 12,513 / val 1,788 / calib 1,788 / test 1,788; `pos_weight = 19.69`
  - Reproducible: seed + source SHA-256 + per-split checksums
- [x] **1.2b** Derivability measured — `eval/derivability_report.md`
- [x] **1.3** TF-IDF baselines — **Gate 1.3 PASS**, best PR-AUC **0.8769**
- [x] **1.4** Rule engine + email/contact rules (5 features)
- [x] **1.6** React + Vite frontend — **Gate 1.6 PASS**
  - Score gauge, rule cards, in-place evidence highlighting, extracted fields
  - 6 one-click demo examples (scam / legitimate / borderline × en / id)
  - Stub warning, API-unreachable banner, unassessed-rules disclosure
  - 19 tests; typecheck, lint and production build all clean

### Day 2
- [x] **2.4** Remaining rules (4 features) — **Gate 2.4 PASS**, `pending_features == ()`

### Ahead of schedule
- [x] **3.1 (partial)** Scoring layer — hand-set weights, bounded per rule
- [x] **3.5 (partial)** Frontend renders real rule evidence already
- [x] Eval-set framework — schema, validator, Cohen's κ, synthetic guard
- [x] **Locale layer** — English lexicon + wage table; English scam 2/9 → **5/9** rules

---

## ⛔ Blocked — waiting on your team

- [ ] **0.2** GPU verification on the RTX 5050 box (cu128 wheels + a real matmul)
      **Blocks all of step 2.1.** ~15 minutes.
- [ ] **1.5** Holdout evaluation set, ~200 items
      Blocks the headline metric, 2.2, and 3.1 validation.
      Run `python ml/validate_eval_set.py` while annotating.

      **English is now acceptable** and far easier to source (FTC / Action Fraud
      advisories, r/scams, BBB reports). Same schema either way.

---

## ⬜ Not started

### Day 2
- [ ] **2.1** Fine-tune mDistilBERT — *needs 0.2* — Gate: val PR-AUC ≥ 0.88
- [ ] **2.2** Zero-shot cross-language probe — *needs 1.5 + 2.1*
- [ ] **2.3** Platt calibration — *needs 2.1* — Gate: ECE ≤ 0.05
- [ ] **2.5** Wire the real model into the API — *needs 2.1*

### Day 3
- [ ] **3.1** Validate rule weights on holdout — *needs 1.5* — harness ready
- [ ] **3.2** Thresholds from a precision target — *needs 2.1*
- [ ] **3.3** ONNX export + INT8 — *needs 2.1* — Gate: p95 < 400 ms
- [ ] **3.4** XAI sentence occlusion — *needs 2.1* — Gate: < 1 s total
- [x] **3.5** Frontend integration — done early, will need re-checking after 2.5

### Day 4 — 🎯 shippable
- [ ] **4.1** Full pipeline wired — *needs 2.5*
- [ ] **4.2** Final evaluation → `eval/results.md` — *needs 2.1 + 1.5*
- [x] **4.3** Demo hardening — done early
  - Input sanitisation: control chars, bidi overrides, zero-width evasion
  - Meaningful-content floor (whitespace/emoji/punctuation-only now 422)
  - Latency **794 ms → 86 ms** worst case at 20k chars
  - `analysed_text` returned so the client renders exactly what was scored
- [ ] **4.4** Tag `v0.1-mvp` — **freeze point**

### Days 5–8 — additive
- [ ] Translate-train if 2.2 shows weak transfer
- [ ] LIME/SHAP validation of occlusion
- [ ] False-positive campaign on 100 real legitimate ads
- [ ] Deployment, feedback endpoint, demo rehearsal

---

## 📝 Paper changes required

| § | Change | Why |
|---|---|---|
| **3.3** | Fusion **not** fitted on EMSCAD | 0.02% of docs have a real email, 0.39% a URL. `email_absent` availability differs by class (74.8% vs 67.9%) — a model using it learns the anonymisation, not the fraud. |
| **3.3** | Cap is **per-rule 0.10**, not aggregate 0.15 | 0.15 was sized for 3 rules; with 9 it is ~2 points each. |
| **3.3** | Weights **set a priori and validated**, not fitted | Fitting needs a second ~200-item set — unaffordable. |
| **3.2** | SMOTE tested, **not adopted** | Measured: 0.8638 with SMOTE vs 0.8662 with class weighting. |
| **3.2** | Note EMSCAD text quality | HTML stripped without separators fused words in 78% of documents. |
| **1.1 / 3.2** | *(pending)* Language scope | Decide once the dataset is known. If English-only, §1.1's Indonesian framing becomes stated motivation with an explicit limitation. |

---

## ⚠️ Open risks

1. **Annotation is the critical path.** Nothing downstream of 1.5 can complete.
2. **Expect weak cross-language transfer.** The baseline partly memorised 2017
   English campaigns — `subsea` appears in 28 train docs, 96.4% fraudulent.
3. **GPU unverified.** A plain `pip install torch` gives sm_120-less wheels that
   report `cuda.is_available() == True` and then die.
4. **Wage figures need confirming** against official sources before publication
   (`umk_2025.json`, `wages_en.json`).

---

## Deployment

`docker compose up --build` → <http://localhost:8000>. One container, one port:
the React bundle is built in and served by FastAPI, so there is no CORS and no
reverse proxy to configure on an unfamiliar demo network.

⚠️ **Not yet built or run** — the Docker daemon was unavailable on this machine.
The Dockerfile is statically validated (every COPY source exists,
`package-lock.json` present for `npm ci`, healthcheck syntax checked) but the build
itself is unverified. **Run it once before demo day.**

Model artifacts and locale resources are mounted, not baked in, so a retrained
model is a restart rather than a rebuild.

---

## Suggested next moves

**Your team:**
1. Run Gate 0.2 on the GPU box (~15 min) — unblocks the entire 2.x/3.x path
2. Start annotation: `python ml/add_eval_item.py` prompts for each field,
   allocates ids, and re-validates after every entry
3. Build the Docker image once to confirm it works

**Me:** genuinely out of unblocked work. Everything remaining needs the GPU box or
the holdout set. What is left is small: a `.env` for configuration, or expanding
the English lexicon — neither is on the critical path.
