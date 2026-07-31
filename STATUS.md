# TELITI — build status

**Updated:** 2026-07-31 · **Tests:** 278 passing · **Tag:** none yet (freeze at Day 4)

> **Locale layer added.** The product now works fully on **English** ads
> ([docs/LOCALES.md](docs/LOCALES.md)). Indonesian is a drop-in: put the two
> reference files in `data/reference/` and it activates on restart. This de-risks
> the possibility that Indonesian data never materialises — the English path is no
> longer a fallback, it is a complete product.

Deadline is 4–8 days. Days 1–4 are the critical path to a demo; everything after is
additive. See [MVP_PLAN.md](MVP_PLAN.md) for the full plan and gates.

---

## Progress at a glance

| Phase | Status |
|---|---|
| Day 0 — environment | ▓▓▓▓▓▓▓░░░ mostly done (GPU box unverified) |
| Day 1 — contract, data, baseline, rules | ▓▓▓▓▓▓▓▓░░ done except annotation + frontend |
| Day 2 — transformer, calibration | ▓▓▓░░░░░░░ rules done, model blocked |
| Day 3 — fusion, XAI, ONNX | ▓▓▓░░░░░░░ scoring done, rest blocked |
| Day 4 — integration, evaluation | ░░░░░░░░░░ not started |

---

## ✅ Done

### Day 0 — environment
- [x] **0.1** CPU/serving deps installed (`.venv`)
- [x] **0.3** EMSCAD downloaded, **Gate 0.3 PASS** — 17,880 × 18, 866 fraud (4.84%)
- [x] **0.4** Repo skeleton, git initialised

### Day 1
- [x] **1.1** API contract frozen — `api/schemas.py`, stub served, **Gate 1.1 PASS**
- [x] **1.2** Data prep — `ml/prepare_data.py`, 4-way stratified split, **Gate 1.2 PASS**
  - train 12,513 / val 1,788 / calib 1,788 / test 1,788; `pos_weight = 19.69`
  - Reproducible: seed + source SHA-256 + per-split checksums in `split_manifest.json`
- [x] **1.2b** Derivability measured — `eval/derivability_report.md`
  - EMSCAD cannot teach 8 of 9 rule weights. Contact details stripped before publication.
- [x] **1.3** TF-IDF baselines — **Gate 1.3 PASS**, best PR-AUC **0.8769** (`tfidf+linearsvc`)
- [x] **1.4** Rule engine + email/contact rules (5 features)

### Day 2
- [x] **2.4** Remaining rules (4 features) — **Gate 2.4 PASS**, `pending_features == ()`
  - `qualification_conflict` (bilingual), `salary_implausible_vs_umk` (UMK),
    `payment_request_id`, `risk_phrase_score_id`
  - Reference data: `data/reference/umk_2025.json`, `risk_phrases_id.yaml`

### Ahead of schedule
- [x] **3.1 (partial)** Scoring layer — `api/scoring.py`, hand-set weights, bounded
- [x] Eval-set framework — schema, validator, Cohen's κ, synthetic guard
- [x] **Locale layer** — `api/locale.py`, English lexicon + wage table
  - English scam: 2/9 rules fired → **5/9**, scores 21/100 Tinggi
  - English legitimate ad: 0/9 rules fire (false-positive guard holds)
  - Missing resources report **unassessed**, never clean

---

## ⛔ Blocked — waiting on your team

- [ ] **0.2** GPU verification on the RTX 5050 box (cu128 wheels + a real matmul)
      **Blocks all of step 2.1.** Takes ~15 minutes.
- [ ] **1.5** Holdout evaluation set, ~200 items
      Blocks the headline metric, 2.2, and 3.1 validation.
      Run `python ml/validate_eval_set.py` while annotating.

      **No longer Indonesian-only.** With the locale layer, an **English** holdout
      is now a complete deliverable — and English scam ads are far easier to source
      (FTC/Action Fraud advisories, r/scams, BBB reports, LinkedIn scam warnings)
      than Indonesian ones. Annotate whichever your team can actually get. Both
      files use the same schema; `eval/schema.json` is unchanged.

---

## ⬜ Not started

### Day 1
- [ ] **1.6** React + Vite frontend shell *(can start now — API stub is live)*

### Day 2
- [ ] **2.1** Fine-tune mDistilBERT — *needs 0.2* — Gate: val PR-AUC ≥ 0.88
- [ ] **2.2** Zero-shot Indonesian probe — *needs 1.5 + 2.1*
- [ ] **2.3** Platt calibration — *needs 2.1* — Gate: ECE ≤ 0.05
- [ ] **2.5** Wire the real model into the API — *needs 2.1*

### Day 3
- [ ] **3.1** Validate rule weights on holdout — *needs 1.5* — harness ready
- [ ] **3.2** Thresholds from a precision target — *needs 2.1*
- [ ] **3.3** ONNX export + INT8 — *needs 2.1* — Gate: p95 < 400ms, F1 drop < 1pp
- [ ] **3.4** XAI sentence occlusion — *needs 2.1* — Gate: < 1s total
- [ ] **3.5** Frontend integration — *needs 1.6*

### Day 4 — 🎯 shippable
- [ ] **4.1** Full pipeline wired
- [ ] **4.2** Final evaluation → `eval/results.md` (touch `test` once)
- [ ] **4.3** Demo hardening
- [ ] **4.4** Tag `v0.1-mvp` — **freeze point**

### Days 5–8 — additive
- [ ] Translate-train if 2.2 shows recall < 0.75
- [ ] LIME/SHAP validation of occlusion
- [ ] False-positive campaign on 100 real legitimate ads
- [ ] Deployment, feedback endpoint, demo rehearsal

---

## 📝 Paper changes required

Measurement contradicted three assumptions. Each is a stronger methods section than
the original, because "we tested it and it failed, so here is what we did" beats an
untested claim.

| § | Change | Why |
|---|---|---|
| **3.3** | Fusion is **not** fitted on EMSCAD | 0.02% of docs have a real email, 0.39% a URL. `email_absent` availability differs by class (74.8% real vs 67.9% fraud) — a model using it would learn the anonymisation, not the fraud. |
| **3.3** | Cap is **per-rule 0.10**, not aggregate 0.15 | 0.15 was sized for 3 rules. With 9 it is ~2 points each. Per-rule is what *"tidak ada satu aturan deterministik pun yang dapat mendominasi"* actually asserts. |
| **3.3** | Weights **set a priori and validated**, not fitted | Fitting needs a second ~200-item Indonesian set — unaffordable. Report whatever validation shows, including a null result. |
| **3.2** | SMOTE tested, **not adopted** | Measured: `tfidf+smote+logreg` 0.8638 vs `class_weight='balanced'` 0.8662. A decision, not an assumption. |
| **3.2** | Note EMSCAD text quality | HTML stripped without separators fused words in 78% of documents; repaired before training. |
| **2.3 / 3.3** | *(optional)* LIME is offline-only | LIME needs 1,000–5,000 forward passes; production uses sentence occlusion, validated against LIME on Day 6. |

---

## ⚠️ Open risks

1. **Annotation is the critical path.** Nothing downstream of 1.5 can complete.
2. **Expect weak zero-shot Indonesian transfer.** The baseline partly memorised 2017
   English campaigns — `subsea` appears in 28 train docs, 96.4% fraudulent; `below
   link` 96.3%. That vocabulary cannot help on Indonesian WhatsApp ads. §3.2 already
   commits to reporting this openly.
3. **GPU unverified.** A plain `pip install torch` gives sm_120-less wheels that
   report `cuda.is_available() == True` and then die.
4. **UMK figures need confirming** against the official Kemnaker release before they
   appear in the paper (`data/reference/umk_2025.json`).

---

## Suggested next moves

**Your team:** start annotation today (`ml/validate_eval_set.py`), and run Gate 0.2
on the GPU box.

**Me, unblocked right now:** step **1.6** (React frontend against the live stub) —
the only remaining Day-1 item needing no Indonesian data and no GPU.
