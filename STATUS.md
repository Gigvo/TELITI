# TELITI — build status

**Updated:** 2026-08-09 · **Tests:** 421 backend + 37 frontend = **458 passing**
**State:** MVP complete. Final evaluation done; `test` split spent.

See [MVP_PLAN.md](MVP_PLAN.md) for the plan and gates, [eval/results.md](eval/results.md)
for every measured number, and [docs/PAPER_GUIDE.md](docs/PAPER_GUIDE.md) for what
still has to change in the paper.

---

## Gates

| Day | Step | State |
|---|---|---|
| 0 | Environment, GPU verification, data load | done |
| 1 | API contract, splits, TF-IDF baseline, rule engine, UI | done |
| 2 | Transformer fine-tune, calibration, rules complete | done — **Gate 2.1 FAILED**, see below |
| 3 | Fusion, thresholds, XAI, ONNX | done except 3.3 (ONNX, deliberately skipped) |
| 4.1 | Wire model into the API | done |
| 4.2 | Final evaluation → `eval/results.md` | done |
| 4.3 | Demo hardening | done |
| 4.4 | Tag `v0.1-mvp` | **outstanding** |

### Gate 2.1 did not pass

Required val PR-AUC ≥ 0.88 against a TF-IDF baseline of 0.8769. The transformer
reached **0.8669**, and on the untouched `test` split **0.7883** against the
baseline's 0.8651.

The transformer ships anyway, for a stated reason rather than by default: TF-IDF
trained on English cannot process Indonesian at all, and the Indonesian holdout
is the actual deployment domain. This is written up in
[docs/PAPER_GUIDE.md](docs/PAPER_GUIDE.md) §B2.1 rather than left as a silent
override.

### Step 3.3 skipped

ONNX INT8 export. The latency budget was met without it (p50 469 ms including
explanation generation), so the optimisation had no problem left to solve. The
paper must not claim ONNX serving — see PAPER_GUIDE §A6.

---

## Headline numbers

| | Value |
|---|---|
| EMSCAD test PR-AUC (transformer) | **0.7883** |
| EMSCAD test PR-AUC (TF-IDF baseline) | 0.8651 |
| Indonesian holdout PR-AUC | **0.9258** (195 items, 36.4% scam) |
| ECE after calibration | 0.0158 |
| Latency p50 / p95 with explanation | 469 ms / 949 ms |

Full detail, including error analysis and the val→test drop, in
[eval/results.md](eval/results.md). **Do not quote a number that is not in that
file** (MVP_PLAN.md Gate 4.2).

---

## Deliberate decisions worth knowing

- **The rule layer is disabled for scoring.** `RULE_LAYER_ENABLED = False` in
  `api/scoring.py`. It lowered PR-AUC from 0.9258 to 0.8617 and raised false
  positives from 5 to 28. It still runs and its findings are shown as context.
- **Two calibrators ship.** `calibrator_deployment.json` (Indonesian) is used;
  `calibrator.json` (EMSCAD) pushed every Indonesian ad to 93–100.
- **The `test` split was scored once**, on 2026-08-09. `eval/.test_split_consumed`
  records it and `ml/evaluate.py` refuses to repeat without `--force`.
- **Weights are not in the repo.** They download from Hugging Face on first run.

---

## Outstanding

| Item | Owner |
|---|---|
| Tag `v0.1-mvp` (step 4.4) | whoever commits next |
| Apply PAPER_GUIDE changes to the paper | paper author |
| Record the 3–5 minute demo video | team |
| Make the GitHub repo public after the deadline | repo owner |
