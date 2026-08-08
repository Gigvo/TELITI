# TELITI — final evaluation

Generated 2026-08-08 17:00 UTC from commit `27a4a5a` by `ml/evaluate.py`.

**Every number quoted in the paper, pitch or demo must appear here** (MVP_PLAN.md Gate 4.2). If a figure is not in this file, it is not a result.

> The EMSCAD `test` split was scored **once**, for this report. It had not been touched by any earlier step — no threshold, no early-stopping choice and no seed selection was made after seeing it.

## Headline

| | EMSCAD test (English) | Indonesian holdout |
|---|---|---|
| PR-AUC | **0.7883** | **0.9258** |
| Sample | 1,788 items, 4.8% scam | 195 items, 36.4% scam |

**These two numbers are not comparable.** PR-AUC scales with prevalence: a random classifier scores ~0.048 on EMSCAD and ~0.364 on the Indonesian holdout. The higher Indonesian figure is not evidence of better performance there.

## 1. EMSCAD test split

| model | PR-AUC | ROC-AUC | precision | recall | F1 | thresh | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf+linearsvc | **0.8651** | 0.9839 | 0.8462 | 0.7674 | 0.8049 | 0.351 | 12 | 20 |
| mdistilbert (calibrated) | **0.7883** | 0.9647 | 0.9273 | 0.5930 | 0.7234 | 0.933 | 4 | 35 |

Confusion matrix at threshold 0.9326 (chosen to maximise F1 on the scam class):

| | predicted real | predicted scam |
|---|---:|---:|
| **actually real** | 1,698 | 4 |
| **actually scam** | 35 | 51 |

Accuracy is 97.82% and is **misleading**: predicting "not a scam" for every item scores 95.19% at this prevalence. It is reported only to say why it is not used.

### Transformer vs TF-IDF baseline

⚠️ The transformer **trails** the TF-IDF baseline by **-0.0768** PR-AUC (0.7883 vs 0.8651).

This is reported rather than omitted. On English EMSCAD a linear model over character and word n-grams is genuinely competitive: scam postings share formulaic surface wording, which is exactly what TF-IDF captures. The transformer earns its place on the multilingual requirement — the baseline cannot transfer to Indonesian at all — not on this number.

Both models were fitted on the same `train` split and scored on the same `test` split. The baseline was refit here rather than reusing its validation figure, so the comparison is on one split.

### Why this is lower than the validation figure

| Split | Transformer | TF-IDF | Role |
|---|---:|---:|---|
| val | 0.8669 | 0.8769 | checkpoint selected here |
| calib | 0.8509 | 0.8948 | calibration fitted here |
| **test** | **0.7883** | **0.8651** | **untouched until now** |

The transformer drops 0.0786 PR-AUC from val to test; the baseline drops 0.0118. The asymmetry is the point, and it is not mysterious: **the transformer checkpoint was chosen by its val score.** Selecting the best of several checkpoints on a split makes that split an optimistic estimate of anything — it is a mild form of fitting to it. The TF-IDF baseline involved no such selection, so it barely moved.

**0.7883 is therefore the honest number** and 0.8669 is not. Quote the former. A paper reporting 0.8669 as generalisation performance would be overstating the result by roughly 0.08 PR-AUC.

Note also the threshold: F1 is maximised at 0.933 on test versus 0.085 on val. The calibrated score distribution shifted, which is a further sign that the val operating point was tuned to that split.

## 2. Calibration

| Metric | Raw softmax | Calibrated |
|---|---:|---:|
| Brier score | 0.0250 | **0.0226** |
| ECE | 0.0234 | **0.0158** |

Lower is better for both. Brier measures accuracy and confidence together; ECE measures the gap between stated confidence and observed frequency.

### Reliability, calibrated

| confidence bin | n | mean predicted | observed | gap |
| --- | ---: | ---: | ---: | ---: |
| 0.0–0.1 | 1699 | 0.0073 | 0.0147 | 0.0074 |
| 0.1–0.2 | 12 | 0.1315 | 0.3333 | 0.2018 |
| 0.2–0.3 | 2 | 0.2509 | 0.0000 | 0.2509 |
| 0.3–0.4 | 2 | 0.3280 | 0.0000 | 0.3280 |
| 0.4–0.5 | 1 | 0.4225 | 1.0000 | 0.5775 |
| 0.5–0.6 | 2 | 0.5545 | 1.0000 | 0.4455 |
| 0.6–0.7 | 5 | 0.6479 | 0.4000 | 0.2479 |
| 0.7–0.8 | 1 | 0.7392 | 0.0000 | 0.7392 |
| 0.8–0.9 | 5 | 0.8572 | 0.2000 | 0.6572 |
| 0.9–1.0 | 59 | 0.9540 | 0.8644 | 0.0896 |

A score is presented to a jobseeker as a number out of 100. If the model says 80 and is right 55% of the time, the number is a lie regardless of how well it ranks. That is what these two rows measure and PR-AUC does not.

> **Domain note.** The figures above use `calibrator.json`, fitted on EMSCAD. The product serves `calibrator_deployment.json`, refitted on the Indonesian holdout, because the EMSCAD calibrator carried a 4.8% base rate into a domain with ~36% scams and pushed every Indonesian ad to 93–100.

## 3. Ablation — why the rule layer is disabled

Measured on the Indonesian holdout (195 items, 36.4% scam), the domain the rules were written for. Source: `eval/indonesian_results.md`.

| Configuration | PR-AUC | False positives |
|---|---:|---:|
| **model only** | **0.9258** | 5 |
| model + rules | 0.8617 | 28 |
| rules only | 0.4167 | 93 |

Adding the rules **lowered** PR-AUC by 0.064 and multiplied false positives by 5.6. Concept paper §3.6 names false positives against real businesses as the expensive error, so the rule layer is disabled for scoring (`RULE_LAYER_ENABLED = False` in `api/scoring.py`) and runs advisory-only: its findings are shown as context and do not move the score.

Rules-only at 0.4167 is barely above the 0.364 prevalence floor — hand-written rules are close to uninformative on their own here.

## 4. Latency

CPU, single request, 30 samples, through the serving path (`api.model` + `api.explain`) rather than a bare forward pass.

| Path | p50 | p95 | p99 | mean |
|---|---:|---:|---:|---:|
| score only | 89 ms | 118 ms | 129 ms | 92 ms |
| score + occlusion | 519 ms | 1026 ms | 1099 ms | 589 ms |

The product always pays the occlusion cost — evidence is not optional in the UI — so the second row is the number that matters. Occlusion runs one extra forward pass per sentence, capped at 12.

## 5. Error analysis

The most confident mistakes at threshold 0.9326, ranked by model confidence rather than distance from the boundary: a wrong answer the model was sure about says something about what it learned.

### False positives — real postings scored as scams (4 shown)

§3.6 calls this the expensive error: a real employer wrongly flagged.

| p(scam) | Excerpt |
|---:|---|
| 0.953 | Accounting Clerk Enter accounts payable transaction Enter accounts receivable transaction Collections Maintain accuracy of records in Accounting software Enter new vendor information and update as necessary Enters Invent… |
| 0.938 | Regional Sales Manager Regional Sales Manager | $25,000 – $28,000 MXN + international training & benefits | Monterrey, Nuevo León Transnational company, leader developer and provider of Construction, Mining and Forestry … |
| 0.937 | Part-Time Administrative/Data Entry I As Part Time Administrative Assistant/ Data Entry I you will be responsible for:- Reporting directly to Account Managers- Professional phone communication; phones, fax, eMail- Respon… |
| 0.933 | Administrative Assistant We are currently looking for a dynamic Administrative Assistant to help support our Human Resources department. The ideal Administrative Assistant provides office services with a great can-do and… |

### False negatives — scams scored as real (10 shown)

A jobseeker sees reassurance where there should be a warning.

| p(scam) | Excerpt |
|---:|---|
| 0.006 | HR PROCESS LEADER Position Summary:EXPRO’s Global Business Services organization seeks a Human Resources Process Leader to provide full time process support to EXPRO’s Accountable Executive at our Corporate Headquarters … |
| 0.006 | Frame Stylist / Optician If you enjoy sales and fashion, consider this position as a full-time Optical Frame Stylist in a private optometry practice in Newcastle, OK. Experience is preferred, however we will consider an … |
| 0.006 | Entry Level Medical Transcriptionist We need a Medical Transcriptionist who will transcribe reports recorded by physicians and other healthcare practitioners. The types of documents include items such as, letters, chart … |
| 0.007 | Assistant Accountant/immediate start Our organisation is seeking students / graduates with a finance, business or commerce related degree qualifications. We want to hear from you if you are seeking experience in genuine,… |
| 0.007 | i Series Team Lead Position: i Series Team Lead Location: Monett, MODuration: 7+ months Contract Technical skills:1. 8 - 10 years of experience in IBM i Series (earlier known as AS400) systemsa. Very Efficient in RPG cod… |
| 0.008 | Assistant Accountant/immediate start Our organisation is seeking students / graduates with a finance, business or commerce related degree qualifications. We want to hear from you if you are seeking experience in genuine,… |
| 0.008 | Project Controller Oil and Gas Approval Center/Houston Position Summary:EXPRO is seeking a Project Controller for the Oil & Gas Approval Center in Houston. The Project Controller will report to the Head of Oil & Gas Busi… |
| 0.008 | Human Resource Director HR responsibilities for assigned client groups in the traditional ares of human resources (ie., employee relations, HRIS, LMS, training, employee development, performance management, etc.) Develop… |
| 0.009 | Fraud Detection Analyst POSITION : Fraud Detection Analyst Customer facing fraud detection role. Key Accountabilities (95%): Â· Thoroughly review accounts/applications by handling inbound calls to determine their validit… |
| 0.009 | Customer Service Specialist We are a specialty retailer offering the very best of what’s next in fashion for men, women and children since 1901.THE REWARDS ARE ENDLESS. With locations in Colorado, California and Arizona,… |

> Read these before writing the discussion section. Patterns here are the difference between "the model has limitations" and knowing which ones.

## 6. Limitations

- **Trained on English.** EMSCAD is English; Indonesian performance rests on 195 manually annotated items. Indicative, not established.
- **EMSCAD is redacted.** Emails, URLs and phone numbers were replaced with placeholders before publication, so the model never learned from real contact details — the signal that matters most in a real scam advertisement.
- **86 positives** in this split. Interval estimates around recall are wide; treat differences of a few points as noise.
- **Ghost jobs are out of scope.** Advertisements for roles that do not exist are a separate problem and are not detected.
- **Not a verdict.** A high score is not proof of fraud and a low score is not a guarantee of safety. The product states this and offers an appeal route.

## Reproducing

```bash
python ml/evaluate.py --split test
```

Model: `Gigvo/teliti-job-scam-mdistilbert` · max_length 256 · commit `27a4a5a`
