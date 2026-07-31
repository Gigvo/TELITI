# Rule-weight validation

> ⚠️ **SYNTHETIC — NOT A RESULT.** These numbers come from fabricated fixtures written to exercise the pipeline. They measure the assumptions of whoever wrote the fixtures, not the behaviour of real job ads. Do not cite, screenshot, or put them in a slide.

Weights: `handset-1.0.0` (set a priori, see ml/rule_weights.py)

Items: 24 (12 scam, 12 legitimate)

## Does the rule layer help?

| metric | text only | + rules | delta |
| --- | ---: | ---: | ---: |
| pr_auc | nan | 0.9724 | +nan |
| roc_auc | nan | 0.9757 | +nan |
| precision | nan | 0.5000 | +nan |
| recall | nan | 1.0000 | +nan |
| f1 | nan | 0.6667 | +nan |

## Separation

- mean probability shift on scam ads: **0.0337**
- mean probability shift on legitimate ads: **0.0018**
- mean rules fired, scam: 1.42
- mean rules fired, legitimate: 0.08
- legitimate ads triggering **any** rule: 1 / 12

The shift on scam ads must exceed the shift on legitimate ones. If it does
not, the rule layer is adding noise and should be reported as such.

The false-positive count is the number section 3.6 cares about most.

## Rule fire counts

| rule | weight | times fired |
| --- | ---: | ---: |
| email_absent | 0.01 | 7 |
| contact_messaging_only | 0.08 | 5 |
| email_free_provider | 0.04 | 4 |
| email_domain_mismatch | 0.04 | 1 |
| url_shortener | 0.03 | 1 |
| payment_request_id | 0.10 | 0 |
| salary_implausible_vs_umk | 0.07 | 0 |
| risk_phrase_score_id | 0.06 | 0 |
| qualification_conflict | 0.02 | 0 |

A rule that never fires contributes nothing and should be cut or fixed.
A rule that fires on almost everything is not discriminating.
