# Locales — English now, Indonesian when you have the data

TELITI is fully functional on **English** job ads today, and gains **Indonesian**
capability by dropping two data files into place. No code change, no redeploy logic.

---

## Why this exists

The rule layer was built Indonesian-first. On an English scam ad only **2 of 9 rules
fired**, and the three highest-weighted ones — payment requests, risk phrases, salary
plausibility — were all silent, because their lexicons and wage table were
Indonesian-only.

Meanwhile the Indonesian *evaluation* data may not materialise. The product therefore
has to stand on English alone, without foreclosing Indonesian later.

After the locale layer, the same English scam fires **5 of 9** and scores 21/100.

---

## Current state

```bash
curl localhost:8000/health
```

```json
{
  "locales_available": ["en", "id"],
  "locale_resources": { "en": [], "id": [] }
}
```

`locales_available` lists locales with enough installed resources to contribute.
`locale_resources` names the files each locale is still missing. The API advertises
its **real** capability, not a hoped-for one.

---

## Resource files

| Locale | Lexicon | Wage table |
|---|---|---|
| `en` | `data/reference/risk_phrases_en.yaml` | `data/reference/wages_en.json` (USD/month) |
| `id` | `data/reference/risk_phrases_id.yaml` | `data/reference/umk_2025.json` (IDR/month) |

Both are optional per locale and discovered at start-up.

- **No lexicon** → `payment_request_id` and `risk_phrase_score_id` report
  `available=False`
- **No wage table** → `salary_implausible_vs_umk` reports `available=False`

⚠️ **Missing means UNASSESSED, never clean.** A rule that could not run is not a rule
that found nothing. `unassessed_rules` in the response names them, so a partial
analysis is never mistaken for a complete one. This is the same tri-state used for
redacted corpora — see `api/rules/base.py`.

---

## Which locale gets used

1. An explicit `locale` in the request wins — **if** that locale is usable
2. Otherwise the language is auto-detected from the text
3. If the detected locale has no resources, fall back to `en`

Step 3 is deliberate: scoring Indonesian text with English rules is degraded but
honest, whereas scoring it with an empty rule set is silently useless. Detection is
marker-counting over function words — no dependency, microseconds, and the two
languages share almost no function words.

Every response reports both:

```json
{ "locale": "en", "locale_detected": "id", "unassessed_rules": [] }
```

A mismatch between those two is the signal that a fallback happened.

---

## Adding Indonesian later

1. Put `risk_phrases_id.yaml` and `umk_2025.json` in `data/reference/`
2. Restart the API

`id` appears in `locales_available`, and Indonesian text routes to Indonesian rules
automatically. That is the whole procedure.

Both files are already present in this repo, so `id` is active. To simulate the
English-only situation, move them aside and restart.

---

## Locale-aware vs locale-neutral rules

| Rule | Locale-aware? |
|---|---|
| `email_free_provider` | No — domains are universal |
| `email_absent` | No |
| `email_domain_mismatch` | No |
| `url_shortener` | No |
| `contact_messaging_only` | Partly — link detection universal, phrases per language |
| `qualification_conflict` | **Bilingual patterns in one rule** |
| `salary_implausible_vs_umk` | **Yes** — wage table + number format |
| `payment_request_id` | **Yes** — lexicon |
| `risk_phrase_score_id` | **Yes** — lexicon |

`qualification_conflict` carries both languages in a single rule rather than
splitting by locale, because it is the only feature EMSCAD can teach (see
`eval/derivability_report.md`) and EMSCAD is English. Indonesian-only patterns would
forfeit that.

---

## The separator trap

The number conventions are exactly **inverted**:

| | Indonesian | English |
|---|---|---|
| thousands | `.` — `Rp9.000.000` = 9,000,000 | `,` — `$9,000` = 9,000 |
| decimal | `,` — `4,5jt` = 4,500,000 | `.` — `$5,000.50` |

Reading either with the other convention produces a rule that never fires or fires
constantly. `Locale.dot_is_thousands` selects the convention; both are covered by
tests in `tests/test_locale.py`.

---

## Writing a lexicon for a new language

Copy `risk_phrases_en.yaml` and translate. Keep the group names — the rules key on
`payment` to route it to its own feature.

Weight scale (0–1):

| Range | Meaning |
|---|---|
| 0.9–1.0 | near-conclusive alone |
| 0.6–0.8 | strong, occasionally in legitimate ads |
| 0.3–0.5 | suggestive only alongside others |
| 0.1–0.2 | weak, common in ordinary ads |

Be conservative. Every weight is a chance to flag a real employer, and §3.6 of the
concept paper makes suppressing false positives an explicit requirement. Aggregation
saturates (`1 - prod(1 - w)`), so weak phrases cannot pile up into certainty — but
that is a backstop, not a licence.

**Do not encode corpus artefacts.** The TF-IDF baseline weighted `subsea` heavily
(28 EMSCAD documents, 96.4% fraudulent), but that is one 2017 oil-and-gas scam
campaign, not a property of job scams. It is deliberately absent from
`risk_phrases_en.yaml`.
