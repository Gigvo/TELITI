# Indonesian real/fake job posting dataset — what to prepare

This is the **holdout evaluation set** referenced throughout the project as step
1.5. It is the single most important dataset in the entire MVP — more important
than the model itself, because it's the only thing that tells you whether the model
actually works on real Indonesian job ads instead of just on English EMSCAD data
from 2017.

Target file: **`eval/indonesian_holdout.jsonl`** (does not exist yet — you're
creating it from scratch).

---

## Why this dataset exists (read this first)

The transformer is being trained on **EMSCAD**, a public dataset of ~18,000
English-language job postings from 2017, scraped mostly from formal Western job
boards. That's the only large labeled dataset available — nobody has published a
labeled Indonesian scam/legitimate job posting dataset.

The problem: a model trained on 2017 English job-board postings might not
generalise to 2026 Indonesian WhatsApp/Telegram job scams at all. Different
language, different era, different distribution channel, different scam tactics.

So this holdout set is the **reality check**. After training, the model runs
*zero-shot* against these Indonesian items — never trained on them — and however it
scores is the honest number that goes in the report. If it does badly, that's not a
failure to hide; it's exactly the finding `MVP_PLAN.md` §1.2b already anticipated
and budgeted a mitigation for (translate-train, if needed).

This also directly answers whether the rule layer (§3.1's deterministic rules —
salary vs. UMK, payment-request phrases, Telegram-only contact, etc.) is actually
catching real Indonesian scam patterns, since those rules were *designed* around
Indonesian-specific signals but never tested against real Indonesian ads.

---

## The target

| | |
|---|---|
| **File** | `eval/indonesian_holdout.jsonl` |
| **Format** | JSONL — one JSON object per line, no wrapping array |
| **Target size** | ~200 items |
| **Minimum useful size** | ~50 items (enables a first Cohen's κ check) |
| **Composition** | Roughly **35% scam, 65% legitimate** — see "Why skewed toward legitimate" below |
| **Language** | Indonesian (Bahasa Indonesia, informal register is fine and expected) |

---

## Every field, explained

Each line is one JSON object. Here's a real example of what a complete, valid entry
looks like:

```json
{"id": "id-holdout-0001", "text": "LOWONGAN KERJA ADMIN ONLINE\nDibutuhkan segera admin online untuk perusahaan ternama.\nGaji Rp9.000.000 per bulan, tanpa pengalaman, langsung kerja dari rumah.\nKuota terbatas hanya untuk 10 orang pertama!\nWajib membayar biaya administrasi sebesar Rp250.000 untuk proses berkas.\nInterview dilakukan via Telegram.\nKirim CV dan foto KTP ke hrd.rekrutmen2024@gmail.com", "label": 1, "source_url": "https://www.instagram.com/p/example123/", "source_type": "watchdog_account", "channel": "whatsapp", "annotator_a": "ivan", "annotator_b": "revo", "label_a": 1, "label_b": 1, "resolved_by": null, "collected_at": "2026-08-04", "campaign": "admin-online-telegram-2026", "notes": "Classic pattern: upfront fee + Telegram interview + ID photo request"}
```

### Required fields

| Field | Type | What it means |
|---|---|---|
| `id` | string | Must match `id-holdout-0001`, `id-holdout-0002`, etc. Sequential, zero-padded to 4 digits. **Don't hand-write these** — `ml/add_eval_item.py` allocates them automatically so you never get a collision. |
| `text` | string | The job ad **verbatim**. Minimum 30 characters. See "The verbatim rule" below — this is the field most likely to be done wrong. |
| `label` | integer | `1` = scam or ghost job. `0` = legitimate. This is the *final* label after any disagreement is resolved — see the labeling protocol below. |
| `source_url` | string | Where you found it. A link, not a description. This is what turns an entry from "someone's opinion" into "evidence." Minimum 4 characters, but in practice always a real URL. |
| `source_type` | string | One of a fixed set — see table below. |
| `channel` | string | Where the ad circulated — see table below. |
| `annotator_a` | string | Name/initial of the first person who labeled it independently. |
| `annotator_b` | string | Name/initial of the second person who labeled it independently. |
| `collected_at` | string | Date you added it, format `YYYY-MM-DD`. |

### Fields required for everything except synthetic fixtures

| Field | Type | What it means |
|---|---|---|
| `label_a` | integer (0/1) | Annotator A's label, given **before** seeing annotator B's label or discussing it. |
| `label_b` | integer (0/1) | Annotator B's label, same independence requirement. |

These two exist so the tool can compute **Cohen's κ** (inter-annotator agreement) —
a statistic that tells you whether "scam" means the same thing to both annotators.
If you skip independent labeling and just agree on a label together, κ becomes
meaningless and the validator will complain.

### Optional fields

| Field | Type | What it means |
|---|---|---|
| `resolved_by` | string or null | Only set when `label_a != label_b`. Name of the third person (or one of the two, after discussion) who made the tie-breaking call. If `label_a == label_b`, leave this `null` — there was nothing to resolve. |
| `campaign` | string or null | A grouping key when multiple ads clearly belong to the same scam operation or the same real company. Explained below — this matters more than it looks like it should. |
| `notes` | string | Free text — why you labeled it the way you did, anything unusual about it. Not validated, purely for your own future reference. |
| `synthetic` | boolean | Never set this to `true` for real data. It exists only for the fixture file (`eval/synthetic_fixture.jsonl`) that developers use to test the pipeline without polluting real metrics. If you ever see `SYNTHETIC-0001`-style IDs, that's fixture data, not this dataset. |

### `source_type` — pick one

| Value | Use for |
|---|---|
| `bareskrim` | Official Indonesian police (Bareskrim Polri) fraud advisories/reports |
| `kominfo` | Ministry of Communication and Informatics advisories |
| `media` | News articles (Kompas, Detik, CNN Indonesia, Tirto, etc.) reporting on a specific scam |
| `watchdog_account` | Instagram/Twitter/TikTok accounts that specifically track and repost job scams |
| `community_report` | Reddit (r/indonesia), Kaskus, Facebook groups, or similar — someone posting "is this legit?" |
| `jobstreet` | A legitimate posting pulled directly from Jobstreet |
| `glints` | A legitimate posting pulled directly from Glints |
| `kalibrr` | A legitimate posting pulled directly from Kalibrr |
| `karir_com` | A legitimate posting pulled directly from Karir.com |
| `campus_career` | A university career-center posting |
| `company_official` | Pulled from a verified company's own careers page or official social media |

### `channel` — pick one

| Value | Use for |
|---|---|
| `whatsapp` | Ad or forward was on WhatsApp (screenshot, forwarded message, WA group) |
| `telegram` | Circulated on Telegram |
| `instagram` | Posted as an Instagram post/story/DM |
| `facebook` | Posted on Facebook (page, group, marketplace) |
| `job_board` | A formal job board (Jobstreet, Glints, LinkedIn, company career page) |
| `other` | Doesn't fit the above — note what it actually was in `notes` |

---

## The verbatim rule — the easiest thing to get wrong

**Copy the ad exactly as it appears. Do not clean it up.**

Keep:
- Original line breaks
- ALL CAPS if that's how it was written
- Emoji (🔥💰✅ etc. — scams use these constantly and it's a real signal)
- Typos and grammar errors
- Excessive punctuation ("!!!", "???")

Why this matters: the model and the rule layer both operate on messy, real-world
text — a user is going to paste an actual screenshot transcript, not a cleaned-up
version. If your training/eval text is all tidy and grammatically corrected, you're
testing the system against text that doesn't resemble what it will actually see in
production. A scam ad's chaotic formatting is often itself part of the signal.

**Do** fix genuine copy-paste artifacts (e.g., if a screenshot OCR tool inserted a
stray character that clearly isn't part of the original) — but when in doubt, leave
it as-is.

---

## The labeling protocol — why two people, and in what order

1. **Two people independently label every item as scam (1) or legitimate (0),
   without discussing it first, and without seeing each other's answer.** This is
   what `label_a` and `label_b` capture.
2. If they agree, that's the final `label`. `resolved_by` stays `null`.
3. If they disagree, a **third person** looks at it and decides — that becomes the
   final `label`, and `resolved_by` records who made the call.
4. **Never label as a group discussion first, then write down the "agreed" label as
   if it were independent.** That defeats the entire point of `label_a`/`label_b` —
   the system computes Cohen's κ from these two fields specifically to check
   whether your labeling criteria are consistent enough to trust. If both fields
   are secretly the same post-discussion answer, κ will report perfect agreement
   even if your actual criteria are shaky, and that's a false signal.

`ml/add_eval_item.py` enforces this ordering in its prompts — it asks for
annotator A's label, then annotator B's label, before ever asking about
resolution — precisely so this protocol isn't skippable by accident.

---

## Why the mix should skew toward legitimate (~35% scam / 65% legit)

It's tempting to load this file with scams because they're the interesting case.
Don't. Two reasons:

1. **False positives are the expensive error, not false negatives.** A legitimate
   company getting flagged as a scam is reputationally damaging and undermines
   trust in the whole tool. A dataset that's 80% scams doesn't stress-test that
   failure mode — you need plenty of *real, ordinary, boring* Indonesian job
   postings to confirm the system doesn't cry wolf on them.
2. **A skewed eval set produces a misleading precision number.** If you evaluate
   against 90% scams, a model that just says "scam" to everything scores great on
   recall and the number looks fantastic while being useless.

A rough target: for every 2 scam ads, aim for about 3-4 legitimate ones.

---

## The `campaign` field — why it's worth the extra effort

If you find 5 different screenshots that are all clearly the same scam operation
(same phrasing template, same fake company name, same contact pattern, just
reposted by different people), tag them all with the same `campaign` value, e.g.
`"campaign": "admin-online-telegram-2026"`.

Why: if 5 near-duplicate items from one campaign end up in the dataset, the
evaluation metrics get inflated — the model "correctly" catches the same pattern 5
times, which looks like broad generalization but is actually just memorizing one
template. The `campaign` field lets the tooling detect and account for this. Same
logic applies to legitimate ads — if you pull 5 postings from one company that are
templated, tag them.

If you're not sure, leave it `null`. It's a nice-to-have, not a blocker.

---

## Where to actually find items

### Scam sources (aim for ~70 items)

- **Bareskrim Polri** press releases and official warnings about job scam networks
- **Kominfo** fraud advisories
- News articles from Kompas, Detik, CNN Indonesia, Tirto, Kumparan — search terms
  like `"lowongan kerja palsu"`, `"modus penipuan loker"`, `"korban penipuan kerja"`
- Instagram/TikTok watchdog accounts that repost scam screenshots (search
  `"loker palsu"`, `"awas penipuan kerja"`)
- Reddit `r/indonesia` — search `"loker"` + `"scam"` or `"penipuan"`
- Kaskus forums, Facebook groups where people post "is this job legit?" with
  screenshots attached
- Your own family/friend network — if anyone has ever received a suspicious job
  offer via WhatsApp, that's a genuinely valuable real-world sample (get their
  permission and strip any personal info that isn't part of the ad itself)

### Legitimate sources (aim for ~130 items)

- **Jobstreet, Glints, Kalibrr, Karir.com** — just pull real postings directly
- University career center pages (many Indonesian universities publish these
  publicly — UI, ITB, UGM, etc.)
- Verified company Instagram/LinkedIn accounts posting "kami sedang membuka
  lowongan..."
- Company official career pages

**Try to get a spread of industries, company sizes, and formality levels** —
a mix of big-corporate polished postings and small-business informal ones. Real
users will encounter both.

---

## How to actually add items — use the tool, don't hand-write JSON

Don't write JSONL lines by hand. Use:

```bash
python ml/add_eval_item.py
```

It walks you through every field interactively:
1. Paste the ad text (end with a line containing just `.`)
2. Prompts for annotator A's name, then A's label (0 or 1)
3. Prompts for annotator B's name, then B's label
4. If they disagree, asks who resolved it and what the final label is
5. Asks for the source URL, source type (numbered menu), channel (numbered menu)
6. Optional campaign key and notes
7. Auto-allocates the next `id-holdout-XXXX`, appends the line, and immediately
   re-validates the whole file so a mistake is caught right away, not on day 5

To check overall progress at any time (item count, scam/legit balance, Cohen's κ so
far, any validation problems):

```bash
python ml/validate_eval_set.py
```

---

## What "done" looks like

- [ ] ~200 items total (50 is a usable floor, 200 is the target)
- [ ] Roughly 35% scam / 65% legitimate
- [ ] Every item has independent `label_a` / `label_b` from two different people
- [ ] Disagreements have a `resolved_by` and a final `label`
- [ ] `python ml/validate_eval_set.py` reports no schema errors
- [ ] Cohen's κ is reasonably high (above ~0.6 is a good sign your two annotators
      agree on what "scam" means; much lower means you should talk through your
      criteria before continuing)
- [ ] A decent spread of `source_type` and `channel` values — not everything from
      one source

Once this file exists with real content, `eval/indonesian_holdout.jsonl` becomes the
input to:
- **Step 2.2** — the zero-shot cross-language probe (the headline "does this
  actually work on Indonesian text" number)
- **Step 3.1** — validating whether the hand-set rule weights help or hurt on real
  Indonesian ads
- **Step 4.2** — the final evaluation report

None of those steps can run meaningfully without this file.
