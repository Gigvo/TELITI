# The Indonesian dataset: what's wrong and what "correct" looks like

**File:** `eval/indonesian_holdout.jsonl`
**Status:** 200 items collected · 114 are fine · ~86 need fixing

This document explains the problem in plain terms, states the requirement each item
must meet, and gives a step-by-step fix.

---

## Part 1 — The problem

### The one-sentence version

Most of the "legitimate" examples aren't job advertisements. They're **one-line
summaries from a search results page**, while every scam example is a **full
WhatsApp message**. So the two groups differ in an obvious way that has nothing to do
with fraud.

### What the bad items look like

Here are real entries from the file:

```
Driver • Cititrans • Kompetitif • SMA/SMK • 1–2 thn • Luar Jabodetabek
Staff Gudang • Roschel • Kompetitif • SMA/SMK • 0–2 thn • Jakarta Barat
Kasir • Andifa Vet Clinic • Kompetitif • SMA/SMK • 0–2 thn • Jakarta Selatan
Helper Mekanik • PT. Anugrah Vindo Abadi • Kompetitif • SMK • 1–2 thn • Bekasi
```

That's the row you see on a job board's *search results* page — title, company,
salary band, education, experience, location. It is not the advertisement. The
advertisement is on the page you reach after clicking it.

### What a scam item looks like

```
PANGGILAN INTERVIEW KERJA DARI
Bagian Personalia
IBU. AMEL
Kepada Yth. Sdr/i Di Tempat
JADWAL WAWANCARA. Diharapkan hadir,
Pada Hari : Kamis 26 MARET 2020
Waktu : Bisa datang antara Pukul 07:30 s/d 12:00 siang
ALAMAT KANTOR : Jl. Raden Inten II 100-150, Duren Sawit, Jakarta Timur 13440
POSISI YANG KOSONG
- Staff Admin  - Staff Admin Gudang  - Packing Barang  - Receptionis
...
```

918 characters. A complete forwarded message — exactly what a user would paste.

### The numbers

| | Median length | Shortest | Longest |
|---|---:|---:|---:|
| Scam items | **767 chars** | 185 | 2,478 |
| Legitimate items | **125 chars** | 70 | 1,289 |

**86 of the 200 items are under 200 characters. 85 of those are labelled
legitimate.**

### Why this breaks the evaluation

Because the classes differ so sharply in length, a program that does nothing but
count characters scores well:

```
Text length ALONE as a scam detector on this dataset:
  PR-AUC   0.8847
  ROC-AUC  0.9355

Best single rule: "text >= 500 characters => scam"
  F1 0.8082   precision 0.7973   recall 0.8194
```

For comparison, the trained transformer scores **0.8763** on EMSCAD. So on this
holdout, **counting characters outperforms a fine-tuned neural network.**

That means if you run the model here and it scores 0.92, you cannot tell whether it
detected fraud or just noticed the text was long. The number would be
uninterpretable.

### The second reason it's wrong

Even ignoring the statistics: **nobody will ever paste a 70-character index row into
TELITI.** Users paste job ads they received. Evaluating on index rows measures
performance on input that does not occur in practice.

### What is NOT wrong

To be clear about what doesn't need touching:

- ✅ **The 72 scam items are fine.** Real forwarded messages from community reports,
  proper length, exactly what users paste.
- ✅ **114 items overall are fine** — all 72 scams plus 42 legitimate ones.
- ✅ **The size and balance are right.** 200 items, 36% scam. The spec asked for ~200
  at ~35%.
- ✅ **The schema is clean.** `ml/validate_eval_set.py` passes with no errors.

This is a sourcing problem on a subset, not a design failure.

---

## Part 2 — The requirement

### The single rule everything follows from

> **Every `text` field must be exactly what a user would paste into TELITI.**

If a real person encountered this job ad and wanted to check it, whatever they'd
select and copy — that's the item.

### For legitimate ads

A complete job posting. It should normally include:

- Job title and the company name
- What the role involves
- Requirements or qualifications
- How to apply (email, link, or contact)

Optional but good: salary, location, benefits, employment type.

### For scam ads

The complete message as received — WhatsApp forward, Telegram message, Instagram
caption. Keep it verbatim: emoji, ALL CAPS, typos, weird spacing, broken grammar.
**The messiness is signal.** Don't tidy it.

### Length is the check, not the rule

Don't pad text to hit a number. Length is how you *verify* you copied the whole ad,
not the goal.

| | Typical range | Red flag |
|---|---|---|
| Legitimate ad | 300 – 1,500 chars | Under 200 → you probably copied a summary |
| Scam message | 200 – 2,500 chars | Under 150 → probably truncated |

If a genuine full posting really is short, that's fine — keep it. The point is that
it should be *complete*, not *long*.

### ⛔ What must never be done

**Do not generate or extend text with AI.** Do not write your own job ads. Do not
paraphrase.

Three reasons:

1. **It answers the wrong question.** The holdout tests "does this work on real job
   ads?" Fabricated items test "does this work on text we invented?"
2. **AI text has a fingerprint.** The model would likely learn *"sounds like an LLM →
   legitimate"* — a new confound, harder to detect than the length one.
3. **It cannot be reported.** The first thing anyone asks about an evaluation set is
   where the data came from.

The codebase already enforces this: `ml/eval_set.py` marks fabricated items and
refuses to compute reportable metrics from any file containing them.

---

## Part 3 — How to fix it

### Step 1 — List what needs fixing

```bash
python -c "
import json
rows=[json.loads(l) for l in open('eval/indonesian_holdout.jsonl',encoding='utf-8') if l.strip()]
bad=[r for r in rows if len(r['text']) < 200]
print(f'{len(bad)} items need fixing\n')
for r in bad:
    print(f\"{r['id']}  {len(r['text']):4d} chars  {r['source_url']}\")
"
```

That prints 86 items. **85 are legitimate ads needing the fix below.** One
(`id-holdout-0040`) is a scam — handle it separately, see the note at the end of this
step.

### Step 2 — For each one

1. Open the `source_url`
2. Find the actual job posting (not the search results row)
3. Select the advertisement body — role, description, requirements, how to apply
4. Replace the `text` field with it

**Change only `text`.** Leave `id`, `label`, `source_url`, `source_type`, `channel`
exactly as they are.

⚠️ **Watch out for site boilerplate.** Job board pages often carry SEO filler —
paragraphs about "lowongan kerja Jakarta Timur" and district statistics that appear
on *every* page. Copying that would give all your legitimate items identical text,
which is a worse confound than the one you're fixing. Copy only the job-specific
content.

#### The one short scam item

`id-holdout-0040` (185 chars) is not a job ad at all — it's a forum comment
*describing* receiving one:

```
ya ni saya hr ini dapat wa panggilan kerja PT.ALLIANCE LOGISTIC NETWORK
alamat Jl.Tanjakan AURI BLOK A no.36 rt 10/01 UJUNG MENTENG.
UNTUK HR sabtu tgl 09/11/24 Jam 07.30 s/d 14.00 wib.
```

That's someone talking *about* a scam message, not the message itself. A user would
never paste this. Either find the original message in that thread, or delete the
item.

Worth checking the other community-report scams for the same pattern — a forum post
quoting a scam is fine, a forum post *summarising* one is not.

### If a source page is dead

Some links will have expired. Options, in order of preference:

1. Replace with a different real posting from the same source type — update
   `source_url` to match
2. Delete the item (you'll end up with slightly fewer than 200, which is fine)

Don't leave the stub in place.

### Step 3 — Verify

```bash
python ml/validate_eval_set.py
```

Then check the confound is actually gone:

```bash
python -c "
import json, numpy as np, statistics as st
from sklearn.metrics import average_precision_score
rows=[json.loads(l) for l in open('eval/indonesian_holdout.jsonl',encoding='utf-8') if l.strip()]
y=np.array([r['label'] for r in rows])
L=np.array([len(r['text']) for r in rows],float)
print('length-only PR-AUC :', round(float(average_precision_score(y,L)),4))
print('prevalence (floor) :', round(float(y.mean()),4))
print('scam  median chars :', st.median([len(r['text']) for r in rows if r['label']==1]))
print('legit median chars :', st.median([len(r['text']) for r in rows if r['label']==0]))
print('items under 200    :', sum(1 for r in rows if len(r['text'])<200))
"
```

**What good looks like:**

| Check | Target |
|---|---|
| Items under 200 chars | 0 (or only genuinely short real ads) |
| Scam vs legit median length | Within roughly 2× of each other |
| Length-only PR-AUC | Close to the prevalence floor — under ~0.60 |

The last one is the real test. If length-only PR-AUC drops near the prevalence
(≈0.36), the confound is gone and any model score you report is interpretable.

---

## Part 4 — Two smaller issues

Lower priority than the length problem, but worth knowing.

### Source perfectly predicts label

| source_type | Legitimate | Scam |
|---|---:|---:|
| `community_report` | 0 | 72 |
| `job_board` | 110 | 0 |
| `company_official` | 10 | 0 |
| `glints` | 5 | 0 |
| `media` | 3 | 0 |

Zero overlap. `source_type` isn't fed to the model so this isn't leakage directly,
but it means the classes differ systematically in origin — the length gap is one
symptom, and there may be others (formatting conventions, vocabulary).

**Ideal fix:** a few legitimate ads that circulated on WhatsApp, and a few scams
posted to job boards. Realistically hard to source, so this may just be a stated
limitation.

### No Telegram items at all

Channels present: `job_board` 123, `whatsapp` 44, `other` 33. **Telegram: zero.**

This matters because the `contact_messaging_only` rule deliberately weights Telegram
at **0.75** versus WhatsApp at **0.35** — in Indonesia, WhatsApp is ordinary business
communication while Telegram is not. That asymmetry is one of the more considered
Indonesian-context decisions in the rule layer, and nothing in the holdout tests it.

**Fix:** even 5–10 Telegram-channel scam items would let that decision be evaluated.

---

## Part 5 — If there isn't time to fix it

You can still report a valid result. When a dataset has a known confound, standard
practice is to **report the confound's baseline alongside your number**:

> Pada himpunan evaluasi Indonesia, model mencapai PR-AUC **[X]**. Sebagai
> pembanding, klasifikasi berdasarkan panjang teks semata mencapai **0,885** pada
> himpunan yang sama — karena lowongan penipuan bersumber dari pesan WhatsApp yang
> lebih panjang daripada ringkasan papan lowongan. Model karena itu memberikan
> peningkatan sebesar **[X − 0,885]** di luar yang dapat dijelaskan oleh panjang
> dokumen semata.

This is honest and defensible. It shows you found a confound in your own data,
measured it, and reported it — which is what good methodology looks like.

⚠️ What would **not** be acceptable is reporting the model's score without
mentioning that length alone achieves 0.885 on the same data.

---

## Summary

| | |
|---|---|
| **Problem** | 86 items are search-result summaries, not job ads |
| **Effect** | Counting characters scores 0.8847 — the evaluation can't be interpreted |
| **Requirement** | Every item = exactly what a user would paste |
| **Fix** | Open each `source_url`, copy the real ad body, replace `text` only |
| **Never** | Generate or extend text with AI |
| **Verify** | Length-only PR-AUC drops near the prevalence floor (~0.36) |
| **If no time** | Report the model score *and* the 0.885 length baseline together |
