# Concept paper revisions

Five changes required by what measurement actually showed, plus one pending your
dataset decision. Each gives the current text, the replacement, and the evidence.

Indonesian drafts are written to slot into the existing prose. Every number traces
to a committed artefact — nothing here is estimated.

---

## 1. §3.3 — the fusion mechanism ⚠️ **required, factual**

### Current text (paraphrased)

> Sebagian sinyal aturan (domain email, keberadaan profil perusahaan) tersedia
> langsung pada metadata EMSCAD sehingga dapat dipelajari model meta.

### Why it must change

Measurably false. `ml/verify_derivability.py` over all 17,880 documents:

| property | rate |
|---|---:|
| documents containing a real email address | **0.02%** |
| documents containing a real URL | **0.39%** |
| documents with a visible `#EMAIL_x#` placeholder | 25.6% |

EMSCAD removed contact details before publication. Only a quarter left a marker
behind; the rest were stripped with no trace, so a rule cannot even detect that it
is looking at redacted text.

Worse, availability differs by class — **74.8%** on genuine postings versus
**67.9%** on fraudulent ones. A model given that feature partly learns *how the
corpus was anonymised* rather than what a scam looks like: excellent offline
metrics from an artefact that cannot exist in a user's pasted text.

### Replacement

> Rancangan awal mengasumsikan bobot lapis aturan dapat dipelajari dari metadata
> EMSCAD. Verifikasi empiris terhadap seluruh 17.880 dokumen menunjukkan asumsi ini
> tidak dapat dipertahankan: hanya 0,02% dokumen memuat alamat surel nyata dan
> 0,39% memuat URL nyata, karena EMSCAD menyensor detail kontak sebelum dipublikasi.
> Lebih penting lagi, ketersediaan sinyal tersebut berbeda antar kelas (74,8% pada
> lowongan asli berbanding 67,9% pada lowongan palsu), sehingga model yang
> memakainya akan mempelajari proses anonimisasi korpus, bukan karakteristik
> penipuan — menghasilkan metrik luring yang tinggi namun tidak dapat
> digeneralisasi ke teks yang ditempel pengguna.
>
> Karena itu bobot lapis aturan ditetapkan secara *a priori* berdasarkan
> penalaran domain, lalu **divalidasi** pada himpunan evaluasi held-out, bukan
> dilatih pada EMSCAD.

**Evidence:** `eval/derivability_report.md`, reproducible via
`python ml/verify_derivability.py`

---

## 2. §3.3 — the bounded-contribution cap ⚠️ **required**

### Current text

> ...diterapkan sebagai penalti aditif terbatas (*bounded*) dengan batas maksimum
> [agregat 0,15]...

### Why it must change

0.15 was sized for three penalty rules. The implemented system has **nine**. Shared
across nine, each rule is worth ~1.7 score points and the rule layer becomes
decorative — which matters because the text model transfers across languages
imperfectly, making the rules the most reliable local signal.

The paper's actual claim is *"tidak ada **satu** aturan deterministik pun yang dapat
mendominasi"* — a statement about any single rule, not about their sum.

### Replacement

> ...diterapkan sebagai penalti aditif terbatas (*bounded*), dengan batas
> **per-aturan** sebesar 0,10 pada probabilitas terfusi (setara maksimum 10 poin
> pada skala skor 0–100) dan plafon agregat 0,45. Batas per-aturan inilah yang
> menjamin tidak ada satu aturan deterministik pun yang dapat mendominasi keputusan
> model, sementara plafon agregat menjaga agar mayoritas keputusan tetap berasal
> dari model teks.

**Evidence:** `ml/feature_contract.py` (`PER_RULE_CONTRIBUTION_CAP = 0.10`,
`MAX_TOTAL_RULE_SHIFT = 0.45`), enforced at import and by
`tests/test_scoring.py::test_no_single_rule_can_dominate`

---

## 3. §3.3 — a priori weights, validated not fitted ⚠️ **required**

### Why

Follows from change 1. Fitting weights on Indonesian data would need a second
~200-item annotated set on top of the held-out set — not affordable within the
timeline.

### Text to add

> Sembilan bobot aturan ditetapkan sebelum evaluasi, diurutkan menurut kekuatan
> indikasi pada laporan penipuan yang dirujuk pada §1.1: permintaan pembayaran
> (0,10) sebagai penanda terkuat, diikuti pengalihan ke kanal percakapan pribadi
> (0,08), kewajaran gaji (0,07), dan seterusnya hingga ketiadaan alamat surel
> (0,01). Bobot ini **tidak** disetel ulang berdasarkan kinerja pada himpunan
> held-out; penyetelan semacam itu akan menjadikan himpunan tersebut data latih
> dan membatalkan metrik utama. Kontribusi lapis aturan dilaporkan apa adanya,
> termasuk apabila validasi menunjukkan lapis tersebut tidak meningkatkan kinerja.

⚠️ **Operational rule for the team:** never adjust `ml/rule_weights.py` in response
to holdout numbers. The moment you do, the holdout is training data.

**Evidence:** `ml/rule_weights.py`; validation harness `ml/validate_rule_weights.py`

---

## 4. §3.2 — SMOTE tested, not adopted ✅ **strengthens the paper**

### Current text

> ...ditangani dengan teknik *oversampling* (SMOTE) atau pembobotan kelas...

### Why change

Measured on the val split — SMOTE performed slightly *worse*:

| model | PR-AUC |
|---|---:|
| tfidf + linearsvc | **0.8769** |
| tfidf + logreg (class weighting) | 0.8662 |
| tfidf + SMOTE + logreg | 0.8638 |
| tfidf + histgb | 0.8212 |

There is also a correctness point worth stating: SMOTE interpolates feature vectors.
That is meaningful for TF-IDF but **meaningless for token sequences** — you cannot
average two sentences into a valid sentence — so it cannot be applied to the
transformer at all.

### Replacement

> Ketidakseimbangan kelas (4,83% penipuan, `pos_weight` ≈ 19,7) diuji dengan dua
> pendekatan. SMOTE diterapkan pada jalur TF-IDF dan menghasilkan PR-AUC 0,8638,
> sedikit di bawah pembobotan kelas sederhana (0,8662); pendekatan pembobotan
> karena itu yang digunakan. Untuk model *transformer*, SMOTE tidak dapat
> diterapkan secara prinsipil: teknik ini menginterpolasi vektor fitur, operasi
> yang bermakna pada representasi TF-IDF namun tidak terdefinisi pada barisan token.
> Ketidakseimbangan pada model utama ditangani melalui *weighted cross-entropy*.

**Evidence:** `eval/baseline_results.md`

---

## 5. §3.2 — corpus quality note ✅ **recommended**

### Why

Signals that the data was actually inspected. The Kaggle EMSCAD distribution has
HTML stripped **without inserting separators**, fusing words across former tag
boundaries in **78.1%** of documents (median 6 per document):

> "...Research **InstituteOur** passion for improving..."
> "...Account **ExecutiveAs** a member of..."

### Text to add

> Distribusi EMSCAD yang tersedia publik telah dibersihkan dari markah HTML tanpa
> menyisipkan pemisah, sehingga kata-kata menyatu pada bekas batas tag pada 78,1%
> dokumen (median 6 kejadian per dokumen). Kondisi ini diperbaiki sebelum
> pelatihan, dengan daftar-lindung istilah teknis (*JavaScript*, *PostgreSQL*,
> *PowerPoint*) agar kosakata sah tidak ikut terpecah.

**Evidence:** `ml/text_cleaning.py`; `data/processed/prepare_report.md`

---

## 6. §1.1 / §3.2 — language scope ⏸️ **pending your decision**

Not yet actionable — depends on which evaluation set your team can source.

### If you obtain Indonesian data

No change. The paper stands as written, and §3.2's cross-language validation plan
proceeds as designed.

### If the evaluation set ends up English-only

§1.1's Indonesian framing stays as **motivation**, but must be paired with an
explicit scope limitation. Do not delete the Indonesian context — it is the
reason the problem matters — but do not imply coverage you have not measured.

Suggested addition to §3.2:

> Himpunan evaluasi yang berhasil dikumpulkan pada tahap ini berbahasa Inggris.
> Sistem dirancang dengan lapisan lokal (*locale layer*) yang memisahkan sumber
> daya bahasa dari logika penilaian: kamus frasa risiko dan tabel upah minimum
> dimuat per bahasa, sehingga dukungan bahasa Indonesia dapat diaktifkan dengan
> menambahkan berkas sumber daya tanpa mengubah kode. Aturan yang sumber dayanya
> belum tersedia melaporkan diri sebagai *tidak dapat dinilai*, bukan *bersih*,
> sehingga analisis parsial tidak pernah tampak sebagai analisis lengkap.
> Evaluasi pada lowongan berbahasa Indonesia merupakan pekerjaan lanjutan.

**Evidence:** `docs/LOCALES.md`, `api/locale.py`

---

## What does NOT change

The Integrity Score formula, the four-layer architecture, the transformer choice,
the XAI approach, the ethics section, the roadmap, and every citation.

These are refinements from contact with data, not a redesign. A methods section
that says *"we tested our assumption, it failed, and here is what we did instead"*
is stronger than one that never checked.

---

## Numbers you may quote

All traceable to committed artefacts:

| claim | value | source |
|---|---|---|
| EMSCAD size | 17,880 postings, 866 fraudulent (4.84%) | `data/processed/split_manifest.json` |
| Baseline PR-AUC | 0.8769 (TF-IDF + LinearSVC) | `eval/baseline_results.md` |
| Baseline precision / recall | 0.930 / 0.767 | `eval/baseline_results.md` |
| Real emails in corpus | 0.02% | `eval/derivability_report.md` |
| Class skew in availability | 74.8% vs 67.9% | `eval/derivability_report.md` |
| Word fusion rate | 78.1% of documents | `data/processed/prepare_report.md` |
| Rule-layer latency (20k chars) | 86 ms p50 | step 4.3 hardening |

⚠️ **Do not quote any Integrity Score yet.** The text model is stubbed; every score
the system currently produces is synthetic. `/health` reporting
`model_loaded: false` is the check.
