# Paper revision guide — semifinal submission

Everything that must change in the concept paper, why, and the replacement text.

This supersedes the earlier `PAPER_REVISIONS.md`; all of its content is folded in
below and updated where measurement has since overtaken it.

**Deadline: 10 August 2026.** Max 12 pages, NeurIPS 2015 format, submitted as PDF.
The problem and the proposed product must stay consistent with the preliminary
concept paper — these are corrections and additions, not a redesign.

---

## How to use this document

Each item gives you: **what the paper currently says**, **why it is now wrong or
incomplete**, **replacement text in Indonesian** ready to paste, and the
**artefact that proves it**. Nothing here is estimated; every number traces to a
committed file.

Priority markers:

| | Meaning |
|---|---|
| 🔴 | **Required.** The paper currently states something measurement has falsified. Leaving it is a factual error a judge can check against your repo. |
| 🟡 | **Required by the semifinal brief.** New sections the round asks for. |
| 🟢 | **Recommended.** Strengthens the paper; not a correction. |

Work 🔴 first, then 🟡. 🟢 only if pages remain.

---

## Change register

| # | Section | Change | Priority |
|---|---|---|---|
| A1 | §3.1, §3.3 | Rule layer is **advisory only** — it does not affect the score | 🔴 |
| A2 | §3.3 | Rule weights cannot be learned from EMSCAD metadata | 🔴 |
| A3 | §3.3 | Bounded penalty: per-rule 0,10 + aggregate 0,45, not aggregate 0,15 | 🔴 |
| A4 | §3.3 | Weights set *a priori* and validated, not fitted | 🔴 |
| A5 | §3.2 | SMOTE tested and rejected, with reasoning | 🔴 |
| A6 | §1.5 / §3.5 | ONNX INT8 **not** implemented — remove or reframe | 🔴 |
| A7 | §3.4 | XAI operates on logit margins, not probabilities | 🔴 |
| A8 | §3.3 | Calibration is domain-specific; two calibrators exist | 🔴 |
| A9 | §3.2 | EMSCAD corpus quality: redaction and word fusion | 🟢 |
| B1 | **new** | Results section | 🟡 |
| B2 | **new** | Discussion section | 🟡 |
| B3 | **new** | Limitations subsection | 🟡 |
| C1 | §1.2 | URL ingestion implemented, with SSRF defence | 🟢 |
| C2 | §3.6 | Appeal/correction mechanism implemented | 🟢 |
| C3 | §3.2 | Indonesian evaluation set exists — resolve the open scope question | 🟢 |
| C4 | §4 | Reproducibility: model and code links | 🟢 |

---

# PART A — Required factual corrections

## A1 🔴 §3.1 and §3.3 — the rule layer does not affect the score

### What the paper says

That the Integrity Score fuses a text-model probability with a deterministic rule
layer, and that this fusion is the core of the method.

### Why it must change

It was implemented, measured, and **turned off**, because it made the system
worse. On the Indonesian holdout (195 items, 36,4% scam):

| Configuration | PR-AUC | False positives |
|---|---:|---:|
| **model only** | **0,9258** | 5 |
| model + rules | 0,8617 | 28 |
| rules only | 0,4167 | 93 |

Adding the rules lowered PR-AUC by 0,064 and multiplied false positives by 5,6.
§3.6 of your own paper names false positives against real businesses as the
expensive error, so keeping the layer would contradict your ethics section.

The layer still runs. Its findings are shown to the user as context. It simply
does not move the number.

This is the single most important correction in this document. A judge who reads
`api/scoring.py` will find `RULE_LAYER_ENABLED = False` on line 44 with the
measurement in a comment above it. The paper must agree.

### Replacement text

> Lapis aturan deterministik diimplementasikan penuh sebagaimana dirancang,
> kemudian **dievaluasi dan dinonaktifkan dari perhitungan skor**. Pada himpunan
> evaluasi berbahasa Indonesia (195 item, prevalensi 36,4%), penggabungan lapis
> aturan dengan model teks menurunkan PR-AUC dari 0,9258 menjadi 0,8617 dan
> meningkatkan positif palsu dari 5 menjadi 28. Karena §3.6 menetapkan positif
> palsu terhadap perusahaan sah sebagai kesalahan yang paling mahal, lapis aturan
> tidak digunakan untuk menggeser skor.
>
> Lapis tersebut tetap dijalankan dalam **modus informatif**: temuannya
> ditampilkan kepada pengguna sebagai konteks yang dapat diperiksa — misalnya
> permintaan pembayaran atau gaji yang tidak wajar terhadap UMK — tanpa
> memengaruhi nilai Skor Integritas. Keputusan ini diambil berdasarkan
> pengukuran, bukan asumsi, dan dilaporkan apa adanya.

**Evidence:** `api/scoring.py:44`, `eval/indonesian_results.md`, `eval/results.md` §3

---

## A2 🔴 §3.3 — rule weights cannot be learned from EMSCAD

### What the paper says

> Sebagian sinyal aturan (domain email, keberadaan profil perusahaan) tersedia
> langsung pada metadata EMSCAD sehingga dapat dipelajari model meta.

### Why it must change

Measurably false. `ml/verify_derivability.py` across all 17.880 documents:

| Property | Rate |
|---|---:|
| documents containing a real email address | **0,02%** |
| documents containing a real URL | **0,39%** |
| documents with a visible `#EMAIL_x#` placeholder | 25,6% |

EMSCAD removed contact details before publication. Only a quarter left a marker;
the rest were stripped without trace, so a rule cannot even detect that it is
looking at redacted text.

Worse, availability differs by class — **74,8%** on genuine postings versus
**67,9%** on fraudulent ones. A model given that feature partly learns *how the
corpus was anonymised* rather than what a scam looks like.

### Replacement text

> Rancangan awal mengasumsikan bobot lapis aturan dapat dipelajari dari metadata
> EMSCAD. Verifikasi empiris terhadap seluruh 17.880 dokumen menunjukkan asumsi
> ini tidak dapat dipertahankan: hanya 0,02% dokumen memuat alamat surel nyata dan
> 0,39% memuat URL nyata, karena EMSCAD menyensor detail kontak sebelum
> dipublikasi. Lebih penting lagi, ketersediaan sinyal tersebut berbeda antar
> kelas (74,8% pada lowongan asli berbanding 67,9% pada lowongan palsu), sehingga
> model yang memakainya akan mempelajari proses anonimisasi korpus, bukan
> karakteristik penipuan — menghasilkan metrik luring yang tinggi namun tidak
> dapat digeneralisasi ke teks yang ditempel pengguna.
>
> Karena itu bobot lapis aturan ditetapkan secara *a priori* berdasarkan penalaran
> domain, lalu **divalidasi** pada himpunan evaluasi held-out, bukan dilatih pada
> EMSCAD.

**Evidence:** `eval/derivability_report.md`, reproducible via `python ml/verify_derivability.py`

---

## A3 🔴 §3.3 — the bounded-contribution cap

### What the paper says

> ...diterapkan sebagai penalti aditif terbatas (*bounded*) dengan batas maksimum
> [agregat 0,15]...

### Why it must change

0,15 was sized for three penalty rules. The implemented system has **nine**.
Shared across nine, each rule is worth ~1,7 score points and the cap stops
meaning what the paper claims.

The paper's actual claim is *"tidak ada **satu** aturan deterministik pun yang
dapat mendominasi"* — a statement about any single rule, not about their sum.

### Replacement text

> ...diterapkan sebagai penalti aditif terbatas (*bounded*), dengan batas
> **per-aturan** sebesar 0,10 pada probabilitas terfusi (setara maksimum 10 poin
> pada skala skor 0–100) dan plafon agregat 0,45. Batas per-aturan inilah yang
> menjamin tidak ada satu aturan deterministik pun yang dapat mendominasi
> keputusan model, sementara plafon agregat menjaga agar mayoritas keputusan tetap
> berasal dari model teks. Batas ini tetap berlaku pada implementasi meskipun
> lapis aturan saat ini dinonaktifkan dari perhitungan skor (lihat A1).

**Evidence:** `ml/feature_contract.py` (`PER_RULE_CONTRIBUTION_CAP = 0.10`,
`MAX_TOTAL_RULE_SHIFT = 0.45`), enforced at import and by
`tests/test_scoring.py::test_no_single_rule_can_dominate`

---

## A4 🔴 §3.3 — weights set *a priori*, validated not fitted

### Why

Follows from A2. Fitting weights on Indonesian data would need a second ~200-item
annotated set on top of the held-out set.

### Text to add

> Sembilan bobot aturan ditetapkan sebelum evaluasi, diurutkan menurut kekuatan
> indikasi pada laporan penipuan yang dirujuk pada §1.1: permintaan pembayaran
> (0,10) sebagai penanda terkuat, diikuti pengalihan ke kanal percakapan pribadi
> (0,08), kewajaran gaji terhadap UMK (0,07), dan seterusnya hingga ketiadaan
> alamat surel (0,01). Bobot ini **tidak** disetel ulang berdasarkan kinerja pada
> himpunan held-out; penyetelan semacam itu akan menjadikan himpunan tersebut data
> latih dan membatalkan metrik utama. Kontribusi lapis aturan dilaporkan apa
> adanya, termasuk ketika validasi menunjukkan lapis tersebut justru menurunkan
> kinerja.

⚠️ **Operational rule for the team:** never adjust `ml/rule_weights.py` in
response to holdout numbers. The moment you do, the holdout becomes training data.

**Evidence:** `ml/rule_weights.py`; harness `ml/validate_rule_weights.py`

---

## A5 🔴 §3.2 — SMOTE tested, not adopted

### What the paper says

> ...ditangani dengan teknik *oversampling* (SMOTE) atau pembobotan kelas...

### Why it must change

Measured on val — SMOTE performed slightly *worse*:

| Model | PR-AUC |
|---|---:|
| tfidf + linearsvc | **0,8769** |
| tfidf + logreg (class weighting) | 0,8662 |
| tfidf + SMOTE + logreg | 0,8638 |
| tfidf + histgb | 0,8212 |

There is also a correctness point worth stating: SMOTE interpolates feature
vectors. That is meaningful for TF-IDF but **meaningless for token sequences** —
you cannot average two sentences into a valid sentence — so it cannot be applied
to the transformer at all.

### Replacement text

> Ketidakseimbangan kelas (4,83% penipuan, `pos_weight` ≈ 19,7) diuji dengan dua
> pendekatan. SMOTE diterapkan pada jalur TF-IDF dan menghasilkan PR-AUC 0,8638,
> sedikit di bawah pembobotan kelas sederhana (0,8662); pendekatan pembobotan
> karena itu yang digunakan. Untuk model *transformer*, SMOTE tidak dapat
> diterapkan secara prinsipil: teknik ini menginterpolasi vektor fitur, operasi
> yang bermakna pada representasi TF-IDF namun tidak terdefinisi pada barisan
> token. Ketidakseimbangan pada model utama ditangani melalui *weighted
> cross-entropy*.

**Evidence:** `eval/baseline_results.md`

---

## A6 🔴 §1.5 / §3.5 — ONNX INT8 was not implemented

### Why it must change

If the paper claims ONNX INT8 serving, **remove or reframe it**. The product
serves PyTorch directly. The optimisation was planned, then skipped because the
latency target was already met without it — but an unimplemented claim in a paper
whose repo a judge can read is the worst kind of error.

Check your §1.5, §3.5 and any architecture figure for the word ONNX.

### Replacement text

> Model dilayani langsung menggunakan PyTorch pada CPU. Kuantisasi INT8 melalui
> ONNX Runtime direncanakan sebagai optimisasi, namun tidak diperlukan: latensi
> terukur sudah memenuhi anggaran interaktif (p50 469 ms termasuk pembangkitan
> penjelasan, lihat §Hasil). Optimisasi ini dicatat sebagai pekerjaan lanjutan
> apabila beban meningkat.

**Evidence:** `requirements.txt` (onnxruntime present but unused for serving),
`api/model.py` (loads PyTorch), `eval/results.md` §4

---

## A7 🔴 §3.4 — XAI operates on logit margins

### Why it must change

If the paper describes occlusion measured as a change in *probability*, that is
now wrong, and the reason is worth a sentence — it demonstrates you debugged your
own explanation layer.

Deleting a sentence and re-scoring changed the calibrated probability by
`+0,0039`, `+0,0013`, `+0,0002` — visually indistinguishable from zero. The
calibrated sigmoid is saturated near its ends, so genuinely important sentences
produced deltas that rounded away. Measuring the same deletions in **logit-margin
space** gave `+2,083`, `+1,538`, `+0,607` — the same ranking, legible.

### Replacement text

> Bukti tingkat kalimat dihasilkan melalui *leave-one-out occlusion*: setiap
> kalimat dihapus secara bergantian dan model dijalankan ulang. Selisih diukur
> pada **ruang margin logit**, bukan pada probabilitas terkalibrasi. Pada
> probabilitas, fungsi sigmoid berada di daerah jenuh sehingga penghapusan kalimat
> yang jelas penting hanya menghasilkan selisih sebesar 0,0039 — tidak dapat
> dibedakan dari nol oleh pengguna. Pada ruang margin, penghapusan yang sama
> menghasilkan selisih 2,083, dengan peringkat kalimat yang identik. Pengukuran
> dilakukan pada 12 kalimat pertama, sejalan dengan batas 256 token pada masukan
> model: kalimat setelahnya tidak pernah dibaca model sehingga selisihnya
> dipastikan nol.

**Evidence:** `api/explain.py`, `tests/test_explain.py`

---

## A8 🔴 §3.3 — calibration is domain-specific

### Why it must change

This is a genuine methodological contribution and the paper should claim it.

The model was trained where scams are 4,8% of postings and deployed where they
are ~36%. With the EMSCAD-fitted calibrator every Indonesian advertisement scored
**93–100** — ranked correctly, but useless as a number a person can act on. After
recalibrating on the Indonesian holdout the range became **0–97**.

Two calibrators therefore ship: `calibrator.json` (EMSCAD) and
`calibrator_deployment.json` (Indonesian, used in production).

### Replacement text

> Kalibrasi dilakukan dengan *Platt scaling* pada margin logit. Satu temuan
> penting muncul pada tahap ini: kalibrator yang dipasang pada EMSCAD membawa
> serta prevalensi 4,83% ke ranah penerapan yang prevalensinya ±36%, sehingga
> setiap iklan berbahasa Indonesia memperoleh skor 93–100. Urutan peringkat tetap
> benar, namun angka yang ditampilkan kepada pengguna kehilangan makna.
>
> Kalibrator karena itu dipasang ulang pada ranah penerapan
> (`calibrator_deployment.json`), mengembalikan rentang skor menjadi 0–97. Pada
> himpunan uji EMSCAD, kalibrasi menurunkan ECE dari 0,0234 menjadi 0,0158 dan
> skor Brier dari 0,0250 menjadi 0,0226. Pemisahan ini penting karena Skor
> Integritas disajikan sebagai bilangan 0–100 kepada pencari kerja: model yang
> menyatakan 80 namun benar hanya 55% dari waktu menyampaikan angka yang
> menyesatkan, terlepas dari sebaik apa peringkatnya.

**Evidence:** `eval/results.md` §2, `eval/thresholds_report.md`, `ml/fit_thresholds.py`

---

## A9 🟢 §3.2 — EMSCAD corpus quality

### Why

Signals that the data was actually inspected. The Kaggle EMSCAD distribution has
HTML stripped **without inserting separators**, fusing words across former tag
boundaries in **78,1%** of documents (median 6 per document):

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

# PART B — New sections required by the semifinal

The brief says the paper keeps its structure **with the addition of a Results and
Discussion section**. This is 10% of the rubric on its own
(*Evaluation, Reliability & Responsible AI*), and another 20% depends on it
(*AI Implementation & Technical Excellence*).

Every number below is in `eval/results.md`. Quote nothing that is not.

---

## B1 🟡 Results section

### B1.1 Experimental setup

> Model dasar `distilbert-base-multilingual-cased` disetel-halus pada EMSCAD
> (17.880 lowongan, 866 penipuan, 4,83%). Data dibagi menurut *stratified split*
> dengan benih tetap 42 menjadi train (12.513), val (1.788), calib (1.788) dan
> test (1.788); rasio penipuan tiap bagian berada dalam ±0,5 poin persen dari
> rasio populasi. Panjang masukan dibatasi 256 token, laju pembelajaran 3×10⁻⁵,
> ukuran *batch* 16, tiga *epoch*, `pos_weight` 19,69.
>
> Pemilihan *checkpoint* dilakukan pada val. Kalibrasi dipasang pada calib.
> Bagian **test tidak disentuh sama sekali** hingga evaluasi akhir, dan hanya
> dinilai satu kali.

### B1.2 Primary metric

> Metrik utama adalah **PR-AUC** pada kelas penipuan. Akurasi tidak digunakan
> sebagai metrik utama: pada prevalensi 4,81%, menebak "bukan penipuan" untuk
> seluruh item menghasilkan akurasi 95,19%. Perlu ditegaskan bahwa PR-AUC
> bergantung pada prevalensi sehingga **tidak dapat dibandingkan lintas himpunan
> data** — nilai pada EMSCAD dan pada himpunan Indonesia tidak setara.

### B1.3 Main results — copy this table

| Himpunan | Model | PR-AUC | ROC-AUC | Presisi | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| EMSCAD test | mDistilBERT | **0,7883** | 0,9647 | 0,9273 | 0,5930 | 0,7234 |
| EMSCAD test | TF-IDF + LinearSVC | **0,8651** | 0,9839 | 0,8462 | 0,7674 | 0,8049 |
| Indonesia holdout | mDistilBERT | **0,9258** | 0,9357 | 0,9231 | 0,8451 | 0,8824 |

Confusion matrix, EMSCAD test, threshold 0,9326:

| | diprediksi asli | diprediksi penipuan |
|---|---:|---:|
| **asli** | 1.698 | 4 |
| **penipuan** | 35 | 51 |

### B1.4 The val → test drop — do not omit this

> | Bagian | Transformer | TF-IDF | Peran |
> |---|---:|---:|---|
> | val | 0,8669 | 0,8769 | pemilihan *checkpoint* |
> | calib | 0,8509 | 0,8948 | pemasangan kalibrator |
> | **test** | **0,7883** | **0,8651** | **tidak tersentuh** |
>
> Transformer turun 0,0786 PR-AUC dari val ke test, sedangkan TF-IDF hanya turun
> 0,0118. Asimetri ini memiliki penyebab yang jelas: *checkpoint* transformer
> dipilih berdasarkan skor pada val, sehingga val merupakan penaksir yang
> optimistis. TF-IDF tidak melalui pemilihan semacam itu. **Angka yang sahih untuk
> generalisasi adalah 0,7883**, dan nilai itulah yang dilaporkan.

### B1.5 Ablation

> | Konfigurasi | PR-AUC | Positif palsu |
> |---|---:|---:|
> | model saja | **0,9258** | 5 |
> | model + aturan | 0,8617 | 28 |
> | aturan saja | 0,4167 | 93 |
> | **panjang teks saja** | **0,3734** | 111 |
>
> Diukur pada himpunan Indonesia. Aturan saja (0,4167) hanya sedikit di atas
> batas bawah prevalensi (0,3641), menunjukkan aturan buatan tangan nyaris tidak
> informatif secara mandiri pada ranah ini.

**Include the length-only row.** It is a *negative control*, and it is one of the
strongest methodological signals in the whole evaluation.

An earlier draft of the Indonesian evaluation set had scam advertisements written
much shorter than genuine ones. A classifier using **nothing but character count**
scored 0,8847 PR-AUC on it — meaning the set could be solved without reading a
single word, and any model evaluated on it would have looked excellent for the
wrong reason. The set was rebuilt with matched lengths, and the same length-only
control now scores 0,3734, barely above the 0,3641 prevalence floor.

> Sebagai kontrol negatif, pengklasifikasi yang hanya menggunakan panjang teks
> diuji pada himpunan evaluasi. Versi awal himpunan tersebut memberikan PR-AUC
> 0,8847 pada kontrol ini — menandakan himpunan dapat diselesaikan tanpa membaca
> isi iklan sama sekali, sehingga metrik model apa pun di atasnya tidak bermakna.
> Himpunan disusun ulang dengan panjang yang seimbang antar kelas, dan kontrol
> yang sama kini memperoleh 0,3734, praktis setara dengan batas bawah prevalensi
> (0,3641). Kontrol ini dilaporkan agar pembaca dapat memastikan hasil utama tidak
> berasal dari artefak panjang teks.

**Evidence:** `eval/indonesian_results.md` (length-only row)

### B1.6 Calibration

> | Metrik | Softmax mentah | Terkalibrasi |
> |---|---:|---:|
> | Skor Brier | 0,0250 | **0,0226** |
> | ECE | 0,0234 | **0,0158** |

### B1.7 Latency

> Diukur pada CPU melalui jalur penyajian lengkap, 30 sampel:
>
> | Jalur | p50 | p95 | p99 |
> |---|---:|---:|---:|
> | skor saja | 86 ms | 99 ms | 103 ms |
> | skor + penjelasan | 469 ms | 949 ms | 992 ms |
>
> Baris kedua adalah angka yang relevan: penjelasan tidak bersifat opsional pada
> antarmuka.

⚠️ **State the hardware, and treat these as approximate.** Unlike the accuracy
figures — which are deterministic and reproduce to four decimal places on every
run — latency is wall-clock and moves with machine load. Two runs minutes apart
gave p50 519 ms and 469 ms on the same laptop. Quote the figures currently in
`eval/results.md`, round them, and name the CPU you measured on. Do not present
them to three significant figures as though they were exact.

### B1.8 Error analysis — write this from the real data

Open `eval/results.md` §5 and read the actual excerpts before writing. Observed
pattern in the false positives: **clerical and administrative roles quoting an
hourly wage** ("Payroll Clerk $20/hr", "Accounting Clerk $15/hr", "Production
Operator $16.50/hr"). Postings that read as terse and transactional resemble
scams in surface form.

> Analisis galat pada 4 positif palsu paling meyakinkan menunjukkan pola yang
> konsisten: lowongan administratif dan klerikal yang mencantumkan upah per jam.
> Iklan sah yang ringkas dan transaksional memiliki ciri permukaan yang mirip
> dengan penipuan. Pada 35 negatif palsu, kesalahan justru terjadi pada penipuan
> yang ditulis dengan bahasa korporat yang wajar dan panjang — jenis penipuan yang
> paling merugikan karena paling meyakinkan.

⚠️ Verify the false-negative characterisation against `eval/results.md` §5 before
submitting. Do not assert a pattern you have not read.

---

## B2 🟡 Discussion section

### B2.1 The uncomfortable result, addressed directly

Do not bury this. A judge who reads `eval/results.md` will find it in thirty
seconds, and a paper that reported it first is far stronger than one that did not.

> Hasil paling penting dari evaluasi ini adalah bahwa *transformer* multibahasa
> **tidak mengungguli** garis dasar TF-IDF pada EMSCAD berbahasa Inggris (0,7883
> berbanding 0,8651). Temuan ini dilaporkan secara terbuka.
>
> Penjelasannya dapat dipertanggungjawabkan. Lowongan penipuan berbahasa Inggris
> pada EMSCAD memiliki pola permukaan yang sangat formulaik — frasa berulang,
> struktur kalimat serupa — dan model linear atas n-gram karakter serta kata
> justru sangat efektif menangkap pola semacam itu. Keunggulan *transformer*
> terletak pada generalisasi semantik lintas bahasa, kemampuan yang tidak diuji
> oleh korpus berbahasa Inggris.
>
> Justifikasi pemilihan *transformer* karena itu bukan angka pada EMSCAD,
> melainkan **kemampuan alih-bahasa**: TF-IDF yang dilatih pada teks Inggris tidak
> memiliki kosakata untuk memproses "lowongan", "gaji", atau "wajib transfer" dan
> secara prinsipil tidak dapat dialihkan ke bahasa Indonesia tanpa data latih
> Indonesia berskala besar. Pada himpunan Indonesia, *transformer* memperoleh
> PR-AUC 0,9258 tanpa satu pun contoh latih berbahasa Indonesia.

### B2.2 On disabling a component you built

> Keputusan menonaktifkan lapis aturan diambil setelah pengukuran, bukan sebelum
> implementasi. Lapis tersebut dibangun lengkap, diuji dengan 9 aturan dan
> serangkaian kasus emas, lalu dinonaktifkan karena bukti menunjukkan ia menaikkan
> positif palsu 5,6 kali lipat. Mempertahankannya demi konsistensi dengan rancangan
> awal akan bertentangan dengan §3.6, yang menetapkan positif palsu terhadap
> perusahaan sah sebagai kesalahan yang paling mahal.

### B2.3 Prior shift as a deployment lesson

> Pemindahan model lintas ranah tidak hanya mengubah akurasi, tetapi juga makna
> angka yang ditampilkan. Model yang dilatih pada prevalensi 4,83% dan diterapkan
> pada ranah berprevalensi 36% tetap mengurutkan dengan benar, namun seluruh skor
> menumpuk pada rentang 93–100. Untuk produk yang menyajikan bilangan 0–100 kepada
> orang awam, hal ini setara dengan kegagalan, meskipun tidak terlihat pada metrik
> berbasis peringkat seperti PR-AUC atau ROC-AUC.

---

## B3 🟡 Limitations subsection

State all of these. *Reliability & Error Analysis* rewards "consistency between
experimental results and product claims."

> **Keterbatasan.**
>
> 1. **Data latih berbahasa Inggris.** Model dilatih pada EMSCAD dan dievaluasi
>    pada 195 item berbahasa Indonesia yang dianotasi manual. Kinerja pada bahasa
>    Indonesia bersifat indikatif, belum mapan.
> 2. **EMSCAD telah disensor.** Alamat surel, URL dan nomor telepon dihapus
>    sebelum publikasi, sehingga model tidak pernah belajar dari detail kontak
>    nyata — justru sinyal yang paling menentukan pada penipuan sesungguhnya.
> 3. **Ambang batas dipasang pada himpunan yang sama dengan pelaporan.** Batas
>    Rendah/Sedang/Tinggi dipasang pada himpunan Indonesia karena pergeseran prior
>    membuat batas turunan EMSCAD melabeli hampir seluruh penipuan Indonesia
>    sebagai "Rendah". PR-AUC tidak terpengaruh karena bersifat bebas-ambang,
>    namun **presisi dan recall pada batas tersebut bersifat optimistis**.
> 4. **Jumlah positif kecil.** 86 positif pada himpunan uji EMSCAD dan 71 pada
>    himpunan Indonesia; selisih beberapa poin pada recall berada dalam derau.
> 5. **Ghost job belum ditangani.** Deteksi lowongan untuk posisi yang tidak
>    pernah ada merupakan masalah terpisah dan tidak diimplementasikan.
> 6. **Bukan vonis.** Skor tinggi bukan bukti penipuan dan skor rendah bukan
>    jaminan keamanan.

Limitation 3 is the one people forget. It is disclosed in
`artifacts/thresholds.json` under `caveat`; a judge reading your artifacts will
find it.

---

# PART C — Recommended additions

## C1 🟢 §1.2 — URL ingestion

Your §1.2 promises link-based analysis. It is now implemented, with a security
property worth a sentence.

> Selain penempelan teks, sistem menerima **tautan** lowongan. Halaman diambil dan
> teksnya diekstraksi otomatis. Pengambilan dilindungi terhadap *Server-Side
> Request Forgery*: seluruh alamat IP hasil resolusi DNS diperiksa sebelum koneksi
> dibuka, dan setiap pengalihan divalidasi ulang per-lompatan, sehingga tautan yang
> mengarah ke jaringan internal ditolak. Halaman yang ternyata bukan iklan lowongan
> — misalnya beranda situs setelah pengalihan — dideteksi dan dilaporkan sebagai
> gagal, bukan dinilai.

**Evidence:** `api/fetch_url.py`, `tests/test_fetch_url.py`

## C2 🟢 §3.6 — appeal mechanism

> Sejalan dengan §3.6, sistem menyediakan jalur **sanggahan**: pengguna dapat
> melaporkan hasil yang dinilai keliru, khususnya positif palsu terhadap
> perusahaan sah. Laporan disimpan untuk ditinjau manusia dan **tidak** digunakan
> untuk melatih ulang model — pembatasan ini disampaikan eksplisit pada antarmuka
> sebelum laporan dikirim, sekaligus menutup jalur peracunan data yang disebut
> pada Roadmap Tahap 3. Antarmuka juga menyatakan bahwa menganalisis tidak
> menyimpan teks, sedangkan mengirim laporan menyimpannya.

**Evidence:** `api/feedback.py`, `web/src/components/ReportForm.tsx`

## C3 🟢 §3.2 — resolve the language-scope question

The earlier revision list left this pending. **It is now resolved:** you have a
195-item annotated Indonesian evaluation set. No scope limitation is needed —
state the set's existence, size and prevalence, and cite it as the headline
evaluation domain.

## C4 🟢 §4 — reproducibility

> Kode sumber lengkap tersedia pada repositori GitHub, dan bobot model beserta
> kalibrator dan tabel ambang dipublikasikan pada Hugging Face. Menjalankan
> layanan pada mesin baru tidak memerlukan pengunduhan manual: bobot diambil
> otomatis pada penyalaan pertama.

---

# PART D — What does NOT change

The Integrity Score formula, the four-layer architecture as a *description of the
system*, the choice of a multilingual transformer, the XAI approach, the ethics
section, the roadmap, and every citation.

These are refinements from contact with data, not a redesign. A methods section
saying *"we tested our assumption, it failed, and here is what we did instead"*
is stronger than one that never checked.

---

# PART E — Master number table

Every figure you may quote. If it is not here, it is not a result.

| Claim | Value | Source |
|---|---|---|
| EMSCAD size | 17.880 postings, 866 fraud (4,83%) | `data/processed/split_manifest.json` |
| Split sizes | 12.513 / 1.788 / 1.788 / 1.788 | same |
| **Transformer, EMSCAD test, PR-AUC** | **0,7883** | `eval/results.md` §1 |
| Transformer, EMSCAD test, ROC-AUC | 0,9647 | same |
| Transformer, EMSCAD test, precision / recall | 0,9273 / 0,5930 | same |
| Transformer, EMSCAD test, F1 | 0,7234 | same |
| Confusion matrix (TN/FP/FN/TP) | 1698 / 4 / 35 / 51 | same |
| **TF-IDF, EMSCAD test, PR-AUC** | **0,8651** | same |
| Transformer, val PR-AUC | 0,8669 | `artifacts/scam_model/training_summary.json` |
| Transformer, calib PR-AUC | 0,8509 | `eval/results.md` §1 |
| TF-IDF, val PR-AUC | 0,8769 | `eval/baseline_results.md` |
| **Indonesian holdout, PR-AUC** | **0,9258** | `eval/indonesian_results.md` |
| Indonesian holdout size | 195 items, 36,41% scam | same |
| Ablation: model+rules | 0,8617 PR-AUC, 28 FP | same |
| Ablation: rules only | 0,4167 PR-AUC, 93 FP | same |
| ECE raw → calibrated | 0,0234 → 0,0158 | `eval/results.md` §2 |
| Brier raw → calibrated | 0,0250 → 0,0226 | same |
| Latency, score only (p50/p95) | 86 / 99 ms | `eval/results.md` §4 |
| Latency, with explanation (p50/p95) | 469 / 949 ms | same |
| Risk band boundaries | Tinggi <67, Rendah ≥93 | `artifacts/thresholds.json` |
| Real emails in corpus | 0,02% | `eval/derivability_report.md` |
| Class skew in signal availability | 74,8% vs 67,9% | same |
| Word fusion rate | 78,1% of documents | `data/processed/prepare_report.md` |
| Per-rule cap / aggregate cap | 0,10 / 0,45 | `ml/feature_contract.py` |
| Number of rules | 9 | same |
| Occlusion delta, probability vs margin | 0,0039 vs 2,083 | `api/explain.py` |
| Length-only control (final set) | 0,3734 PR-AUC | `eval/indonesian_results.md` |
| Length-only control (rejected draft set) | 0,8847 PR-AUC | dataset repair, §B1.5 |
| Split sizes train/val/calib/test | 12.513 / 1.788 / 1.788 / 1.788 | `data/processed/split_manifest.json` |
| pos_weight | 19,69 | same |

---

# PART F — Claims you must NOT make

Each of these is checkable against the repository in under a minute.

| Do not claim | Because |
|---|---|
| The rule layer improves accuracy | It lowered PR-AUC 0,9258 → 0,8617 and is disabled |
| PR-AUC 0,8669 as generalisation | That is val, where the checkpoint was selected. Test is 0,7883 |
| The transformer beats classical baselines | It does not on EMSCAD test (0,7883 vs 0,8651) |
| ONNX INT8 serving | Not implemented; PyTorch is served |
| Indonesian performance is established | 195 items, thresholds fitted on the same set |
| The system detects ghost jobs | Out of scope, not implemented |
| Reports improve the model | Explicitly not used for retraining |
| Accuracy 97,82% as a headline | Majority class alone scores 95,19% |
| The product is deployed publicly | It is not; it runs locally and in Docker |

---

# PART G — Submission checklist

**Paper**
- [ ] A1–A8 applied (🔴 — factual corrections)
- [ ] A9 applied (🟢)
- [ ] Results section added (B1.1–B1.8)
- [ ] Discussion section added (B2.1–B2.3)
- [ ] Limitations subsection added (B3)
- [ ] Error analysis written from `eval/results.md` §5, not from memory
- [ ] Every number cross-checked against Part E
- [ ] No claim from Part F appears anywhere
- [ ] ≤ 12 pages, NeurIPS 2015 format, PDF
- [ ] Problem and product still consistent with the preliminary paper

**Implementation**
- [ ] GitHub repo link submitted
- [ ] Repo public after the deadline
- [ ] README setup instructions verified on a clean clone
- [ ] Hugging Face model link submitted separately
- [ ] Every feature claimed in the paper is present and testable

**Video (3–5 min)**
- [ ] ≥1 min business framing, ≥2 min working product
- [ ] Real product, not slides or mockups
- [ ] Shows: paste text → score → evidence highlighting → URL analysis → appeal
- [ ] Unlisted YouTube or Google Drive link, accessible to the committee

---

## One closing note

The strongest thing about this submission is not the score. It is that you
measured a component you built, found it harmful, and turned it off — and that
you looked at a held-out split once and reported the lower number it gave you.

Write the paper so that is visible. *Reliability & Error Analysis* and
*Evaluation* together are 10% of the rubric, and this is exactly what they ask
for.
