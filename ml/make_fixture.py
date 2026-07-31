"""Generate the synthetic evaluation fixture — MVP_PLAN.md step 1.5.

    python ml/make_fixture.py

Produces `eval/synthetic_fixture.jsonl`: fabricated Indonesian job ads used to
exercise the pipeline while the real annotations are being collected.

## What this is and is not

It is a PLUMBING TEST. Every item is written to trigger a specific rule branch, so
that `train_fusion.py`, the scoring layer and the reports can be built and run
end-to-end before real data exists.

It is NOT a measurement of anything. The ads are written by us, so a model evaluated
on them is being tested against our own assumptions about what a scam looks like.
That produces an excellent-looking number with no external validity.

Deliberately kept to ~24 items. Generating 200 fabricated rows would produce a file
that *looks* like a corpus and invites exactly the confusion this fixture is designed
to avoid. Coverage of rule branches is the goal, not volume.

Every row carries `synthetic: true` and a `SYNTHETIC-NNNN` id; `ml/eval_set.py`
refuses to compute reportable metrics from any file containing them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.eval_set import DEFAULT_FIXTURE  # noqa: E402

COLLECTED = "2026-07-31"

# (text, label, channel, what rule branch this covers)
CASES: list[tuple[str, int, str, str]] = [
    # ---------------- scams: contact-channel branches ----------------
    (
        "LOWONGAN KERJA ADMIN ONLINE\n"
        "Gaji Rp9.000.000 per bulan, tanpa pengalaman, langsung kerja dari rumah.\n"
        "Kuota terbatas! Wajib membayar biaya administrasi sebesar Rp250.000.\n"
        "Interview via Telegram. Kirim CV ke hrd.rekrutmen2024@gmail.com",
        1, "whatsapp", "telegram + free email + payment request (paper 3.4 scenario)",
    ),
    (
        "DIBUTUHKAN SEGERA!! Staff admin data entry\n"
        "Gaji harian Rp350.000 cair setiap hari.\n"
        "Hubungi admin via Telegram @rekrutcepat2024 untuk proses selanjutnya.",
        1, "telegram", "telegram only, no formal route, anonymous recruiter",
    ),
    (
        "Loker admin online WFH\n"
        "Penghasilan tak terbatas, bonus melimpah, tanpa pengalaman.\n"
        "Daftar sekarang https://s.id/loker-cepat2025 sebelum kuota habis!",
        1, "instagram", "url shortener, no email at all",
    ),
    (
        "INFO LOKER TERBARU\n"
        "Posisi: Customer Service. Gaji Rp8.500.000/bulan.\n"
        "Syarat: minimal SMA, tanpa pengalaman, langsung kerja.\n"
        "Chat WA ke 081234567890, wajib transfer uang jaminan Rp500.000.",
        1, "whatsapp", "whatsapp only + payment request + implausible salary",
    ),
    (
        "Rekrutmen PT Sejahtera Abadi\n"
        "Dibutuhkan 50 karyawan baru, gaji Rp12.000.000.\n"
        "Kirim lamaran ke hrd.sejahtera.recruit@mailinator.com",
        1, "other", "disposable email provider",
    ),
    (
        "LOWONGAN PT Maju Bersama\n"
        "Admin keuangan, gaji Rp10.000.000, fresh graduate dipersilakan.\n"
        "Lamaran ke rekrutmen@loker-cepat-indo.com",
        1, "job_board", "corporate-looking domain mismatched to company name",
    ),
    (
        "Dicari karyawan baru untuk posisi packing barang.\n"
        "Gaji Rp7.000.000 per bulan, tanpa pengalaman, tanpa interview.\n"
        "Biaya pelatihan Rp150.000 dikembalikan setelah bekerja.\n"
        "Hubungi kak Rina via WhatsApp.",
        1, "whatsapp", "chat hiring phrase + payment request + anonymous recruiter",
    ),
    (
        "🔥 URGENT!! LOKER ADMIN 🔥\n"
        "Gaji Rp9jt/bulan 💰 kerja dari rumah 🏠 tanpa pengalaman ✅\n"
        "Kuota terbatas!! Chat WA sekarang juga 👉 https://wa.me/6281234567890",
        1, "whatsapp", "emoji-heavy, wa.me link, span integrity under unicode",
    ),
    (
        "Lowongan kerja di luar negeri, gaji 20 juta per bulan.\n"
        "Semua biaya ditanggung perusahaan, tinggal berangkat.\n"
        "Wajib serahkan fotokopi KTP dan KK ke admin via Telegram.",
        1, "telegram", "document harvesting, telegram routing",
    ),
    (
        "Part time online, kerja 2 jam sehari dapat Rp500.000.\n"
        "Cocok untuk mahasiswa dan ibu rumah tangga.\n"
        "Info lengkap hubungi admin, kuota terbatas hari ini saja.",
        1, "instagram", "implausible salary, urgency, no contact route at all",
    ),
    (
        "PT Karya Nusantara membuka lowongan Staff Gudang.\n"
        "Gaji Rp15.000.000, tanpa pengalaman, langsung diterima.\n"
        "Daftar via bit.ly/karyanusantara-loker",
        1, "facebook", "shortener as only route + salary implausible vs UMK",
    ),
    (
        "Loker CS online shop, gaji Rp6.000.000.\n"
        "Dibutuhkan fresh graduate dengan pengalaman minimal 5 tahun.\n"
        "Kirim CV ke adminloker88@yahoo.com",
        1, "other", "qualification conflict + free email",
    ),
    # ---------------- legitimate ----------------
    (
        "Software Engineer (Backend) - PT Teknologi Nusantara\n"
        "Lokasi: Yogyakarta, Indonesia. Tipe: Full-time, hybrid.\n"
        "Kualifikasi: S1 Ilmu Komputer atau setara, pengalaman minimal 2 tahun "
        "membangun layanan backend dengan Python atau Go.\n"
        "Rentang gaji: Rp12.000.000 - Rp18.000.000 per bulan sesuai pengalaman.\n"
        "Lamaran dikirim melalui https://karier.teknologinusantara.co.id",
        0, "job_board", "clean: corporate site, matching domain, plausible salary",
    ),
    (
        "PT Bank Central Asia membuka program Management Trainee 2026.\n"
        "Kualifikasi: S1 semua jurusan, IPK minimal 3.00, usia maksimal 25 tahun.\n"
        "Pendaftaran melalui hrd@bca.co.id atau karir.bca.co.id",
        0, "job_board", "acronym domain match (bca <- Bank Central Asia)",
    ),
    (
        "Lowongan Staff Akuntansi - CV Mitra Sejahtera\n"
        "Kualifikasi: D3/S1 Akuntansi, memahami perpajakan, teliti.\n"
        "Gaji: Rp4.500.000 - Rp6.000.000 sesuai pengalaman.\n"
        "Kirim CV ke recruitment@mitrasejahtera.co.id",
        0, "job_board", "clean SME posting, corporate email",
    ),
    (
        "Career Center UGM - Lowongan Magang Data Analyst\n"
        "Perusahaan: PT Data Cerdas Indonesia.\n"
        "Durasi 6 bulan, uang saku Rp2.500.000 per bulan.\n"
        "Pendaftaran melalui https://ecc.ft.ugm.ac.id/lowongan/12345",
        0, "job_board", "campus career centre, internship stipend below UMK but legitimate",
    ),
    (
        "PT Sinar Mas Agro membuka lowongan Field Supervisor.\n"
        "Penempatan: Kalimantan Tengah. Gaji sesuai standar perusahaan.\n"
        "Kualifikasi: S1 Pertanian, bersedia ditempatkan di lokasi.\n"
        "Lamaran: recruitment@sinarmasagro.co.id",
        0, "job_board", "vague salary but otherwise legitimate — must not fire",
    ),
    (
        "Dibutuhkan Barista untuk Kopi Kenangan Malioboro.\n"
        "Kualifikasi: pengalaman minimal 1 tahun, siap kerja shift.\n"
        "Gaji UMK Yogyakarta + tunjangan.\n"
        "Info lebih lanjut hubungi WhatsApp 081234567890 atau "
        "email hrd@kopikenangan.co.id",
        0, "whatsapp", "WhatsApp PLUS corporate email — chat is convenience, not trap",
    ),
    (
        "Glints - Lowongan UI/UX Designer di PT Kreatif Digital\n"
        "Full-time, Jakarta Selatan. Rp8.000.000 - Rp12.000.000.\n"
        "Kualifikasi: portfolio wajib, pengalaman 2 tahun, menguasai Figma.\n"
        "Lamar melalui https://glints.com/id/opportunities/jobs/12345",
        0, "job_board", "job board aggregator URL, clean",
    ),
    (
        "PT Astra International - Management Development Program\n"
        "Kualifikasi: S1/S2 semua jurusan, IPK min 3.25, fresh graduate dipersilakan.\n"
        "Benefit: gaji kompetitif, asuransi kesehatan, program pengembangan.\n"
        "Pendaftaran: https://career.astra.co.id",
        0, "job_board", "fresh graduate WITHOUT experience conflict — must not fire",
    ),
    (
        "Lowongan Guru Bahasa Inggris - Yayasan Pendidikan Harapan\n"
        "Kualifikasi: S1 Pendidikan Bahasa Inggris, pengalaman mengajar 1 tahun.\n"
        "Gaji: Rp4.000.000 per bulan.\n"
        "Lamaran ke info@yayasanharapan.sch.id",
        0, "job_board", "sch.id multipart suffix, yayasan entity prefix",
    ),
    (
        "Info loker dari Career Center: PT Telkom Indonesia membuka "
        "rekrutmen Network Engineer.\n"
        "Kualifikasi: S1 Teknik Telekomunikasi, CCNA lebih disukai.\n"
        "Semua informasi resmi ada di https://linktr.ee/careercenterugm",
        0, "instagram", "linktree allowlist — must NOT fire url_shortener",
    ),
    (
        "PT Unilever Indonesia - Supply Chain Analyst\n"
        "Lokasi: Cikarang. Kualifikasi: S1 Teknik Industri, pengalaman 2-3 tahun.\n"
        "Lamaran melalui careers@unilever.com sebelum 30 Agustus 2026.",
        0, "job_board", "multinational .com domain, no company suffix in name",
    ),
    (
        "Dicari Admin Media Sosial untuk UMKM Batik Yogya.\n"
        "Kerja part time, 4 jam per hari, gaji Rp2.000.000 per bulan.\n"
        "Kirim portofolio ke batikyogya.umkm@gmail.com",
        0, "instagram", "LEGITIMATE Gmail use by a small business — the false-positive trap",
    ),
]


def main() -> None:
    lines = []
    for index, (text, label, channel, coverage) in enumerate(CASES, start=1):
        lines.append(
            json.dumps(
                {
                    "id": f"SYNTHETIC-{index:04d}",
                    "text": text,
                    "label": label,
                    "source_url": "synthetic://fixture",
                    "source_type": "synthetic",
                    "channel": channel,
                    "annotator_a": "fixture",
                    "annotator_b": "fixture",
                    "collected_at": COLLECTED,
                    "notes": f"COVERAGE: {coverage}",
                    "synthetic": True,
                },
                ensure_ascii=False,
            )
        )

    DEFAULT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_FIXTURE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_scam = sum(1 for _, label, _, _ in CASES if label == 1)
    print(f"wrote {DEFAULT_FIXTURE}")
    print(f"  {len(CASES)} synthetic items ({n_scam} scam, {len(CASES) - n_scam} legitimate)")
    print("\nThis fixture exercises the pipeline. It measures nothing.")


if __name__ == "__main__":
    main()
