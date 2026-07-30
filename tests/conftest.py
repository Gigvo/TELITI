"""Shared fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def scam_text() -> str:
    """The canonical scenario from the concept paper, section 3.4.

    This exact ad is the live demo. On Day 3 (gate 3.4) it becomes a hard regression
    test: the top-ranked sentence must be the one a human would pick, and the whole
    request must complete in under a second. It lives here from Day 1 so that every
    layer is developed against the case we will actually be judged on.
    """
    return (
        "LOWONGAN KERJA ADMIN ONLINE\n"
        "Dibutuhkan segera admin online untuk perusahaan ternama.\n"
        "Gaji Rp9.000.000 per bulan, tanpa pengalaman, langsung kerja dari rumah.\n"
        "Kuota terbatas hanya untuk 10 orang pertama!\n"
        "Wajib membayar biaya administrasi sebesar Rp250.000 untuk proses berkas.\n"
        "Interview dilakukan via Telegram.\n"
        "Kirim CV dan foto KTP ke hrd.rekrutmen2024@gmail.com"
    )


@pytest.fixture(scope="session")
def legit_text() -> str:
    """A plausible legitimate ad. Guards against rules that fire on everything."""
    return (
        "Software Engineer (Backend) - PT Teknologi Nusantara\n"
        "Lokasi: Yogyakarta, Indonesia. Tipe: Full-time, hybrid.\n"
        "Kualifikasi: S1 Ilmu Komputer atau setara, pengalaman minimal 2 tahun "
        "membangun layanan backend dengan Python atau Go, memahami PostgreSQL dan "
        "sistem terdistribusi.\n"
        "Tanggung jawab: merancang dan memelihara layanan API internal, melakukan "
        "code review, berkolaborasi dengan tim produk.\n"
        "Rentang gaji: Rp12.000.000 - Rp18.000.000 per bulan sesuai pengalaman.\n"
        "Lamaran dikirim melalui halaman karier resmi kami di "
        "https://karier.teknologinusantara.co.id"
    )
