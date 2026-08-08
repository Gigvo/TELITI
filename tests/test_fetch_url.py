"""URL ingestion — concept paper §1.2 / §3.1.

The SSRF tests matter most. A server that fetches user-supplied URLs is a way to
reach anything the server can reach, and "we validate the URL" is only true if
something proves it.
"""

from __future__ import annotations

import pytest

from api.fetch_url import (
    MIN_EXTRACTED_CHARS,
    UrlFetchError,
    assert_safe_url,
    extract_text,
    looks_like_job_ad,
)

ENDPOINT = "/api/v1/analyze"


# ===========================================================================
# SSRF protection
# ===========================================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/admin",
        "http://127.0.0.1/",
        "http://127.0.0.1:22/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://192.168.1.1/",
        "http://10.0.0.5/",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud credentials
    ],
)
def test_private_and_loopback_addresses_are_blocked(url):
    """The whole point of the guard.

    169.254.169.254 is the one to care about most: on AWS/GCP it serves instance
    credentials to anything that asks.
    """
    with pytest.raises(UrlFetchError) as exc:
        assert_safe_url(url)
    assert exc.value.reason == "blocked_host"


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/",
     "data:text/html,<h1>hi", "javascript:alert(1)"],
)
def test_non_http_schemes_are_rejected(url):
    with pytest.raises(UrlFetchError) as exc:
        assert_safe_url(url)
    assert exc.value.reason in ("scheme", "no_host")


def test_empty_url_is_rejected():
    with pytest.raises(UrlFetchError) as exc:
        assert_safe_url("   ")
    assert exc.value.reason == "empty"


def test_url_without_a_host_is_rejected():
    with pytest.raises(UrlFetchError) as exc:
        assert_safe_url("http:///path-only")
    assert exc.value.reason == "no_host"


def test_hostname_resolving_to_loopback_is_blocked():
    """Checking the STRING is not enough.

    A hostname the attacker controls can have an A record pointing at 127.0.0.1, so
    the guard has to resolve and inspect the actual addresses.
    """
    with pytest.raises(UrlFetchError) as exc:
        assert_safe_url("http://localtest.me/")  # public DNS name -> 127.0.0.1
    assert exc.value.reason == "blocked_host"


def test_public_url_passes_validation():
    assert assert_safe_url("https://www.example.com/jobs/123").startswith("https://")


# ===========================================================================
# Extraction
# ===========================================================================


def test_block_tags_become_newlines_not_nothing():
    """Stripping tags without separators is what fused words together in EMSCAD
    ("Research InstituteOur passion") — the same mistake, made upstream, cost a whole
    cleaning module."""
    text = extract_text("<p>Software Engineer</p><p>Jakarta</p>")
    assert "Software Engineer" in text
    assert "Jakarta" in text
    assert "EngineerJakarta" not in text


def test_scripts_and_navigation_are_removed():
    html = (
        "<nav>Home About Contact</nav>"
        "<script>var tracking = 1;</script>"
        "<style>.a{color:red}</style>"
        "<article><p>Dibutuhkan admin, kualifikasi S1, gaji Rp5.000.000</p></article>"
    )
    text = extract_text(html)
    assert "tracking" not in text
    assert "color:red" not in text
    assert "kualifikasi" in text


def test_article_content_is_preferred_over_the_whole_page():
    html = (
        "<div>Site-wide boilerplate about lowongan kerja Jakarta Timur statistics</div>"
        "<article><p>Dibutuhkan Staff Admin. Kualifikasi: S1. "
        "Lamaran ke hrd@contoh.co.id</p></article>"
    )
    assert "boilerplate" not in extract_text(html)


def test_html_entities_are_decoded():
    assert "PT Maju & Jaya" in extract_text("<p>PT Maju &amp; Jaya</p>")


# ===========================================================================
# The quality gate
# ===========================================================================


def test_real_job_ad_passes_the_gate():
    text = (
        "Dibutuhkan Staff Admin untuk PT Contoh Sejahtera di Jakarta Selatan.\n"
        "Kualifikasi: S1 semua jurusan, pengalaman minimal 1 tahun, teliti dan "
        "komunikatif.\nTanggung jawab: mengelola dokumen, input data, koordinasi tim.\n"
        "Gaji Rp5.000.000 - Rp7.000.000 per bulan.\n"
        "Kirim lamaran ke rekrutmen@contohsejahtera.co.id sebelum 30 Juni."
    )
    assert looks_like_job_ad(text)[0]


def test_navigation_page_is_refused():
    """The failure this gate exists for: scoring a menu as if it were an ad."""
    nav = "\n".join(["Home", "About", "Contact", "Login", "Register", "Jobs"] * 25)
    ok, reason = looks_like_job_ad(nav)
    assert not ok
    assert reason in ("looks_like_navigation", "no_job_ad_markers")


def test_short_page_is_refused():
    ok, reason = looks_like_job_ad("Lowongan kerja")
    assert not ok
    assert reason == "too_short"


def test_page_with_no_job_vocabulary_is_refused():
    """A news article about something else should not be scored as a job ad."""
    article = (
        "Pemerintah mengumumkan kebijakan baru mengenai transportasi umum di ibu kota. "
        "Kebijakan ini akan mulai berlaku bulan depan dan mencakup beberapa wilayah. "
    ) * 4
    ok, reason = looks_like_job_ad(article)
    assert not ok
    assert reason == "no_job_ad_markers"


def test_gate_threshold_is_meaningful():
    assert MIN_EXTRACTED_CHARS >= 100


# ===========================================================================
# API integration
# ===========================================================================


def test_request_requires_exactly_one_input(client):
    """Accepting both and quietly ignoring the URL would mean a user who pasted a
    link AND text gets a score for the text while believing the link was checked."""
    both = client.post(ENDPOINT, json={"text": "x" * 100, "url": "https://example.com"})
    neither = client.post(ENDPOINT, json={})
    assert both.status_code == 422
    assert neither.status_code == 422


def test_ssrf_attempt_through_the_api_is_refused(client):
    response = client.post(ENDPOINT, json={"url": "http://169.254.169.254/latest/meta-data/"})
    assert response.status_code == 422
    # The message must not confirm whether that internal host exists.
    assert "meta-data" not in response.json()["detail"]


def test_unfetchable_url_returns_a_useful_message(client):
    response = client.post(ENDPOINT, json={"url": "https://this-domain-does-not-exist-x7q2.invalid/"})
    assert response.status_code == 422
    assert "could not be reached" in response.json()["detail"].lower()


def test_text_path_still_works_unchanged(client, scam_text):
    """URL support is additive — the primary path must be untouched."""
    body = client.post(ENDPOINT, json={"text": scam_text}).json()
    assert body["source_url"] is None
    assert body["integrity_score"] >= 0


def test_fetched_page_is_scored_and_reports_its_source(client, monkeypatch):
    """The happy path, with the network stubbed so the test is deterministic."""
    import api.main as main
    from api.fetch_url import FetchedPage

    ad = (
        "Dibutuhkan Staff Admin untuk PT Contoh Sejahtera di Jakarta Selatan.\n"
        "Kualifikasi: S1 semua jurusan, pengalaman minimal 1 tahun.\n"
        "Gaji Rp5.000.000 per bulan.\n"
        "Kirim lamaran ke rekrutmen@contohsejahtera.co.id"
    )
    monkeypatch.setattr(
        main, "fetch_job_ad",
        lambda url: FetchedPage(url=url, final_url=url + "?ref=1", text=ad,
                                html_bytes=1000, redirects=0),
    )
    body = client.post(ENDPOINT, json={"url": "https://example.com/jobs/1"}).json()
    assert body["source_url"] == "https://example.com/jobs/1?ref=1"
    assert body["analysed_text"].startswith("Dibutuhkan Staff Admin")
    assert 0 <= body["integrity_score"] <= 100


def test_listing_page_is_refused_despite_job_vocabulary():
    """A job board homepage is full of job words, so the vocabulary check alone
    passes it. Observed for real: an expired posting redirected to the site root,
    extraction returned 34,215 characters of portal page, and the model scored it
    91/100 — a confident number about a page containing no advertisement."""
    listing = ("Lowongan kerja Jakarta terbaru. Loker admin, loker driver, "
               "loker kasir, gaji menarik, kualifikasi lengkap. ") * 200
    ok, reason = looks_like_job_ad(listing)
    assert not ok
    assert reason == "too_long_for_one_ad"


def test_deep_link_redirecting_to_site_root_is_refused():
    from api.fetch_url import _redirected_to_site_root

    assert _redirected_to_site_root("https://x.com/lowongan/admin-123", "https://x.com/")
    assert _redirected_to_site_root("https://x.com/jobs/1", "https://x.com")
    # A homepage that stays a homepage is not a redirect failure.
    assert not _redirected_to_site_root("https://x.com/", "https://x.com/")
    # A deep link that lands on another deep link is fine.
    assert not _redirected_to_site_root("https://x.com/jobs/1", "https://x.com/jobs/1-detail")
