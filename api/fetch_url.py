"""URL ingestion — concept paper §1.2 / §3.1 ("menerima teks tempel, tautan ...").

Fetches a job advertisement from a URL so the user can paste a link instead of the
text. Two things make this harder than it looks, and both are handled here rather
than left to the caller.

## 1. SSRF is the real risk, not a theoretical one

The moment a server fetches a user-supplied URL, that URL becomes a way to make the
server talk to things the user cannot reach directly:

    http://localhost:8000/          the API itself
    http://169.254.169.254/         cloud instance metadata, i.e. credentials
    http://192.168.1.1/             anything on the private network
    file:///etc/passwd              local files, if the scheme is not restricted

`assert_safe_url` resolves the hostname to its actual IP addresses and rejects
anything that is not a public unicast address. Resolution matters: a hostname under
the attacker's control can point at 127.0.0.1, so checking the string alone is not
enough. Redirects are followed manually and re-checked at every hop, because a
public URL is free to redirect into private space.

## 2. Extraction is unreliable, so it must be allowed to fail

Measured while assembling the evaluation set: fetching the same jakartakerja.com URL
twice returned 87KB and then 122KB, with the job description present in one response
and absent in the other. Pages also carry heavy SEO boilerplate — paragraphs about
"lowongan kerja Jakarta Timur" that appear on every page of the site.

Scoring that boilerplate would produce a confident number about the wrong text. So
extraction ends with a QUALITY GATE: if what came back does not look like a job
advertisement, the caller is told to paste the text instead. Refusing is a better
answer than a fabricated score, and it is the same principle the rule layer applies
when it reports itself unassessed rather than clean.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from html import unescape
from urllib.parse import urlparse, urlunparse

ALLOWED_SCHEMES = frozenset({"http", "https"})

MAX_BYTES = 2 * 1024 * 1024          # 2 MB — job ads are text; anything larger is not one
FETCH_TIMEOUT_SECONDS = 10
MAX_REDIRECTS = 3

#: Identifies the fetcher honestly. A site that wishes to refuse automated requests
#: should be able to recognise and block it.
USER_AGENT = "TELITI-JobAdChecker/0.1 (academic research; +https://github.com/Gigvo/TELITI)"

#: Minimum extracted characters before the result is worth scoring. Below this we
#: have a navigation bar, a cookie banner, or an error page — not an advertisement.
MIN_EXTRACTED_CHARS = 200

#: Upper bound. A single advertisement is not this long: in the 195-item evaluation
#: set the longest real ad is ~2,500 characters and the median is ~800. Text far
#: above that is a LISTING page carrying many postings, or a portal homepage.
#:
#: This bound exists because of an observed failure, not a hypothetical one. A real
#: posting URL from the evaluation set had expired and redirected to the site root;
#: extraction returned 34,215 characters of homepage, the vocabulary gate passed it
#: (a job board homepage is full of job words), and the model returned a confident
#: 91/100 for a page containing no advertisement at all.
MAX_EXTRACTED_CHARS = 8000

#: Words that a genuine job advertisement almost always contains, in either language.
#: Used as the quality gate: page text with none of these is not a job ad, whatever
#: else it is.
JOB_AD_MARKERS = (
    # Indonesian
    "lowongan", "loker", "pekerjaan", "kualifikasi", "persyaratan", "pengalaman",
    "gaji", "lamaran", "melamar", "posisi", "dibutuhkan", "karier", "karir",
    "tanggung jawab", "penempatan", "rekrutmen",
    # English
    "job", "vacancy", "hiring", "position", "requirements", "qualifications",
    "responsibilities", "salary", "apply", "candidate", "experience", "role",
)


class UrlFetchError(Exception):
    """Fetch or extraction failed in a way the user should be told about."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        #: Machine-readable, so the API can map it to a status code without parsing
        #: the human message.
        self.reason = reason


@dataclass(frozen=True)
class FetchedPage:
    url: str
    final_url: str
    text: str
    html_bytes: int
    redirects: int


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------


def _is_public_address(host: str) -> bool:
    """True only if every address `host` resolves to is public unicast.

    Every address, not any: a hostname with both a public and a loopback record
    would otherwise slip through depending on which one the HTTP client picked.

    Raises `UrlFetchError(reason="unresolvable")` when DNS fails, so a mistyped
    domain is reported as "not found" rather than "blocked". Both are refused; the
    distinction matters in the logs, where "blocked" should mean somebody actually
    aimed at private space.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UrlFetchError(
            "That page could not be reached — check the address is correct.",
            reason="unresolvable",
        ) from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False

    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            return False
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local     # 169.254.x.x — cloud metadata lives here
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            return False
    return True


def assert_safe_url(url: str) -> str:
    """Validate a URL for server-side fetching. Returns it normalised.

    Raises `UrlFetchError` rather than returning a flag, so a caller cannot forget
    to check the result.
    """
    url = (url or "").strip()
    if not url:
        raise UrlFetchError("No URL was provided.", reason="empty")

    parsed = urlparse(url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UrlFetchError(
            f"Only http and https links can be checked (got {parsed.scheme or 'none'}).",
            reason="scheme",
        )
    if not parsed.hostname:
        raise UrlFetchError("That does not look like a complete web address.", reason="no_host")

    if not _is_public_address(parsed.hostname):
        # Deliberately vague to the user: confirming which internal hosts exist
        # would itself be useful to an attacker. The specifics go to the log.
        raise UrlFetchError(
            "That address cannot be reached from this service.", reason="blocked_host"
        )

    return urlunparse(parsed)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Suppress automatic redirects so each hop can be re-checked for SSRF."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def fetch_html(url: str) -> tuple[str, str, int]:
    """Fetch a page. Returns (html, final_url, redirects_followed).

    Redirects are followed by hand, revalidating the target each time — a public URL
    is perfectly free to redirect to 127.0.0.1, so validating only the URL the user
    typed would leave the hole wide open.
    """
    current = assert_safe_url(url)
    opener = urllib.request.build_opener(_NoRedirects)

    for hop in range(MAX_REDIRECTS + 1):
        request = urllib.request.Request(
            current,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "id,en;q=0.8",
            },
        )
        try:
            with opener.open(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
                status = response.status
                if status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise UrlFetchError("The page redirected without a target.",
                                            reason="bad_redirect")
                    current = assert_safe_url(urllib.parse.urljoin(current, location))
                    continue

                content_type = (response.headers.get("Content-Type") or "").lower()
                if content_type and "html" not in content_type and "text" not in content_type:
                    raise UrlFetchError(
                        "That link is not a web page.", reason="not_html"
                    )

                # Read one byte past the cap so an oversized body is detectable
                # rather than silently truncated into something that looks valid.
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise UrlFetchError(
                        "That page is too large to check.", reason="too_large"
                    )

                charset = "utf-8"
                if "charset=" in content_type:
                    charset = content_type.split("charset=", 1)[1].split(";")[0].strip()
                return raw.decode(charset, errors="replace"), current, hop

        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get("Location")
                if not location:
                    raise UrlFetchError("The page redirected without a target.",
                                        reason="bad_redirect") from exc
                current = assert_safe_url(urllib.parse.urljoin(current, location))
                continue
            raise UrlFetchError(
                f"The page could not be opened (HTTP {exc.code}).", reason="http_error"
            ) from exc
        except urllib.error.URLError as exc:
            raise UrlFetchError(
                "That page could not be reached.", reason="unreachable"
            ) from exc
        except TimeoutError as exc:
            raise UrlFetchError("That page took too long to respond.",
                                reason="timeout") from exc

    raise UrlFetchError("That link redirects too many times.", reason="too_many_redirects")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_DROP_ELEMENTS = re.compile(
    r"(?is)<(script|style|nav|header|footer|aside|form|noscript|svg|iframe)[^>]*>.*?</\1>"
)
_BLOCK_END = re.compile(r"(?i)</(p|div|li|tr|h[1-6]|br|section|article)>|<br\s*/?>")
_TAG = re.compile(r"(?s)<[^>]+>")
_BLANK_LINES = re.compile(r"\n{3,}")


def extract_text(html: str) -> str:
    """Reduce a page to readable text.

    Block-level closing tags become newlines BEFORE tags are stripped. Removing tags
    without inserting separators is what fused words together in the EMSCAD corpus
    ("Research InstituteOur passion") — the same mistake, made upstream, cost us a
    whole cleaning module (see `ml/text_cleaning.py`).
    """
    body = _DROP_ELEMENTS.sub(" ", html)

    main = re.search(r"(?is)<(article|main)[^>]*>(.*?)</\1>", body)
    if main:
        body = main.group(2)

    body = _BLOCK_END.sub("\n", body)
    text = unescape(_TAG.sub(" ", body))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(line for line in lines if line)).strip()


def looks_like_job_ad(text: str) -> tuple[bool, str]:
    """The quality gate. Returns (ok, reason_if_not).

    Extraction is unreliable enough that scoring whatever comes back would regularly
    mean scoring a navigation menu. Refusing is the honest failure.
    """
    if len(text) < MIN_EXTRACTED_CHARS:
        return False, "too_short"

    if len(text) > MAX_EXTRACTED_CHARS:
        # A listing page or portal homepage, not one advertisement. Vocabulary
        # checks cannot catch this: a job board's front page is full of job words.
        return False, "too_long_for_one_ad"

    lowered = text.lower()
    if not any(marker in lowered for marker in JOB_AD_MARKERS):
        return False, "no_job_ad_markers"

    # A page of one-word navigation fragments has many "lines" and few sentences.
    lines = [line for line in text.split("\n") if line]
    if lines and (sum(len(line) for line in lines) / len(lines)) < 15:
        return False, "looks_like_navigation"

    return True, ""


def _redirected_to_site_root(requested: str, final: str) -> bool:
    """True when a deep link ended up at the site's front page.

    Job boards commonly redirect expired postings to their homepage rather than
    returning 404. Following that and scoring what arrives means scoring a listing
    page — observed in practice with a URL from the evaluation set.
    """
    requested_path = urlparse(requested).path.strip("/")
    final_path = urlparse(final).path.strip("/")
    return bool(requested_path) and not final_path


def fetch_job_ad(url: str) -> FetchedPage:
    """Fetch a URL and return usable advertisement text, or raise `UrlFetchError`."""
    html, final_url, redirects = fetch_html(url)

    if _redirected_to_site_root(url, final_url):
        raise UrlFetchError(
            "That posting is no longer available — the link redirects to the site's "
            "home page. Copy the advertisement text and paste it instead.",
            reason="redirected_to_root",
        )

    text = extract_text(html)

    ok, reason = looks_like_job_ad(text)
    if not ok:
        raise UrlFetchError(
            "We could not find a job advertisement on that page. Copy the "
            "advertisement text and paste it instead.",
            reason=reason,
        )

    return FetchedPage(
        url=url, final_url=final_url, text=text,
        html_bytes=len(html), redirects=redirects,
    )
