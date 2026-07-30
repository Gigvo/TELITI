"""Ingest layer — concept paper section 3.1, "Lapis ingesti".

Cleans the submitted text and extracts the structured fields the rule layer needs:
emails, URLs, phone numbers, company names, salary strings.

Two things this module is strict about:

1. **Spans address the RAW text.** Every extracted item carries character offsets
   into exactly the string the user submitted, so the frontend can highlight by
   slicing their own input. Normalising first and extracting second would shift
   every offset and land highlights on the wrong words.

2. **Redaction is detected, not ignored.** EMSCAD replaces emails and URLs inside
   its description text with `#EMAIL_<hash>#` / `#URL_<hash>#` placeholders. If we
   ran the rule layer over EMSCAD without noticing, "no email present" would fire
   on nearly every row and the fusion model would learn from an artefact of the
   dataset. `IngestResult.has_redaction_placeholders` is how the rules know the
   difference between "no email" and "email was removed before we saw it".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from api.schemas import ExtractedFields, Span

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

#: EMSCAD anonymisation artefacts, e.g. `#EMAIL_a1b2c3#`, `#URL_deadbeef#`.
REDACTION_RE = re.compile(r"#(?:EMAIL|URL|PHONE)_[0-9a-fA-F]+#")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'\[\]{}|\\^`]+", re.IGNORECASE)

#: Indonesian mobile numbers: +62/62/0 then 8xx, allowing spaces and dashes.
#: Does not match salary strings like "Rp9.000.000" — dots are excluded from the
#: body of the pattern, and Indonesian salaries are dot-separated.
PHONE_RE = re.compile(r"(?<![\d.])(?:\+62|62|0)[\s-]?8[1-9][\d\s-]{6,14}\d(?![\d.])")

#: Indonesian legal entity prefixes. Capturing the following Title-Case words gives
#: a usable company name for the email domain-mismatch check.
#:
#: The separator is `[ \t]+`, NOT `\s+`. `\s` matches newlines, so `\s+` runs the
#: match across a line break and swallows the next line's first capitalised word —
#: "PT Teknologi Nusantara\nKualifikasi" becomes the company name, which then fails
#: to match the company's own email domain. A company name never spans lines.
COMPANY_RE = re.compile(
    r"\b(PT|CV|UD|PD|Perum|Perumda|Yayasan|Koperasi)\.?[ \t]+"
    r"([A-Z][A-Za-z0-9&]*(?:[ \t]+[A-Z][A-Za-z0-9&]*){0,4})"
)

#: Indonesian salary expressions: "Rp9.000.000", "Rp 9 juta", "9jt", "gaji 5-7 juta".
SALARY_RE = re.compile(
    r"(?:rp\.?\s*)?\d{1,3}(?:[.,]\d{3})*(?:\s*(?:-|–|s/d|sampai)\s*"
    r"(?:rp\.?\s*)?\d{1,3}(?:[.,]\d{3})*)?\s*(?:jt|juta|ribu|rb|k)?\b",
    re.IGNORECASE,
)

#: Multi-part public suffixes we care about, so that the registrable label of
#: `karier.teknologinusantara.co.id` is `teknologinusantara`, not `co`.
MULTIPART_SUFFIXES = frozenset(
    {
        "co.id", "or.id", "go.id", "ac.id", "sch.id", "web.id", "my.id", "net.id",
        "biz.id", "desa.id", "ponpes.id",
        "co.uk", "org.uk", "ac.uk", "gov.uk",
        "com.au", "com.sg", "com.my", "co.jp", "com.br",
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """A fragment of the raw text plus where it came from."""

    text: str
    start: int
    end: int

    def to_span(self) -> Span:
        return Span(start=self.start, end=self.end)


@dataclass(frozen=True)
class IngestResult:
    raw_text: str
    lowered: str
    emails: tuple[Match, ...] = ()
    urls: tuple[Match, ...] = ()
    phones: tuple[Match, ...] = ()
    companies: tuple[Match, ...] = ()
    redactions: tuple[Match, ...] = ()
    fields: ExtractedFields = field(default_factory=ExtractedFields)

    @property
    def has_redaction_placeholders(self) -> bool:
        """True when the source corpus stripped contact details before we saw them.

        Rules that inspect contact details must report themselves UNAVAILABLE rather
        than "clean" when this is set — see `api/rules/base.py::RuleOutcome.available`.
        """
        return bool(self.redactions)

    @property
    def has_contact_route(self) -> bool:
        """Any route at all by which a candidate could apply."""
        return bool(self.emails or self.urls or self.phones)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRAILING_PUNCTUATION = ".,;:!?)]}\"'>"


def _strip_trailing_punctuation(match: Match) -> Match:
    """URLs at the end of a sentence swallow the full stop; trim it back off."""
    text = match.text
    end = match.end
    while text and text[-1] in _TRAILING_PUNCTUATION:
        text = text[:-1]
        end -= 1
    return Match(text=text, start=match.start, end=end)


def _overlaps(inner: Match, outer: Iterable[Match]) -> bool:
    return any(o.start <= inner.start and inner.end <= o.end for o in outer)


def _find_all(pattern: re.Pattern[str], text: str) -> list[Match]:
    return [Match(text=m.group(0), start=m.start(), end=m.end()) for m in pattern.finditer(text)]


def registrable_label(domain: str) -> str:
    """The distinctive part of a domain.

    `karier.teknologinusantara.co.id` -> `teknologinusantara`
    `mail.gmail.com`                  -> `gmail`
    """
    domain = domain.lower().strip().rstrip(".")
    parts = domain.split(".")
    if len(parts) < 2:
        return domain
    for suffix in MULTIPART_SUFFIXES:
        if domain.endswith("." + suffix):
            head = domain[: -(len(suffix) + 1)].split(".")
            return head[-1] if head else domain
    return parts[-2]


def email_domain(address: str) -> str:
    return address.rsplit("@", 1)[-1].lower() if "@" in address else ""


def url_host(url: str) -> str:
    stripped = re.sub(r"^https?://", "", url.strip(), flags=re.IGNORECASE)
    stripped = stripped.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    return stripped.split("@")[-1].split(":")[0].lower()


def normalize_for_match(value: str) -> str:
    """Reduce a name to comparable characters: 'PT Maju Jaya!' -> 'majujaya'."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def acronym_of(value: str) -> str:
    """'Bank Central Asia' -> 'bca'. Lets bca.co.id match its full company name."""
    tokens = [t for t in re.split(r"\s+", value.strip()) if t]
    return "".join(t[0] for t in tokens if t and t[0].isalpha()).lower()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def ingest(raw_text: str) -> IngestResult:
    """Extract everything the rule layer needs from one raw job ad."""
    redactions = tuple(_find_all(REDACTION_RE, raw_text))
    urls = tuple(_strip_trailing_punctuation(m) for m in _find_all(URL_RE, raw_text))

    # An email inside a URL (`https://site.com/contact@x`) is not a contact address.
    emails = tuple(m for m in _find_all(EMAIL_RE, raw_text) if not _overlaps(m, urls))

    phones = tuple(
        m for m in _find_all(PHONE_RE, raw_text) if not _overlaps(m, urls) and not _overlaps(m, emails)
    )

    companies = tuple(
        Match(text=m.group(0).strip(), start=m.start(), end=m.start() + len(m.group(0).strip()))
        for m in COMPANY_RE.finditer(raw_text)
    )

    lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]

    fields = ExtractedFields(
        title=lines[0][:120] if lines else None,
        company=companies[0].text if companies else None,
        emails=[m.text for m in emails],
        phones=[m.text for m in phones],
        urls=[m.text for m in urls],
    )

    return IngestResult(
        raw_text=raw_text,
        lowered=raw_text.lower(),
        emails=emails,
        urls=urls,
        phones=phones,
        companies=companies,
        redactions=redactions,
        fields=fields,
    )
