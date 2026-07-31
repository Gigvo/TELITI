"""Text cleaning for EMSCAD — MVP_PLAN.md step 1.2.

Measured properties of the Kaggle copy of EMSCAD (`fake_job_postings.csv`),
17,880 rows, taken 2026-07-31:

| property                        | rate  |
|---------------------------------|-------|
| HTML tags present               |  0.0% |
| HTML entities (`&amp;`, `&#39;`)| 22.0% |
| double-encoded entities         |  0.006% |
| documents with glued words      | 78.1% |
| U+FFFD replacement characters   | common |

The tags were stripped before publication, but stripped **without inserting a
separator**, so words are fused across former tag boundaries:

    "...Research InstituteOur passion for improving..."
    "...Account ExecutiveAs a member of..."

78.1% of documents are affected, median 6 occurrences each. Left alone, TF-IDF
sees `instituteour` as a distinct out-of-vocabulary term and the transformer wastes
subword pieces on nonsense.

The obvious fix — split on every lowercase→uppercase boundary — damages real
technical vocabulary, which is dense in this corpus: JavaScript appears 917 times,
PowerPoint 699, MySQL 592, PostgreSQL 143. So desegmentation runs against an
allowlist of protected terms.

`CLEANING_VERSION` is recorded in the split manifest. Bump it on any behavioural
change here, so that a model artefact can always be traced to the exact text it saw.
"""

from __future__ import annotations

import html
import re

CLEANING_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Protected vocabulary
# ---------------------------------------------------------------------------

#: CamelCase terms that are genuine words, not artefacts of tag stripping.
#: Longest-first at match time so "JavaScript" wins over a hypothetical "Java".
PROTECTED_CAMELCASE: tuple[str, ...] = (
    # Languages / runtimes / frameworks
    "JavaScript", "TypeScript", "CoffeeScript", "ActionScript", "VBScript",
    "NodeJS", "AngularJS", "BackboneJS", "EmberJS", "ReactJS", "jQuery",
    "ASPNet", "DotNet", "JavaBeans", "JavaServer", "JavaFX",
    # Databases / infra
    "PostgreSQL", "MySQL", "NoSQL", "MongoDB", "SQLite", "MariaDB", "DynamoDB",
    "BigQuery", "GraphQL", "RabbitMQ", "ElasticSearch", "OpenStack", "VMware",
    # Products / platforms
    "PowerPoint", "SharePoint", "QuickBooks", "WordPress", "WooCommerce",
    "SalesForce", "NetSuite", "MailChimp", "HubSpot", "ZenDesk", "PayPal",
    "GitHub", "GitLab", "BitBucket", "StackOverflow", "LinkedIn", "YouTube",
    "FaceBook", "InDesign", "PhotoShop", "AutoCAD", "SolidWorks", "MatLab",
    # Apple / devices
    "iPhone", "iPad", "iPod", "iOS", "iMac", "MacBook", "macOS", "iTunes", "iCloud",
    # Business / misc
    "eCommerce", "eBay", "eLearning", "eMail", "DevOps", "PhD", "MBA", "CPA",
    "JavaScripts", "WiFi", "PowerShell", "TensorFlow", "PyTorch",
)

_PROTECT_SENTINEL = "\x00{}\x00"
_PROTECTED_SORTED = tuple(sorted(PROTECTED_CAMELCASE, key=len, reverse=True))
_PROTECT_PATTERN = re.compile(
    "|".join(re.escape(term) for term in _PROTECTED_SORTED), re.IGNORECASE
)
_PROTECT_INDEX = {term.lower(): i for i, term in enumerate(_PROTECTED_SORTED)}

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

#: Invisible / non-breaking whitespace that survives HTML stripping.
_INVISIBLE = re.compile(r"[\xa0​‌‍﻿⁠]")

#: U+FFFD between a letter and a lowercase letter is almost always a mangled
#: apostrophe: "Esri�s" -> "Esri's". Elsewhere it carries no information.
_MOJIBAKE_APOSTROPHE = re.compile(r"(?<=[A-Za-z])�(?=[a-z])")

#: A leftover HTML tag, in case a future data source is not pre-stripped.
#: Replaced with a SPACE, not "" — that is the mistake that created the glued words.
_HTML_TAG = re.compile(r"<[^>]{1,200}>")

#: lowercase followed by Uppercase-lowercase: the tag-boundary fusion signature.
_GLUED = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])")

#: Digit fused to a letter across a former boundary: "20 yearsExperience".
_GLUED_DIGIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[A-Z][a-z])")

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


#: EMSCAD anonymisation placeholders, e.g. `#EMAIL_a1b2c3d4#`.
#: These MUST survive cleaning intact. They are the only marker telling the rule
#: layer that contact details were stripped before we saw the text, which is what
#: keeps `email_absent` from reporting "clean" on a redacted corpus
#: (see api/rules/base.py). `_GLUED_DIGIT` would otherwise rewrite the hex body as
#: "#EMAIL_a 1b 2c 3d 4#", destroying the marker without any error.
_REDACTION_TOKEN = re.compile(r"#(?:EMAIL|URL|PHONE)_[0-9a-fA-F]+#")
_REDACTION_SENTINEL = "\x00R{}\x00"


def _protect(text: str) -> str:
    return _PROTECT_PATTERN.sub(
        lambda m: _PROTECT_SENTINEL.format(_PROTECT_INDEX[m.group(0).lower()]), text
    )


def _restore(text: str) -> str:
    for index, term in enumerate(_PROTECTED_SORTED):
        text = text.replace(_PROTECT_SENTINEL.format(index), term)
    return text


def desegment(text: str) -> str:
    """Undo word fusion caused by tag stripping, without harming real CamelCase
    or the redaction placeholders."""
    stashed: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        stashed.append(match.group(0))
        return _REDACTION_SENTINEL.format(len(stashed) - 1)

    protected = _REDACTION_TOKEN.sub(_stash, text)
    protected = _protect(protected)
    protected = _GLUED.sub(" ", protected)
    protected = _GLUED_DIGIT.sub(" ", protected)
    protected = _restore(protected)

    for index, token in enumerate(stashed):
        protected = protected.replace(_REDACTION_SENTINEL.format(index), token)
    return protected


def clean_text(text: str, *, apply_desegment: bool = True) -> str:
    """Normalise one EMSCAD text field.

    Order matters: entities are decoded before anything inspects character classes,
    and desegmentation runs after tag removal so it sees the fused boundaries.
    """
    if not text:
        return ""

    # Twice: 0.006% of descriptions are double-encoded ("&amp;#39;").
    text = html.unescape(html.unescape(text))

    # A space, never "" — see the module docstring.
    text = _HTML_TAG.sub(" ", text)

    text = _MOJIBAKE_APOSTROPHE.sub("'", text)
    text = text.replace("�", " ")
    text = _INVISIBLE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if apply_desegment:
        text = desegment(text)

    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def build_document(row: dict, fields: tuple[str, ...], *, apply_desegment: bool = True) -> str:
    """Concatenate the profile's permitted text fields into one document.

    Newline-joined rather than space-joined so that the sentence splitter used by
    the XAI module (step 3.4) has real boundaries to work with.
    """
    parts = []
    for name in fields:
        value = row.get(name)
        if value is None:
            continue
        cleaned = clean_text(str(value), apply_desegment=apply_desegment)
        if cleaned:
            parts.append(cleaned)
    return "\n".join(parts)
