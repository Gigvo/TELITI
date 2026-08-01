"""Salary plausibility against a minimum wage — MVP_PLAN.md step 2.4, locale-aware.

Owns `salary_implausible_vs_umk`.

The concept paper's worked example (section 3.4) is "admin online, gaji Rp9 juta,
tanpa pengalaman": the salary is not impossible, it is impossible *for that role*.
So this rule compares the advertised figure against the regional minimum wage and
only fires when the multiple is large AND the role is entry-level.

## Locale handling

Both the wage table and the number format come from `api/locale.py`, so the same
logic serves English and Indonesian:

| | Indonesian | English |
|---|---|---|
| currency | `Rp9.000.000`, `9jt`, `9 juta` | `$5,000`, `USD 5000`, `5k` |
| thousands separator | `.` | `,` |
| decimal separator | `,` | `.` |
| wage table | `umk_2025.json` (IDR/month) | `wages_en.json` (USD/month) |

The separator convention is exactly inverted between the two, which is the single
most dangerous detail here: reading `9.000.000` as nine-point-zero, or `$5,000` as
five, produces a rule that either never fires or fires constantly.

If the active locale has no wage table, the rule reports itself UNAVAILABLE rather
than guessing — the same "not assessed" state used for redacted corpora.

## Why it is deliberately hard to trigger

Legitimate high-paying jobs exist. A senior engineer earning 6x minimum wage is
ordinary. Firing on that would flag exactly the employers section 3.6 tells us to
protect.
"""

from __future__ import annotations

import re

from api.ingest import IngestResult, Match
from api.locale import Locale, load_registry
from api.rules.base import Rule, RuleOutcome
from api.schemas import RuleCategory, Span

CATEGORY = RuleCategory.COMPENSATION

_LABEL = ("Gaji yang ditawarkan tidak wajar untuk kualifikasi yang diminta",
          "Advertised salary is implausible for the qualifications requested")

_NO_WAGE_DATA = (
    "Tidak ada data upah minimum untuk bahasa/wilayah ini, kewajaran gaji tidak dinilai."
)

#: Multiples of the regional minimum below which nothing fires.
ENTRY_LEVEL_MULTIPLE = 2.5
GENERAL_MULTIPLE = 6.0

#: Salary quoted per day/hour/week is not a monthly figure.
_PER_DAY = re.compile(r"/\s*(?:hari|day)|per\s+(?:hari|day)|sehari|harian|daily", re.IGNORECASE)
_PER_HOUR = re.compile(r"/\s*(?:jam|h|hr|hour)\b|per\s+(?:jam|hour)|sejam|hourly", re.IGNORECASE)
_PER_WEEK = re.compile(r"/\s*(?:minggu|week|wk)\b|per\s+(?:minggu|week)|mingguan|weekly", re.IGNORECASE)
#: `pa` (per annum) needs a LEADING boundary too. Written as bare `pa\b` it matches
#: inside "per bulan" — the Indonesian for "per month" — and every monthly salary
#: was silently divided by 12, disabling the salary rule entirely.
_PER_YEAR = re.compile(
    r"/\s*(?:tahun|year|yr|annum)\b|per\s+(?:tahun|year|annum)\b|"
    r"\b(?:tahunan|annually|annual|pa|p\.a\.)\b",
    re.IGNORECASE,
)

#: Context marking a number as compensation.
_SALARY_CONTEXT = re.compile(
    r"\b(gaji|salary|upah|penghasilan|pendapatan|honor|thp|take home pay|"
    r"income|bayaran|imbalan|fee|pay|wage|compensation|earn|earning|earnings|"
    r"stipend|remuneration)\b",
    re.IGNORECASE,
)

#: Context marking a number as a COST to the applicant, not compensation.
#: Without this, "administration fee of $50" is read as a salary.
_COST_CONTEXT = re.compile(
    r"\b(biaya|bayar|transfer|deposit|jaminan|denda|potongan|iuran|"
    r"cost|charge|fee\s+of|payment\s+of|pay\s+us|refundable)\b",
    re.IGNORECASE,
)

_CONTEXT_WINDOW = 60

#: Abbreviation multipliers, both locales.
_MULTIPLIERS = {
    "jt": 1_000_000, "juta": 1_000_000, "rb": 1_000, "ribu": 1_000,
    "k": 1_000, "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000,
}


def _amount_pattern(locale: Locale) -> re.Pattern[str]:
    """Number pattern for the locale's separator convention.

    The abbreviated form ("9jt", "5k") is listed FIRST and requires its unit. With
    the unit optional in a single branch, the regex is satisfied by the bare digits
    and stops before consuming "jt" — so "9jt" parses as 9, falls below the amount
    floor, and is silently dropped. Ordering the mandatory-unit branch first forces
    the longer match.
    """
    currency = "|".join(re.escape(t) for t in locale.currency_tokens) or "rp"
    units = "|".join(sorted(_MULTIPLIERS, key=len, reverse=True))

    if locale.dot_is_thousands:
        # Indonesian: 9.000.000 grouped, 4,5 decimal
        grouped = r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?"
        bare = r"\d{1,4}(?:,\d{1,3})?"
    else:
        # English: 5,000.50 grouped, 5.5 decimal
        grouped = r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?"
        bare = r"\d{1,4}(?:\.\d{1,3})?"

    # 1. number + REQUIRED unit ("9jt", "4,5 juta", "5k")
    # 2. grouped number, no unit ("9.000.000", "5,000")
    # 3. currency-prefixed bare number, any length ("$25" for an hourly rate)
    # 4. bare number of 4+ digits ("9000000")
    #
    # Branch 3 must allow short numbers: an hourly wage like "$25 per hour" is a
    # perfectly normal figure that becomes ~$4,333/month once normalised. Requiring
    # 4+ digits everywhere made every hourly rate unmatchable.
    return re.compile(
        rf"(?:(?:{currency})\s*)?(?:{grouped}|{bare})\s*(?:{units})\b"
        rf"|(?:(?:{currency})\s*)?{grouped}"
        rf"|(?:{currency})\s*\d{{1,4}}(?:[.,]\d{{1,2}})?"
        rf"|\d{{4,}}",
        re.IGNORECASE,
    )


#: Unit suffix at the END of a fragment. A leading `\b` fails on "9jt", where there
#: is no word boundary between the digit and the letters — that silently dropped
#: every abbreviated Indonesian salary. Anchoring to the end and requiring only a
#: trailing boundary handles "9jt", "4,5 juta" and "5k" alike.
_UNIT_SUFFIX = re.compile(
    rf"({'|'.join(sorted(_MULTIPLIERS, key=len, reverse=True))})\s*$", re.IGNORECASE
)


def _parse_amount(fragment: str, locale: Locale) -> int | None:
    """Convert one salary expression to a plain integer in the locale's currency."""
    unit_match = _UNIT_SUFFIX.search(fragment.strip())
    number_text = _UNIT_SUFFIX.sub("", fragment.strip())
    number_text = re.sub(r"[^\d.,]", "", number_text).strip(" .,")
    if not number_text:
        return None

    if locale.dot_is_thousands:
        # "9.000.000" -> 9000000 ; "4,5" -> 4.5
        normalized = number_text.replace(".", "").replace(",", ".")
    else:
        # "5,000.50" -> 5000.50
        normalized = number_text.replace(",", "")

    try:
        value = float(normalized)
    except ValueError:
        return None

    if unit_match:
        value *= _MULTIPLIERS[unit_match.group(1).lower()]
    return int(value)


def extract_monthly_salary(text: str, locale: Locale) -> tuple[int, Match] | None:
    """Find the advertised monthly salary, normalised to the locale's currency.

    Returns the LOWEST plausible figure when a range is given: judging an ad by the
    bottom of its own range is the conservative choice, since the top is marketing.
    """
    lowered = text.lower()

    # A salary keyword must appear SOMEWHERE before any number is worth inspecting.
    # Without this, every number-shaped token in a 20,000-character document built a
    # context window and ran two regexes over it — 68 ms, the single slowest rule.
    # One pre-scan makes the common case (no salary mentioned) nearly free.
    if not _SALARY_CONTEXT.search(lowered):
        return None

    pattern = _amount_pattern(locale)
    # Below this, a figure is a fee or a typo rather than a monthly wage.
    floor = 100_000 if locale.dot_is_thousands else 100
    ceiling = 500_000_000 if locale.dot_is_thousands else 100_000

    candidates: list[tuple[int, Match]] = []

    for match in pattern.finditer(text):
        fragment = match.group(0).strip()
        if not re.search(r"\d", fragment):
            continue
        start, end = match.start(), match.end()

        window = lowered[max(0, start - _CONTEXT_WINDOW) : end]
        salary_hit = _SALARY_CONTEXT.search(window)
        cost_hit = _COST_CONTEXT.search(window)

        if not salary_hit:
            continue
        # A cost keyword nearer than the salary keyword means this is a fee.
        if cost_hit and cost_hit.start() > salary_hit.start():
            continue

        amount = _parse_amount(fragment, locale)
        if amount is None:
            continue

        # Normalise to a MONTHLY figure before applying the plausibility bounds.
        # Checking the raw number first would reject "$25 per hour" (a perfectly
        # normal wage, ~$4,333/month) for being below the floor.
        tail = lowered[start : min(len(lowered), end + 25)]
        if _PER_HOUR.search(tail):
            amount = int(amount * 173.33)
        elif _PER_DAY.search(tail):
            amount *= 22
        elif _PER_WEEK.search(tail):
            amount *= 4
        elif _PER_YEAR.search(tail):
            amount = int(amount / 12)

        if not (floor <= amount <= ceiling):
            continue

        candidates.append((amount, Match(text=fragment, start=start, end=end)))

    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[0])


def looks_entry_level(text: str, locale: Locale) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in locale.senior_markers):
        return False
    return any(m in lowered for m in locale.entry_level_markers) or any(
        r in lowered for r in locale.entry_level_roles
    )


def _format_amount(value: int, currency: str) -> str:
    if currency == "IDR":
        return "Rp" + f"{value:,}".replace(",", ".")
    symbol = {"USD": "$", "GBP": "£", "EUR": "€"}.get(currency, currency + " ")
    return f"{symbol}{value:,}"


class SalarySanityRule(Rule):
    feature_ids = ("salary_implausible_vs_umk",)

    def __init__(self, locale: Locale | None = None) -> None:
        self._locale = locale

    def _resolve(self, ctx: IngestResult) -> Locale:
        return self._locale or load_registry().resolve(ctx.raw_text)

    def evaluate(self, ctx: IngestResult) -> list[RuleOutcome]:
        locale = self._resolve(ctx)

        if not locale.has_wages:
            # No wage table for this locale: report unassessed, never "clean".
            # Guessing against a foreign wage table would be worse than silence.
            return [
                self._unavailable(
                    "salary_implausible_vs_umk", *_LABEL, CATEGORY, _NO_WAGE_DATA
                )
            ]

        parsed = extract_monthly_salary(ctx.raw_text, locale)
        if parsed is None:
            # No salary stated, or unparseable. Silence beats a guess.
            return [self._clean("salary_implausible_vs_umk", *_LABEL, CATEGORY)]

        amount, match = parsed
        minimum, region = locale.wages.lookup(ctx.raw_text)
        multiple = amount / minimum if minimum else 0.0

        entry_level = looks_entry_level(ctx.raw_text, locale)
        threshold = ENTRY_LEVEL_MULTIPLE if entry_level else GENERAL_MULTIPLE

        if multiple < threshold:
            return [self._clean("salary_implausible_vs_umk", *_LABEL, CATEGORY)]

        severity = min(0.4 + 0.6 * (multiple - threshold) / threshold, 1.0)
        currency = locale.wages.currency

        # Narrate in the language of the ad — see the equivalent note in
        # api/rules/risk_phrases.py.
        if locale.code == "id":
            role = "tanpa pengalaman/entry level" if entry_level else "posisi ini"
            evidence = (
                f"Gaji {_format_amount(amount, currency)} setara {multiple:.1f}x "
                f"upah minimum {region} ({_format_amount(minimum, currency)}) "
                f"untuk {role}."
            )
        else:
            role = "an entry-level role" if entry_level else "this position"
            evidence = (
                f"Salary of {_format_amount(amount, currency)} is {multiple:.1f}x the "
                f"minimum wage for {region} ({_format_amount(minimum, currency)}) "
                f"for {role}."
            )

        return [
            RuleOutcome(
                feature_id="salary_implausible_vs_umk",
                severity=round(severity, 4),
                label_id=_LABEL[0],
                label_en=_LABEL[1],
                category=CATEGORY,
                evidence=evidence,
                span=Span(start=match.start, end=match.end),
            )
        ]
