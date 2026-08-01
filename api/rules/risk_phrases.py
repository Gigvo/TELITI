"""Indonesian risk-phrase rules — MVP_PLAN.md step 2.4.

Owns two features, driven by `data/reference/risk_phrases_id.yaml`:

- `payment_request_id`     the `payment` group, kept separate because an employer
                           asking a candidate for money is the single most reliable
                           marker in the Indonesian scam reporting the concept paper
                           cites (section 1.1), and deserves its own weight
- `risk_phrase_score_id`   every other group, aggregated

## Saturating aggregation, not a sum

Scoring is `1 - prod(1 - w_i)` over matched phrases: each additional phrase closes
some fraction of the remaining distance to 1.0, so the score rises with corroboration
but never runs away.

A plain sum would let a handful of weak, individually-innocent marketing phrases
("kerja dari rumah", "gaji besar", "segera hubungi") add up to a maximum-severity
verdict on an ordinary small-business ad. That is precisely the false positive
section 3.6 tells us to suppress, and it is the failure mode this shape prevents.

## Group multipliers

Each group carries a multiplier expressing how diagnostic that *category* is.
`data_harvesting` is weighted near full strength because of its link to the TPPO
cases in section 1.1; `mlm_recruitment` is discounted because network marketing is
misrepresentation rather than fraud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from api.ingest import IngestResult
from api.locale import Locale, load_registry
from api.rules.base import Rule, RuleOutcome
from api.schemas import RuleCategory, Span

CATEGORY = RuleCategory.LANGUAGE

LEXICON_PATH = Path("data/reference/risk_phrases_id.yaml")

_NO_LEXICON = (
    "Tidak ada kamus frasa risiko untuk bahasa ini, pola bahasa tidak dinilai."
)

#: The group routed to its own feature rather than the aggregate.
PAYMENT_GROUP = "payment"

_LABEL_PAYMENT = ("Meminta pembayaran atau transfer dari pelamar",
                  "Requests payment or a transfer from the applicant")
_LABEL_AGGREGATE = ("Menggunakan pola bahasa yang lazim pada lowongan penipuan",
                    "Uses language patterns common in fraudulent job ads")


#: Phrasing that flips a cost phrase from a demand into a BENEFIT.
#:
#: "Perusahaan menanggung seluruh biaya pelatihan" — the company covers all training
#: costs — contains "biaya pelatihan" and is the exact opposite of a scam signal.
#: Naive substring matching cannot tell who pays, and firing here would flag good
#: employers for advertising a perk (section 3.6).
#:
#: Checked in the window BEFORE the phrase, where Indonesian puts the payer.
_BENEFIT_CONTEXT = re.compile(
    r"\b(?:"
    # Indonesian
    r"ditanggung|menanggung|tanggung|digratiskan|gratis|bebas|tanpa|"
    r"tidak\s+ada|tidak\s+dipungut|tidak\s+memungut|"
    r"disediakan|diberikan|difasilitasi|dibayarkan\s+perusahaan|"
    r"perusahaan\s+menanggung|kami\s+menanggung|"
    # English
    r"no|never|without|free|complimentary|waived|"
    r"no\s+charge|free\s+of|at\s+no\s+cost|we\s+cover|we\s+pay|"
    r"company\s+covers|company\s+pays|employer\s+pays|fully\s+funded|"
    r"covered\s+by|paid\s+by\s+(?:the\s+)?company|reimbursed|we\s+provide|"
    r"there\s+is\s+no|there's\s+no|you\s+will\s+never\s+be\s+asked"
    r")\b",
    re.IGNORECASE,
)

#: Phrasing AFTER the phrase that also indicates the employer bears the cost:
#: "biaya pelatihan ditanggung perusahaan".
_BENEFIT_SUFFIX = re.compile(
    r"^\s*(?:"
    r"(?:sepenuhnya\s+)?(?:akan\s+)?(?:ditanggung|digratiskan|gratis|"
    r"dibayarkan|disediakan|ditiadakan)"
    r"|(?:is\s+|are\s+|will\s+be\s+)?(?:waived|covered|reimbursed|"
    r"paid\s+by\s+(?:the\s+)?(?:company|employer))"
    r")\b",
    re.IGNORECASE,
)

#: How far back to look for the payer. Indonesian puts it close to the noun phrase.
_BENEFIT_WINDOW = 45

#: Contrastive conjunctions and clause breaks. A benefit stated BEFORE one of these
#: does not cover a demand stated after it:
#:
#:     "Pelatihan gratis, namun pelamar wajib membayar biaya administrasi."
#:      ^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#:      benefit                 still a demand
#:
#: Without this, a scam that opens with a reassuring clause disables the strongest
#: rule in the system — an easy and obvious evasion.
_CONTRAST_BREAK = re.compile(
    r"\b(?:namun|tetapi|tapi|akan\s+tetapi|melainkan|hanya\s+saja|"
    r"but|however)\b|[.;!?\n]"
)


def _is_benefit_framing(text: str, start: int, end: int) -> bool:
    """True when the surrounding text says the EMPLOYER bears this cost."""
    before = text[max(0, start - _BENEFIT_WINDOW) : start]

    # Only consider the clause immediately preceding the phrase: anything before a
    # contrastive conjunction or sentence break belongs to a different claim.
    breaks = list(_CONTRAST_BREAK.finditer(before))
    if breaks:
        before = before[breaks[-1].end() :]

    if _BENEFIT_CONTEXT.search(before):
        return True

    after = text[end : end + _BENEFIT_WINDOW]
    return bool(_BENEFIT_SUFFIX.search(after))


@dataclass(frozen=True)
class PhraseMatch:
    text: str
    weight: float
    group: str
    start: int
    end: int


@lru_cache(maxsize=1)
def load_lexicon(path: str = str(LEXICON_PATH)) -> dict:
    file = Path(path)
    if not file.exists():
        return {"version": "missing", "groups": {}}
    return yaml.safe_load(file.read_text(encoding="utf-8")) or {"groups": {}}


def _compile_lexicon(lexicon: dict[str, Any]) -> tuple[tuple[re.Pattern, float, str], ...]:
    """Word-boundary patterns so 'deposit' does not match inside 'depositor'."""
    compiled: list[tuple[re.Pattern, float, str]] = []
    for group_name, group in (lexicon.get("groups") or {}).items():
        multiplier = float(group.get("weight_multiplier", 1.0))
        for entry in group.get("phrases", []):
            text = str(entry["text"]).strip().lower()
            if not text:
                continue
            weight = min(float(entry.get("weight", 0.5)) * multiplier, 1.0)
            compiled.append(
                (re.compile(rf"(?<!\w){re.escape(text)}(?!\w)", re.IGNORECASE), weight, group_name)
            )
    return tuple(compiled)


@lru_cache(maxsize=8)
def _compiled_for_locale(code: str) -> tuple[tuple[re.Pattern, float, str], ...]:
    locale = load_registry().get(code)
    return _compile_lexicon(locale.lexicon or {})


def find_phrases(text: str, locale: Locale | None = None) -> list[PhraseMatch]:
    """Locate risk phrases, skipping any framed as an employer-paid benefit."""
    if locale is None:
        locale = load_registry().resolve(text)
    matches: list[PhraseMatch] = []
    for pattern, weight, group in _compiled_for_locale(locale.code):
        for found in pattern.finditer(text):
            # "biaya pelatihan ditanggung perusahaan" is a perk, not a demand.
            if _is_benefit_framing(text, found.start(), found.end()):
                continue
            matches.append(
                PhraseMatch(
                    text=found.group(0),
                    weight=weight,
                    group=group,
                    start=found.start(),
                    end=found.end(),
                )
            )
            break  # one match per phrase; repetition is not extra evidence
    return matches


def saturating_score(weights: list[float]) -> float:
    """Combine independent evidence without letting weak signals accumulate freely.

    1 - prod(1 - w). Two 0.5 phrases give 0.75, not 1.0; five 0.2 phrases give 0.67,
    not 1.0. Corroboration raises the score, repetition of weak signals does not
    saturate it.
    """
    remaining = 1.0
    for weight in weights:
        remaining *= 1.0 - max(0.0, min(weight, 1.0))
    return 1.0 - remaining


class RiskPhraseRule(Rule):
    feature_ids = ("payment_request_id", "risk_phrase_score_id")

    def __init__(self, locale: Locale | None = None) -> None:
        self._locale = locale

    def evaluate(self, ctx: IngestResult) -> list[RuleOutcome]:
        locale = self._locale or load_registry().resolve(ctx.raw_text)

        if not locale.has_lexicon:
            # No lexicon for this language: report unassessed, never "clean".
            # A missing lexicon looking like a clean ad is exactly the silent
            # failure api/rules/base.py exists to prevent.
            return [
                self._unavailable("payment_request_id", *_LABEL_PAYMENT, CATEGORY, _NO_LEXICON),
                self._unavailable("risk_phrase_score_id", *_LABEL_AGGREGATE, CATEGORY, _NO_LEXICON),
            ]

        matches = find_phrases(ctx.raw_text, locale)
        payment = [m for m in matches if m.group == PAYMENT_GROUP]
        other = [m for m in matches if m.group != PAYMENT_GROUP]
        return [
            self._payment_outcome(payment, locale),
            self._aggregate_outcome(other, locale),
        ]

    def _payment_outcome(self, matches: list[PhraseMatch], locale: Locale) -> RuleOutcome:
        if not matches:
            return self._clean("payment_request_id", *_LABEL_PAYMENT, CATEGORY)

        strongest = max(matches, key=lambda m: m.weight)
        severity = saturating_score([m.weight for m in matches])
        quoted = ", ".join(f'"{m.text}"' for m in sorted(matches, key=lambda m: -m.weight)[:3])

        # Narrate in the language of the ad. An English posting explained in
        # Indonesian is unreadable to the person who submitted it.
        evidence = (
            f"Ditemukan {quoted} pada teks lowongan."
            if locale.code == "id"
            else f"Found {quoted} in the job posting."
        )

        return RuleOutcome(
            feature_id="payment_request_id",
            severity=round(severity, 4),
            label_id=_LABEL_PAYMENT[0],
            label_en=_LABEL_PAYMENT[1],
            category=CATEGORY,
            evidence=evidence,
            span=Span(start=strongest.start, end=strongest.end),
        )

    def _aggregate_outcome(self, matches: list[PhraseMatch], locale: Locale) -> RuleOutcome:
        if not matches:
            return self._clean("risk_phrase_score_id", *_LABEL_AGGREGATE, CATEGORY)

        strongest = max(matches, key=lambda m: m.weight)
        severity = saturating_score([m.weight for m in matches])
        top = sorted(matches, key=lambda m: -m.weight)[:3]
        quoted = ", ".join(f'"{m.text}"' for m in top)
        remainder = len(matches) - len(top)

        if locale.code == "id":
            extra = f" (+{remainder} lainnya)" if remainder else ""
            evidence = f"Frasa berisiko: {quoted}{extra}."
        else:
            extra = f" (+{remainder} more)" if remainder else ""
            evidence = f"Risk phrases: {quoted}{extra}."

        return RuleOutcome(
            feature_id="risk_phrase_score_id",
            severity=round(severity, 4),
            label_id=_LABEL_AGGREGATE[0],
            label_en=_LABEL_AGGREGATE[1],
            category=CATEGORY,
            evidence=evidence,
            span=Span(start=strongest.start, end=strongest.end),
        )
