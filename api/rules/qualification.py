"""Qualification-conflict detection — MVP_PLAN.md step 2.4.

Owns `qualification_conflict`: an ad that simultaneously wants no experience and
several years of it, e.g. the concept paper's example (section 3.1) of "fresh
graduate" alongside "pengalaman 5 tahun".

## Why this rule is bilingual

It is the ONLY feature EMSCAD can teach us (see `eval/derivability_report.md`) —
every other rule depends on contact details the corpus stripped. That is because
this signal lives in the prose itself rather than in metadata.

But EMSCAD is English. If the patterns here were Indonesian-only, the rule would fire
on nothing during EMSCAD evaluation and its one advantage would be lost. So both
languages are matched, and `EMSCAD_DERIVABLE_FEATURES` in the feature contract
depends on that staying true.

## Why severity is capped low

Contradictory requirements are frequently just sloppy copywriting by a real HR team
reusing an old template — not fraud. The rule reports the inconsistency; it does not
claim deception. Its weight (0.02) is correspondingly among the lowest.
"""

from __future__ import annotations

import re

from api.ingest import IngestResult, Match
from api.rules.base import Rule, RuleOutcome
from api.schemas import RuleCategory, Span

CATEGORY = RuleCategory.QUALIFICATION

_LABEL = ("Persyaratan pengalaman saling bertentangan",
          "Experience requirements contradict each other")

#: "no experience needed", both languages.
NO_EXPERIENCE = (
    "tanpa pengalaman", "tidak perlu pengalaman", "tidak memerlukan pengalaman",
    "pengalaman tidak diutamakan", "fresh graduate", "fresh grad", "lulusan baru",
    "belum berpengalaman", "pemula",
    "no experience", "no prior experience", "without experience",
    "experience not required", "entry level", "entry-level", "no experience necessary",
)

#: "N years of experience required", both languages. The captured group is N.
_EXPERIENCE_YEARS = re.compile(
    r"(?:"
    r"pengalaman\s*(?:kerja\s*)?(?:minimal|min\.?|minimum|min)?\s*(\d{1,2})\s*(?:\+\s*)?tahun"
    r"|(\d{1,2})\s*(?:\+\s*)?tahun\s*pengalaman"
    r"|(?:minimum|minimal|at\s+least|min\.?)\s*(\d{1,2})\s*(?:\+\s*)?years?"
    r"|(\d{1,2})\s*(?:\+\s*)?years?\s*(?:of\s*)?(?:relevant\s*|prior\s*|work\s*)?experience"
    r")",
    re.IGNORECASE,
)

#: Years below this are not a real contradiction — "fresh graduate, 1 year of
#: internship experience" is an ordinary and coherent ask.
MIN_CONFLICTING_YEARS = 2

#: A senior title alongside "entry level" is a different (and weaker) signal than a
#: numeric contradiction, so titles alone do not fire this rule.
_NEGATION_WINDOW = 40


def _find_no_experience(text: str) -> Match | None:
    lowered = text.lower()
    for phrase in NO_EXPERIENCE:
        index = lowered.find(phrase)
        if index == -1:
            continue
        # Skip a negated mention: "bukan tanpa pengalaman", "not entry level".
        prefix = lowered[max(0, index - _NEGATION_WINDOW) : index]
        if re.search(r"\b(bukan|tidak|not|non)\s*$", prefix):
            continue
        return Match(text=text[index : index + len(phrase)], start=index, end=index + len(phrase))
    return None


def _find_experience_years(text: str) -> tuple[int, Match] | None:
    best: tuple[int, Match] | None = None
    for match in _EXPERIENCE_YEARS.finditer(text):
        years = next((int(g) for g in match.groups() if g), None)
        if years is None or years < MIN_CONFLICTING_YEARS:
            continue
        candidate = (
            years,
            Match(text=match.group(0), start=match.start(), end=match.end()),
        )
        if best is None or years > best[0]:
            best = candidate
    return best


class QualificationConflictRule(Rule):
    feature_ids = ("qualification_conflict",)

    def evaluate(self, ctx: IngestResult) -> list[RuleOutcome]:
        no_experience = _find_no_experience(ctx.raw_text)
        if no_experience is None:
            return [self._clean("qualification_conflict", *_LABEL, CATEGORY)]

        years_found = _find_experience_years(ctx.raw_text)
        if years_found is None:
            return [self._clean("qualification_conflict", *_LABEL, CATEGORY)]

        years, years_match = years_found

        # Wider gaps are more clearly contradictory: "fresh graduate + 2 years" is
        # careless, "fresh graduate + 8 years" is incoherent.
        severity = min(0.35 + 0.10 * (years - MIN_CONFLICTING_YEARS), 0.85)

        return [
            RuleOutcome(
                feature_id="qualification_conflict",
                severity=round(severity, 4),
                label_id=_LABEL[0],
                label_en=_LABEL[1],
                category=CATEGORY,
                evidence=(
                    f"\"{no_experience.text}\" bertentangan dengan "
                    f"\"{years_match.text}\"."
                ),
                span=Span(start=years_match.start, end=years_match.end),
            )
        ]
