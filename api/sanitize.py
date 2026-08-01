"""Input sanitisation — MVP_PLAN.md step 4.3.

Runs before ingestion, on exactly the text the user submitted.

## The offset constraint

Spans in the response index the submitted text, and the frontend slices the user's
own string with them (`api/schemas.py::RuleHit`). So sanitisation must be
**length-preserving**: every removed character is replaced by a space rather than
deleted. Dropping characters would shift every subsequent offset and land highlights
on the wrong words — the exact failure `tests/test_api.py` guards against.

The sanitised text is what gets scored AND what the client should render. It is
returned as `analysed_text` so the two can never diverge.

## What is removed and why

- **C0/C1 control codes** corrupt extracted fields and log output.
- **Bidirectional overrides** (U+202A–U+202E, U+2066–U+2069) visually reorder text.
  An advertisement can be made to render one way in the browser while we score
  something else — a spoofing vector against the product's whole purpose.
- **Zero-width characters** are used to split risk phrases so a lexicon lookup
  fails: "b​iaya administrasi" reads normally and matches nothing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from api.constants import CONTROL_CHAR_PATTERN, MIN_MEANINGFUL_CHARS

_CONTROL_RE = re.compile(CONTROL_CHAR_PATTERN)

#: A run of this many identical characters is padding or an attack, not content.
_RUN_LIMIT = 200
_LONG_RUN_RE = re.compile(r"(.)\1{%d,}" % (_RUN_LIMIT - 1), re.DOTALL)


@dataclass(frozen=True)
class SanitizeResult:
    text: str
    removed_control_chars: int
    meaningful_chars: int
    truncated_runs: int

    @property
    def is_analysable(self) -> bool:
        """False when there is not enough real content to say anything about."""
        return self.meaningful_chars >= MIN_MEANINGFUL_CHARS


def count_meaningful_chars(text: str) -> int:
    """Letters and digits only.

    Whitespace, punctuation and emoji are excluded: 80 spaces or a wall of emoji
    satisfies a length check while carrying nothing a rule or a model can read.
    """
    return sum(1 for ch in text if ch.isalnum())


def sanitize(text: str) -> SanitizeResult:
    """Normalise submitted text without changing its length."""
    # NFC first: composed and decomposed forms of the same string would otherwise
    # match lexicon entries inconsistently. NFC is the only normalisation applied —
    # NFKC would rewrite currency and digit variants, changing what the user wrote.
    normalized = unicodedata.normalize("NFC", text)

    # Length-preserving substitution. See the module docstring.
    cleaned, removed = _CONTROL_RE.subn(" ", normalized)

    # Collapse absurd character runs to bound downstream regex work. Replaced with
    # spaces so offsets still line up.
    truncated = 0

    def _shrink(match: re.Match[str]) -> str:
        nonlocal truncated
        truncated += 1
        run = match.group(0)
        return run[:_RUN_LIMIT] + " " * (len(run) - _RUN_LIMIT)

    cleaned = _LONG_RUN_RE.sub(_shrink, cleaned)

    if len(cleaned) != len(text):  # pragma: no cover - invariant guard
        raise AssertionError(
            f"sanitise changed length {len(text)} -> {len(cleaned)}; span offsets "
            f"would no longer address the submitted text"
        )

    return SanitizeResult(
        text=cleaned,
        removed_control_chars=removed,
        meaningful_chars=count_meaningful_chars(cleaned),
        truncated_runs=truncated,
    )
