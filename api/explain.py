"""Sentence-level explanation by occlusion — MVP_PLAN.md step 3.4.

Answers the question the concept paper poses in §3.1: *kalimat mana yang
mencurigakan* — which sentence is suspicious.

## How it works

Remove one sentence, re-score the remainder, and see how the model's opinion moves:

    delta = p(scam | full text) − p(scam | text without this sentence)

A large positive delta means the advertisement looked markedly safer without that
sentence, so the sentence was carrying the risk. A negative delta means removing it
made things look *worse* — the sentence was evidence of legitimacy.

This is the model's own reasoning, not a guess about it. Whatever the model reacts
to shows up here, including things nobody thought to put in a keyword list.

## Why occlusion rather than LIME

LIME needs 1,000–5,000 perturbed forward passes per explanation — tens of seconds on
CPU, against a one-second budget. Occlusion needs one pass per sentence, batched into
a single call, and it answers the question the paper actually asks. LIME and SHAP
remain the right tools for the offline analysis notebook, where the time budget is
different (MVP_PLAN.md §1.4).

## Cost control

Every sentence costs a forward pass. A long advertisement with fifty sentences would
blow the latency budget, so `MAX_SENTENCES_FOR_OCCLUSION` caps how many are scored;
the longest sentences are kept, because a two-word line rarely carries the signal.
Sentences beyond the cap are reported with `delta = 0.0` rather than silently
dropped — an absent sentence would look like one that was examined and cleared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from api.constants import MAX_SENTENCES_FOR_OCCLUSION, TOP_K_SENTENCE_EVIDENCE
from api.schemas import Polarity, SentenceEvidence, Span

#: Abbreviations whose full stop does NOT end a sentence. Without these, Indonesian
#: job ads split in the wrong places: "min. 2 tahun" becomes two sentences, and the
#: occlusion then measures the effect of removing "min." on its own.
_ABBREVIATIONS = (
    "min", "maks", "max", "no", "hal", "dll", "dsb", "dkk", "yth", "sdr", "sdri",
    "tgl", "jl", "pt", "cv", "ud", "tlp", "telp", "hp", "wa", "a.n", "u.p",
    "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "co", "vs", "etc", "e.g", "i.e",
)

#: A full stop only ends a sentence when it is not part of an abbreviation and not
#: inside a number ("Rp9.000.000").
#:
#: Each guard includes the DOT — `(?<!\bpt\.)`, not `(?<!\bpt)`. Without it the
#: lookbehind inspects the characters before the whitespace rather than before the
#: period, so "PT. Maju Jaya" still split into "Lowongan di PT." and "Maju Jaya."
#: `(?!\d)` keeps thousand separators intact.
_ABBREVIATION_GUARD = "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVIATIONS)

_SENTENCE_BOUNDARY = re.compile(
    rf"(?:{_ABBREVIATION_GUARD}(?<=[.!?])\s+(?!\d))|\n+",
    re.IGNORECASE,
)

#: Below this a "sentence" is a bullet marker or a stray character, not a claim.
MIN_SENTENCE_CHARS = 12


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


def split_sentences(text: str) -> list[Sentence]:
    """Split into sentences, carrying offsets into the ORIGINAL text.

    Offsets matter as much as the split: the frontend highlights by slicing the
    user's own string, so a sentence that cannot be located is a sentence that
    cannot be shown.
    """
    sentences: list[Sentence] = []
    cursor = 0
    for piece in _SENTENCE_BOUNDARY.split(text):
        if not piece:
            continue
        stripped = piece.strip()
        if not stripped:
            continue
        start = text.find(stripped, cursor)
        if start == -1:  # pragma: no cover - defensive
            continue
        sentences.append(Sentence(stripped, start, start + len(stripped)))
        cursor = start + len(stripped)
    return sentences


def _text_without(text: str, sentence: Sentence) -> str:
    """The advertisement with one sentence removed.

    Joined with a space rather than concatenated, so the neighbouring sentences do
    not fuse into a single token — the same mistake that produced "InstituteOur" in
    the EMSCAD corpus (see `ml/text_cleaning.py`).
    """
    remainder = f"{text[: sentence.start]} {text[sentence.end :]}"
    return re.sub(r"[ \t]{2,}", " ", remainder).strip()


def occlusion_evidence(
    text: str,
    base_margin: float,
    margin_batch,
    top_k: int = TOP_K_SENTENCE_EVIDENCE,
    max_sentences: int = MAX_SENTENCES_FOR_OCCLUSION,
) -> list[SentenceEvidence]:
    """Rank sentences by how much the model's verdict depends on each one.

    Works in LOGIT MARGIN space, not probability space. On a blatant scam the model
    outputs p ≈ 0.9999, and removing any single sentence moves that by ~0.001 —
    not because the sentence is unimportant, but because the probability is pinned
    against its ceiling and the remaining sentences are independently damning.
    Measured on the concept paper's own §3.4 example, probability deltas came out as
    +0.0039, +0.0013, +0.0002: a ranking indistinguishable from noise.

    The margin is unbounded, so the same removals produce differences that can
    actually be compared.

    `margin_batch` takes a list of texts and returns their margins — injected rather
    than imported so this module is testable without loading a transformer.
    """
    sentences = split_sentences(text)

    # With one sentence there is nothing to compare against: removing it leaves an
    # empty document, whose score says nothing about the sentence.
    if len(sentences) < 2:
        return []

    scorable = [s for s in sentences if len(s.text) >= MIN_SENTENCE_CHARS]
    if not scorable:
        return []

    # Keep the FIRST n, not the longest.
    #
    # The model truncates at `MAX_LENGTH` tokens, so sentences past that point are
    # never read. Occluding one of them measures nothing: both variants are identical
    # up to the truncation boundary and score the same. Early sentences are also
    # where job ads put the title, the salary promise and the pitch.
    #
    # Selecting the longest sentences instead would spend the budget on verbose
    # late-document boilerplate while skipping a short, decisive line like
    # "Interview via Telegram."
    skipped: list[Sentence] = []
    if len(scorable) > max_sentences:
        scorable, skipped = scorable[:max_sentences], scorable[max_sentences:]

    variants = [_text_without(text, s) for s in scorable]
    without_margins = margin_batch(variants)

    evidence = [
        SentenceEvidence(
            text=sentence.text,
            # Positive: the ad looked safer without this sentence, so it carried risk.
            delta=round(base_margin - without, 4),
            polarity=Polarity.RISK if base_margin - without > 0 else Polarity.SAFE,
            span=Span(start=sentence.start, end=sentence.end),
        )
        for sentence, without in zip(scorable, without_margins)
    ]

    # Not examined, so reported as neutral rather than omitted — an absent sentence
    # would be indistinguishable from one that was checked and found harmless.
    evidence.extend(
        SentenceEvidence(
            text=sentence.text, delta=0.0, polarity=Polarity.SAFE,
            span=Span(start=sentence.start, end=sentence.end),
        )
        for sentence in skipped
    )

    evidence.sort(key=lambda item: abs(item.delta), reverse=True)
    return evidence[:top_k]
