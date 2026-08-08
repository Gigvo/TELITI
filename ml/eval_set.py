"""Loading and validating the Indonesian evaluation set — MVP_PLAN.md step 1.5.

This module is the single gate between annotated data and anything that produces a
number. It exists mainly to enforce one rule:

    SYNTHETIC DATA MUST NEVER PRODUCE A REPORTABLE RESULT.

The framework is built before the real annotations exist, so the pipeline is
developed against fabricated fixtures. That is a normal and sensible way to sequence
the work, and it has one specific failure mode: in a short sprint somebody runs the
pipeline, sees `PR-AUC 0.94`, screenshots it, and the provenance is lost by the time
it reaches a slide. Nobody lies — the number simply outlives the caveat.

Fabricated Indonesian scam text is especially treacherous here, because whoever
writes it encodes their own assumptions about what a scam looks like, and the model
is then evaluated against those same assumptions. The result looks excellent and
measures nothing.

So `EvalSet.is_synthetic` propagates into every artefact and report, and callers that
compute metrics must pass `allow_synthetic=True` explicitly to proceed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

DEFAULT_HOLDOUT = Path("eval/indonesian_holdout.jsonl")
DEFAULT_FIXTURE = Path("eval/synthetic_fixture.jsonl")

#: The dataset is labelled SINGLE-PASS from provenance (the `label` field), not by
#: two independent annotators. `annotator_a` / `annotator_b` / `collected_at` are
#: therefore optional rather than required, and Cohen's kappa is reported as n/a
#: rather than faked — see the note in `ml/validate_eval_set.py`.
#:
#: The trade-off is explicit: provenance labelling trusts the sourcing rather than
#: two humans agreeing. If an item was mis-sourced, nothing here catches it.
REQUIRED_FIELDS = ("id", "text", "label", "source_type", "channel")

#: `source_url` is required for anything found in public — it is what makes an item
#: evidence rather than assertion. But some of the most valuable items are messages a
#: team member received personally on WhatsApp: there is no link, and those are
#: exactly the input the product is built for. For those, provenance must be recorded
#: in `notes` instead. One or the other is mandatory; neither is not acceptable.
PROVENANCE_FIELDS = ("source_url", "notes")

VALID_SOURCE_TYPES = frozenset({
    "bareskrim", "kominfo", "media", "watchdog_account", "community_report",
    "jobstreet", "glints", "kalibrr", "karir_com", "campus_career",
    "company_official", "job_board", "synthetic",
})

VALID_CHANNELS = frozenset({
    "whatsapp", "telegram", "instagram", "facebook", "job_board", "other",
})

MIN_TEXT_LENGTH = 30

#: Banner stamped on any report derived from fabricated data.
SYNTHETIC_BANNER = (
    "> ⚠️ **SYNTHETIC — NOT A RESULT.** These numbers come from fabricated fixtures "
    "written to exercise the pipeline. They measure the assumptions of whoever wrote "
    "the fixtures, not the behaviour of real job ads. Do not cite, screenshot, or "
    "put them in a slide."
)


class EvalSetError(ValueError):
    """Raised when an evaluation file is malformed or used unsafely."""


@dataclass(frozen=True)
class EvalItem:
    id: str
    text: str
    label: int
    source_type: str
    channel: str
    # Optional: personally received messages have no public link, so provenance is
    # carried in `notes` instead. See PROVENANCE_FIELDS above.
    source_url: str | None = None
    # Optional under single-pass provenance labelling — see REQUIRED_FIELDS above.
    annotator_a: str | None = None
    annotator_b: str | None = None
    collected_at: str | None = None
    label_a: int | None = None
    label_b: int | None = None
    resolved_by: str | None = None
    campaign: str | None = None
    notes: str = ""
    synthetic: bool = False


@dataclass(frozen=True)
class EvalSet:
    items: tuple[EvalItem, ...]
    path: Path
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[EvalItem]:
        return iter(self.items)

    @property
    def is_synthetic(self) -> bool:
        """True if ANY item is fabricated.

        Deliberately not a ratio. A file that is 5% synthetic is not 95% trustworthy —
        it is a file whose provenance is mixed, and no number from it is safe to quote.
        """
        return any(item.synthetic for item in self.items)

    @property
    def texts(self) -> list[str]:
        return [item.text for item in self.items]

    @property
    def labels(self) -> list[int]:
        return [item.label for item in self.items]

    @property
    def n_scam(self) -> int:
        return sum(self.labels)

    @property
    def n_legit(self) -> int:
        return len(self.items) - self.n_scam

    def require_real(self, action: str, *, allow_synthetic: bool = False) -> None:
        """Refuse to let fabricated data produce a reportable number."""
        if self.is_synthetic and not allow_synthetic:
            raise EvalSetError(
                f"Refusing to {action}: {self.path} contains synthetic fixtures.\n"
                f"Synthetic data exercises the pipeline; it cannot measure anything.\n"
                f"Pass allow_synthetic=True (CLI: --allow-synthetic) only for a "
                f"plumbing check, and never quote the output."
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_item(raw: dict[str, Any], line_no: int) -> list[str]:
    problems: list[str] = []

    for name in REQUIRED_FIELDS:
        if name not in raw or raw[name] in (None, ""):
            problems.append(f"line {line_no}: missing required field {name!r}")
    if problems:
        return problems

    # Provenance: a public link, or a note explaining where a privately received
    # message came from. An item with neither cannot be traced back to anything.
    if not any(str(raw.get(field) or "").strip() for field in PROVENANCE_FIELDS):
        problems.append(
            f"line {line_no}: needs either 'source_url' (public items) or 'notes' "
            f"describing provenance (personally received messages)"
        )

    if raw["label"] not in (0, 1):
        problems.append(f"line {line_no}: label must be 0 or 1, got {raw['label']!r}")
    if len(str(raw["text"])) < MIN_TEXT_LENGTH:
        problems.append(
            f"line {line_no}: text is {len(str(raw['text']))} chars, "
            f"minimum {MIN_TEXT_LENGTH}"
        )
    if raw["source_type"] not in VALID_SOURCE_TYPES:
        problems.append(f"line {line_no}: unknown source_type {raw['source_type']!r}")
    if raw["channel"] not in VALID_CHANNELS:
        problems.append(f"line {line_no}: unknown channel {raw['channel']!r}")

    is_synthetic = bool(raw.get("synthetic", False))
    if is_synthetic != str(raw["id"]).startswith("SYNTHETIC-"):
        problems.append(
            f"line {line_no}: id {raw['id']!r} and synthetic={is_synthetic} disagree. "
            f"Fabricated items must be named SYNTHETIC-NNNN so provenance is visible "
            f"in every downstream artefact."
        )

    # Independent double-annotation is OPTIONAL under single-pass provenance
    # labelling. When both labels are present they are still checked for
    # consistency, so a partially double-annotated file keeps its guarantees; when
    # they are absent, Cohen's kappa is reported as n/a rather than fabricated.
    if raw.get("label_a") is not None and raw.get("label_b") is not None:
        for name in ("label_a", "label_b"):
            if raw[name] not in (0, 1):
                problems.append(f"line {line_no}: {name} must be 0 or 1, got {raw[name]!r}")
        if raw["label_a"] != raw["label_b"] and not raw.get("resolved_by"):
            problems.append(
                f"line {line_no}: annotators disagree but resolved_by is empty"
            )

    unknown = set(raw) - set(EvalItem.__dataclass_fields__)
    if unknown:
        problems.append(f"line {line_no}: unknown field(s) {sorted(unknown)}")

    return problems


def load_eval_set(path: Path | str, *, strict: bool = True) -> EvalSet:
    """Parse a JSONL evaluation file, validating every row."""
    path = Path(path)
    if not path.exists():
        raise EvalSetError(
            f"{path} not found. Real annotations go in {DEFAULT_HOLDOUT}; "
            f"the fabricated fixture lives at {DEFAULT_FIXTURE}."
        )

    items: list[EvalItem] = []
    problems: list[str] = []
    seen_ids: dict[str, int] = {}
    seen_texts: dict[str, str] = {}
    warnings: list[str] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue

        item_problems = _validate_item(raw, line_no)
        if item_problems:
            problems.extend(item_problems)
            continue

        if raw["id"] in seen_ids:
            problems.append(
                f"line {line_no}: duplicate id {raw['id']!r} (first seen line {seen_ids[raw['id']]})"
            )
            continue
        seen_ids[raw["id"]] = line_no

        # Exact duplicate text inflates every metric: the same ad scored twice looks
        # like two independent successes.
        normalized = " ".join(str(raw["text"]).split()).lower()
        if normalized in seen_texts:
            warnings.append(
                f"line {line_no}: {raw['id']} duplicates the text of "
                f"{seen_texts[normalized]}; drop one or metrics are inflated"
            )
        else:
            seen_texts[normalized] = raw["id"]

        items.append(EvalItem(**raw))

    if problems and strict:
        raise EvalSetError(
            f"{path} failed validation ({len(problems)} problem(s)):\n  - "
            + "\n  - ".join(problems[:40])
            + ("\n  ... (truncated)" if len(problems) > 40 else "")
        )

    return EvalSet(items=tuple(items), path=path, warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# Annotation quality
# ---------------------------------------------------------------------------


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Inter-annotator agreement, corrected for chance.

    Raw agreement is misleading at this class balance: two annotators who both label
    everything "legitimate" agree 100% of the time and have learned nothing. Kappa
    subtracts the agreement expected by chance.

    Convention: <0.40 poor, 0.40-0.60 moderate, 0.60-0.80 substantial, >0.80 almost
    perfect. Below ~0.60 the labelling criteria are not consistent enough for the
    headline metric to mean much, and a reviewer will ask.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must be the same length")
    n = len(labels_a)
    if n == 0:
        return float("nan")

    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    expected = 0.0
    for value in (0, 1):
        p_a = sum(1 for a in labels_a if a == value) / n
        p_b = sum(1 for b in labels_b if b == value) / n
        expected += p_a * p_b

    if expected == 1.0:
        # Both annotators used a single class throughout; kappa is undefined.
        return float("nan")
    return (observed - expected) / (1 - expected)


def annotation_report(eval_set: EvalSet) -> dict[str, Any]:
    real = [i for i in eval_set if not i.synthetic]
    with_both = [i for i in real if i.label_a is not None and i.label_b is not None]

    kappa = (
        cohens_kappa([i.label_a for i in with_both], [i.label_b for i in with_both])
        if with_both
        else float("nan")
    )
    disagreements = sum(1 for i in with_both if i.label_a != i.label_b)

    channels: dict[str, int] = {}
    sources: dict[str, int] = {}
    for item in eval_set:
        channels[item.channel] = channels.get(item.channel, 0) + 1
        sources[item.source_type] = sources.get(item.source_type, 0) + 1

    return {
        "path": str(eval_set.path),
        "is_synthetic": eval_set.is_synthetic,
        "n_items": len(eval_set),
        "n_scam": eval_set.n_scam,
        "n_legit": eval_set.n_legit,
        "scam_rate": eval_set.n_scam / len(eval_set) if len(eval_set) else 0.0,
        "n_double_annotated": len(with_both),
        "n_disagreements": disagreements,
        "cohens_kappa": kappa,
        "channels": dict(sorted(channels.items())),
        "source_types": dict(sorted(sources.items())),
        "warnings": list(eval_set.warnings),
    }
