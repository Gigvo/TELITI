"""Rule interface — MVP_PLAN.md step 1.4.

Every rule returns a list of `RuleOutcome`, one per feature it owns. A single rule
module may own several features (the email rule owns three).

## Why outcomes are tri-state

The obvious design gives each rule a boolean `fired`. That is wrong here, and the
reason is specific to our data.

EMSCAD strips emails and URLs out of its description text. If a rule reports
"no email present -> clean" on an EMSCAD row, it is asserting something it cannot
know: the email may well have been there before the corpus was anonymised. Train the
fusion model on that and it learns a property of the *dataset*, not of job scams.

So an outcome is one of three things:

- `available=True,  severity>0`  -> the signal was assessed and it fired
- `available=True,  severity==0` -> the signal was assessed and it is clean
- `available=False`              -> the signal COULD NOT be assessed for this input

`ml/train_fusion.py` must treat the third case as missing data — drop the row for
that feature, or add an explicit indicator column. It must never read it as clean.

## Why severity is continuous

Feature values are severities in [0, 1] rather than 0/1 flags. A logistic-regression
meta-model handles graded inputs fine, and the gradations carry real information: a
WhatsApp-only contact is mildly unusual in Indonesia, a Telegram interview is a
serious signal. Collapsing both to `1` throws that away.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

from api.ingest import IngestResult
from api.schemas import RuleCategory, Span


@dataclass(frozen=True)
class RuleOutcome:
    """The verdict of one rule on one feature."""

    feature_id: str
    severity: float
    label_id: str
    label_en: str
    category: RuleCategory
    evidence: str = ""
    span: Optional[Span] = None
    available: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                f"{self.feature_id}: severity {self.severity} outside [0, 1]. "
                "Severity is a normalised strength, not a score contribution."
            )
        if not self.available and self.severity != 0.0:
            raise ValueError(
                f"{self.feature_id}: an unavailable signal cannot also have a "
                f"severity ({self.severity}). Unavailable means 'not assessed'."
            )

    @property
    def fired(self) -> bool:
        return self.available and self.severity > 0.0


class Rule(ABC):
    """Base class for a deterministic check.

    A class rather than a plain function because later rules carry configuration —
    the salary rule needs the UMK table loaded (step 2.4).
    """

    #: Which slots of `ml.feature_contract.RULE_FEATURE_ORDER` this rule fills.
    #: Validated by the engine at construction time.
    feature_ids: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def evaluate(self, ctx: IngestResult) -> list[RuleOutcome]:
        """Assess `ctx` and return exactly one outcome per id in `feature_ids`."""

    # -- helpers for subclasses ---------------------------------------------

    def _clean(
        self, feature_id: str, label_id: str, label_en: str, category: RuleCategory
    ) -> RuleOutcome:
        return RuleOutcome(
            feature_id=feature_id,
            severity=0.0,
            label_id=label_id,
            label_en=label_en,
            category=category,
        )

    def _unavailable(
        self, feature_id: str, label_id: str, label_en: str, category: RuleCategory, reason: str
    ) -> RuleOutcome:
        return RuleOutcome(
            feature_id=feature_id,
            severity=0.0,
            label_id=label_id,
            label_en=label_en,
            category=category,
            evidence=reason,
            available=False,
        )
