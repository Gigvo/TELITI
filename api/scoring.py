"""Integrity Score computation — concept paper section 3.3.

Combines the calibrated text-model probability with the bounded rule contributions
into one score, and assigns the risk label.

    p_final = clip(p_text + sum(severity_i * weight_i), 0, 1)
    S       = round((1 - p_final) * 100)

## Design notes

**Rules only ever push toward risk.** The combined shift is clamped to be
non-negative before it is applied. A clean ad gets no bonus for being clean — its
score is whatever the text model says. This is deliberate: a rule layer that could
*raise* an ad's score would let a well-formatted scam suppress a correct model
warning, and the rules are heuristics, not evidence of legitimacy.

**Unavailable signals contribute nothing.** A rule that could not be assessed (see
`api/rules/base.py`) is skipped entirely rather than treated as clean.

**The aggregate is clipped.** `MAX_TOTAL_RULE_SHIFT` bounds how far the whole rule
layer can move a score, so the text model always retains the majority of the
decision — the guarantee in section 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

from api.constants import PLACEHOLDER_THRESHOLDS, RISK_RENDAH, RISK_SEDANG, RISK_TINGGI
from api.rules.engine import RuleEvaluation
from api.schemas import RiskLabel
from ml.feature_contract import MAX_TOTAL_RULE_SHIFT, RULE_FEATURE_ORDER
from ml.rule_weights import RULE_WEIGHTS, WEIGHTS_VERSION


@dataclass(frozen=True)
class ScoreBreakdown:
    """Every intermediate value, so a score is always explainable."""

    integrity_score: int
    risk_label: RiskLabel
    model_probability: float
    fused_probability: float
    rule_shift_raw: float
    rule_shift_applied: float
    contributions: dict[str, float]
    weights_version: str

    @property
    def was_clipped(self) -> bool:
        return abs(self.rule_shift_raw - self.rule_shift_applied) > 1e-9


def rule_contributions(evaluation: RuleEvaluation) -> dict[str, float]:
    """Probability shift each fired rule contributes, keyed by feature id."""
    contributions: dict[str, float] = {}
    for feature_id in RULE_FEATURE_ORDER:
        outcome = evaluation.outcomes.get(feature_id)
        # `fired` is already False for unavailable signals, so this skips both
        # "assessed and clean" and "could not assess".
        if outcome is None or not outcome.fired:
            continue
        contributions[feature_id] = outcome.severity * RULE_WEIGHTS[feature_id]
    return contributions


def contribution_points(contributions: dict[str, float]) -> dict[str, float]:
    """Convert probability shifts into score points, for the API response."""
    return {name: round(value * 100, 2) for name, value in contributions.items()}


def compute_score(
    model_probability: float,
    evaluation: RuleEvaluation,
    thresholds: dict[str, int] | None = None,
) -> ScoreBreakdown:
    """Fuse the text model and the rule layer into an Integrity Score."""
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError(f"model_probability {model_probability} outside [0, 1]")

    contributions = rule_contributions(evaluation)
    raw_shift = sum(contributions.values())

    # Rules push toward risk only, and never beyond the aggregate ceiling.
    applied_shift = min(max(raw_shift, 0.0), MAX_TOTAL_RULE_SHIFT)

    fused = min(max(model_probability + applied_shift, 0.0), 1.0)
    score = round((1.0 - fused) * 100)

    return ScoreBreakdown(
        integrity_score=score,
        risk_label=label_for_score(score, thresholds),
        model_probability=model_probability,
        fused_probability=fused,
        rule_shift_raw=raw_shift,
        rule_shift_applied=applied_shift,
        contributions=contributions,
        weights_version=WEIGHTS_VERSION,
    )


def label_for_score(score: int, thresholds: dict[str, int] | None = None) -> RiskLabel:
    """Map a score to Rendah / Sedang / Tinggi.

    Thresholds come from `artifacts/thresholds.json`, derived from a precision target
    on the fraud class (step 3.2). The placeholder values are round numbers and are
    NOT a defensible choice — section 3.3 is explicit that they must not be arbitrary.
    """
    bounds = thresholds or PLACEHOLDER_THRESHOLDS
    if score < bounds["tinggi_below"]:
        return RiskLabel(RISK_TINGGI)
    if score >= bounds["rendah_at_or_above"]:
        return RiskLabel(RISK_RENDAH)
    return RiskLabel(RISK_SEDANG)
