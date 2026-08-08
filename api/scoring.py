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

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from api.constants import PLACEHOLDER_THRESHOLDS, RISK_RENDAH, RISK_SEDANG, RISK_TINGGI
from api.rules.engine import RuleEvaluation
from api.schemas import RiskLabel
from ml.feature_contract import MAX_TOTAL_RULE_SHIFT, RULE_FEATURE_ORDER
from ml.rule_weights import RULE_WEIGHTS, WEIGHTS_VERSION

#: Whether the rule layer may move the Integrity Score.
#:
#: DISABLED 2026-08-08 on measured evidence. Evaluated on the 195-item Indonesian
#: holdout (`eval/indonesian_results.md`):
#:
#:     model only      PR-AUC 0.9258    5 false positives
#:     model + rules   PR-AUC 0.8617   28 false positives
#:     rules only      PR-AUC 0.4167   (prevalence floor 0.3641)
#:
#: The rule layer made the product measurably worse: it cost 0.064 PR-AUC and more
#: than five times the false positives. The dominant cause is `email_absent`, which
#: fired on 91% of legitimate Indonesian advertisements — real Indonesian job ads
#: give a WhatsApp number rather than an email address, so a design assumption
#: carried over from Western/formal recruitment did not hold.
#:
#: Note on method: two rules (`email_domain_mismatch`, `contact_messaging_only`)
#: showed precision 1.00 on the holdout, but each fired only twice in 195 items.
#: Keeping just those would be selecting rules on the strength of the very set used
#: to report results, which turns the holdout into training data. The whole layer is
#: therefore disabled and the outcome reported as measured.
#:
#: Rules still RUN — their findings are returned as advisory evidence, and they stay
#: under test. Only their effect on the score is removed. Set this back to True to
#: restore the previous behaviour.
RULE_LAYER_ENABLED = False


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
    #: False when the rule layer is advisory only — `rule_shift_applied` is 0 and
    #: `contributions` describes what the rules found, not what moved the score.
    rule_layer_enabled: bool = RULE_LAYER_ENABLED

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
    """Convert probability shifts into score points, for the API response.

    Returns zeros while the rule layer is advisory: a card reading "−9.5 pts" next to
    a score those points did not affect would be a lie told by the interface. The
    finding is still shown; the claim that it moved the number is not.
    """
    if not RULE_LAYER_ENABLED:
        return {name: 0.0 for name in contributions}
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

    if RULE_LAYER_ENABLED:
        # Rules push toward risk only, and never beyond the aggregate ceiling.
        applied_shift = min(max(raw_shift, 0.0), MAX_TOTAL_RULE_SHIFT)
    else:
        # Advisory mode: the rules still report what they found, but the score is
        # the calibrated model probability alone. See RULE_LAYER_ENABLED above for
        # the measurements behind this.
        applied_shift = 0.0

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


THRESHOLDS_PATH = Path(os.environ.get("TELITI_THRESHOLDS", "artifacts/thresholds.json"))


@lru_cache(maxsize=1)
def load_thresholds(path: str = str(THRESHOLDS_PATH)) -> tuple[dict[str, int], bool]:
    """Return (bounds, fitted). `fitted` is False when falling back to placeholders.

    Derived boundaries come from `ml/fit_thresholds.py`, which picks them from a
    precision target on the scam class and a recall target on the safe class rather
    than round numbers — concept paper §3.3 is explicit that they must not be
    arbitrary. The placeholders are exactly the arbitrary choice it warns against, so
    the caller can tell the two apart and `/health` reports which is in use.
    """
    file = Path(path)
    if not file.is_file():
        # Not on disk: try the Hugging Face repo, so a fresh clone gets fitted
        # boundaries rather than the arbitrary placeholders §3.3 warns against.
        from api.artifacts import resolve_file

        resolved = resolve_file(Path(path).name)
        if resolved is None:
            return dict(PLACEHOLDER_THRESHOLDS), False
        file = resolved
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        bounds = {
            "tinggi_below": int(data["tinggi_below"]),
            "rendah_at_or_above": int(data["rendah_at_or_above"]),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # A malformed file must not silently degrade to arbitrary numbers without
        # the caller being able to notice.
        return dict(PLACEHOLDER_THRESHOLDS), False

    if not 0 <= bounds["tinggi_below"] <= bounds["rendah_at_or_above"] <= 100:
        return dict(PLACEHOLDER_THRESHOLDS), False
    return bounds, True


def label_for_score(score: int, thresholds: dict[str, int] | None = None) -> RiskLabel:
    """Map a score to Rendah / Sedang / Tinggi."""
    bounds = thresholds if thresholds is not None else load_thresholds()[0]
    if score < bounds["tinggi_below"]:
        return RiskLabel(RISK_TINGGI)
    if score >= bounds["rendah_at_or_above"]:
        return RiskLabel(RISK_RENDAH)
    return RiskLabel(RISK_SEDANG)
