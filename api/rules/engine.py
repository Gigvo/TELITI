"""Rule engine — MVP_PLAN.md step 1.4.

Runs every registered rule and produces two things:

1. A **feature vector** ordered by `ml.feature_contract.RULE_FEATURE_ORDER`, which
   the fusion meta-model consumes positionally (step 3.1).
2. A list of **RuleHit** objects for the API response.

Both come from the same evaluation, so what the user is shown and what the model
scored are guaranteed to be the same thing. Explaining one number while computing a
different one is the classic way an "explainable" system stops being explainable.

## The positional hazard

The fusion model is a `LogisticRegression` over a plain array. If serving builds that
array in a different order than training did, every coefficient lands on the wrong
feature. Nothing raises. The scores are simply wrong, plausibly wrong, and wrong in a
way that unit tests of either half will pass right through. `_ordered_vector` is
built from `RULE_FEATURE_ORDER` itself and checked by `assert_rule_vector`, so there
is one ordering and it is asserted rather than assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from api.ingest import IngestResult
from api.locale import Locale
from api.rules.base import Rule, RuleOutcome
from api.rules.contact_channel import ContactChannelRule
from api.rules.email_domain import EmailDomainRule
from api.rules.qualification import QualificationConflictRule
from api.rules.risk_phrases import RiskPhraseRule
from api.rules.salary import SalarySanityRule
from api.schemas import RuleHit
from ml.feature_contract import (
    RULE_FEATURE_ORDER,
    FeatureContractViolation,
    assert_rule_vector,
)

#: PLACEHOLDER weights, in score points, used only until the fusion model is
#: trained (step 3.1). After that, contributions come from the learned coefficients
#: and the bounded penalties, and this table is deleted. It exists so the API can
#: return non-zero contributions today and the frontend has realistic values to lay
#: out against. These numbers are NOT evidence of anything.
PLACEHOLDER_CONTRIBUTION_POINTS: dict[str, float] = {
    "email_free_provider": 12.0,
    "email_absent": 8.0,
    "email_domain_mismatch": 14.0,
    "contact_messaging_only": 16.0,
    "url_shortener": 10.0,
    "qualification_conflict": 12.0,
    "salary_implausible_vs_umk": 15.0,
    "risk_phrase_score_id": 15.0,
    "payment_request_id": 18.0,
}


@dataclass(frozen=True)
class RuleEvaluation:
    """Result of running every rule over one job ad."""

    outcomes: dict[str, RuleOutcome]

    def feature_vector(self) -> np.ndarray:
        """Severities ordered by `RULE_FEATURE_ORDER`, for the fusion model.

        Features with no rule implemented yet, and features whose signal could not
        be assessed, both read 0.0 here. Those are NOT the same thing — use
        `availability_mask` to tell them apart. Training code must consult it.
        """
        assert_rule_vector(RULE_FEATURE_ORDER)
        return np.array(
            [self._severity_of(fid) for fid in RULE_FEATURE_ORDER], dtype=np.float64
        )

    def availability_mask(self) -> np.ndarray:
        """True where the signal was actually assessed.

        False means "not assessed" — because the corpus redacted the evidence, or
        because no rule fills that slot yet. `ml/train_fusion.py` must treat those
        entries as missing data, never as clean. See `api/rules/base.py`.
        """
        return np.array(
            [self.outcomes[fid].available if fid in self.outcomes else False
             for fid in RULE_FEATURE_ORDER],
            dtype=bool,
        )

    def to_rule_hits(self, contributions: dict[str, float] | None = None) -> list[RuleHit]:
        """Fired rules only, as API objects, strongest first.

        When the rule layer is advisory (`api/scoring.py::RULE_LAYER_ENABLED`), every
        contribution is reported as 0.0 — the findings are real, but they did not
        move the score, and the response must not imply otherwise.
        """
        from api.scoring import RULE_LAYER_ENABLED  # local import: avoids a cycle

        if not RULE_LAYER_ENABLED:
            weights = {name: 0.0 for name in RULE_FEATURE_ORDER}
        elif contributions is not None:
            weights = contributions
        else:
            weights = PLACEHOLDER_CONTRIBUTION_POINTS

        hits = [
            RuleHit(
                rule_id=outcome.feature_id,
                category=outcome.category,
                label_id=outcome.label_id,
                label_en=outcome.label_en,
                severity=round(outcome.severity, 4),
                contribution=round(outcome.severity * weights.get(outcome.feature_id, 0.0), 2),
                evidence=outcome.evidence,
                span=outcome.span,
            )
            for outcome in self.outcomes.values()
            if outcome.fired
        ]
        # Severity is the tiebreak: with contributions zeroed in advisory mode, it is
        # the only thing left that ranks the findings by how notable they are.
        hits.sort(key=lambda h: (h.contribution, h.severity), reverse=True)
        return hits

    @property
    def unavailable_features(self) -> tuple[str, ...]:
        return tuple(
            fid for fid in RULE_FEATURE_ORDER
            if fid in self.outcomes and not self.outcomes[fid].available
        )

    def _severity_of(self, feature_id: str) -> float:
        outcome = self.outcomes.get(feature_id)
        return outcome.severity if outcome is not None else 0.0


class RuleEngine:
    """Holds the registered rules and validates their coverage of the feature vector."""

    def __init__(self, rules: Sequence[Rule]) -> None:
        self._rules = tuple(rules)
        self._validate()

    def _validate(self) -> None:
        seen: dict[str, str] = {}
        for rule in self._rules:
            if not rule.feature_ids:
                raise FeatureContractViolation(
                    f"{type(rule).__name__} declares no feature_ids."
                )
            for feature_id in rule.feature_ids:
                if feature_id not in RULE_FEATURE_ORDER:
                    raise FeatureContractViolation(
                        f"{type(rule).__name__} owns unknown feature {feature_id!r}. "
                        f"Add it to RULE_FEATURE_ORDER first — the fusion model "
                        f"indexes that tuple positionally."
                    )
                if feature_id in seen:
                    raise FeatureContractViolation(
                        f"Feature {feature_id!r} is owned by both {seen[feature_id]} "
                        f"and {type(rule).__name__}. Two rules writing one slot means "
                        f"one silently overwrites the other."
                    )
                seen[feature_id] = type(rule).__name__

    @property
    def implemented_features(self) -> tuple[str, ...]:
        return tuple(fid for rule in self._rules for fid in rule.feature_ids)

    @property
    def pending_features(self) -> tuple[str, ...]:
        """Slots with no rule yet. Empty is the completion check for step 2.4."""
        implemented = set(self.implemented_features)
        return tuple(fid for fid in RULE_FEATURE_ORDER if fid not in implemented)

    def evaluate(self, ctx: IngestResult) -> RuleEvaluation:
        outcomes: dict[str, RuleOutcome] = {}
        for rule in self._rules:
            produced = rule.evaluate(ctx)
            returned = {o.feature_id for o in produced}
            if returned != set(rule.feature_ids):
                raise FeatureContractViolation(
                    f"{type(rule).__name__} declares {rule.feature_ids} but returned "
                    f"{sorted(returned)}. A rule must return exactly one outcome per "
                    f"declared feature, so that a missing slot is never mistaken for "
                    f"a clean one."
                )
            for outcome in produced:
                outcomes[outcome.feature_id] = outcome
        return RuleEvaluation(outcomes=outcomes)


def default_engine(locale: "Locale | None" = None) -> RuleEngine:
    """The complete rule set (step 2.4).

    All nine slots of RULE_FEATURE_ORDER are filled, so `pending_features` is empty —
    that is the completion check, asserted in tests/test_rules.py.

    `locale` pins every locale-aware rule to one language. Leave it None for
    per-request auto-detection, which is what the API does. Pinning is useful for
    evaluation, where the corpus language is known and detection would add noise.
    """
    return RuleEngine(
        [
            EmailDomainRule(),
            ContactChannelRule(),
            QualificationConflictRule(),
            SalarySanityRule(locale),
            RiskPhraseRule(locale),
        ]
    )
