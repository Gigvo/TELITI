"""Hand-set rule weights — MVP_PLAN.md step 3.1 (revised).

## Why these are set rather than fitted

The concept paper (§3.3) planned to fit rule weights on an EMSCAD validation
partition. Measurement killed that: EMSCAD strips contact details before
publication, so 8 of 9 rule signals have no support in the corpus, and
`email_absent` is actively unsafe because its availability differs by class
(74.8% real vs 67.9% fraudulent). See `eval/derivability_report.md`.

Fitting them on Indonesian data instead would need a second annotated set of ~200
items on top of the held-out set. That is not affordable inside the deadline.

So the weights below are set **a priori from domain reasoning**, and then
**validated** on the held-out Indonesian set: `ml/validate_rule_weights.py` measures
whether the rule layer improves over the text model alone. If it does not, that gets
reported, and the honest write-up is "the rule layer did not help on our evaluation
set" — not a quietly dropped component.

This is a weaker claim than fitting, and the paper must say so plainly. It is not a
dishonest one: a priori weights validated on held-out data is a normal, publishable
methodology. What would be dishonest is tuning these numbers against the held-out set
until they look good. **Do not do that** — the moment these are adjusted in response
to holdout performance, the holdout is training data and the headline metric is gone.

## How the numbers were chosen

Ordered by how strongly the signal indicates fraud in the Indonesian reporting the
concept paper cites (§1.1), bounded by `PER_RULE_CONTRIBUTION_CAP` so no single rule
can dominate the model's own judgement.

- **payment_request_id (0.10)** — the defining marker. A real employer never asks a
  candidate for money. Bareskrim's cases centre on exactly this.
- **contact_messaging_only (0.08)** — carries the Telegram signal. The rule itself
  already grades WhatsApp (0.35 severity) far below Telegram (0.75), so this weight
  multiplies an already-calibrated value.
- **salary_implausible_vs_umk (0.07)** — "admin, Rp9jt, tanpa pengalaman" is the
  paper's own example. Strong, but legitimate high-paying roles exist.
- **risk_phrase_score_id (0.06)** — aggregate lexical signal, already graded.
- **email_free_provider (0.04)** — real but weak in Indonesia, where small
  businesses legitimately recruit from Gmail.
- **email_domain_mismatch (0.04)** — a genuine impersonation signal, but our
  company-name matching is deliberately generous, so it fires rarely.
- **url_shortener (0.03)** — hiding the destination is suspicious, not damning.
- **qualification_conflict (0.02)** — often just sloppy copywriting by a real HR team.
- **email_absent (0.01)** — weakest. The rule already returns clean when a real
  career page is present, so what remains is mild.

The exact values are less important than their ORDER, because they are set a priori.
The order encodes a falsifiable claim about Indonesian job scams, and
`ml/validate_rule_weights.py` is what tests it.
"""

from __future__ import annotations

from ml.feature_contract import (
    MAX_TOTAL_RULE_SHIFT,
    PER_RULE_CONTRIBUTION_CAP,
    RULE_FEATURE_ORDER,
    FeatureContractViolation,
)

#: Maximum shift in p(scam) each rule may contribute at severity 1.0.
#: Contribution is `severity * weight`, so a rule at severity 0.5 contributes half.
#: These sum to exactly MAX_TOTAL_RULE_SHIFT (0.45). The aggregate is only reachable
#: if all nine rules fire at full severity at once, which does not occur in practice —
#: but the ceiling must hold in the worst case, not the typical one, or the section
#: 3.3 guarantee is conditional on luck.
RULE_WEIGHTS: dict[str, float] = {
    "payment_request_id": 0.10,
    "contact_messaging_only": 0.08,
    "salary_implausible_vs_umk": 0.07,
    "risk_phrase_score_id": 0.06,
    "email_free_provider": 0.04,
    "email_domain_mismatch": 0.04,
    "url_shortener": 0.03,
    "qualification_conflict": 0.02,
    "email_absent": 0.01,
}

#: Version stamped into artefacts, so a score can be traced to the weights that
#: produced it. Bump on any change here.
WEIGHTS_VERSION = "handset-1.0.0"


def assert_weights_sane() -> None:
    """Checked at import: the section 3.3 guarantee, and full coverage."""
    missing = set(RULE_FEATURE_ORDER) - set(RULE_WEIGHTS)
    if missing:
        raise FeatureContractViolation(
            f"No weight defined for {sorted(missing)}. Every feature in "
            f"RULE_FEATURE_ORDER needs one, or its severity is silently ignored."
        )
    unknown = set(RULE_WEIGHTS) - set(RULE_FEATURE_ORDER)
    if unknown:
        raise FeatureContractViolation(f"Weights for unknown features: {sorted(unknown)}")

    for name, weight in RULE_WEIGHTS.items():
        if weight < 0.0:
            raise FeatureContractViolation(
                f"{name} has a negative weight. A fired risk rule must never make an "
                f"ad look safer."
            )
        if weight > PER_RULE_CONTRIBUTION_CAP + 1e-9:
            raise FeatureContractViolation(
                f"{name} weight {weight} exceeds PER_RULE_CONTRIBUTION_CAP="
                f"{PER_RULE_CONTRIBUTION_CAP}, letting one deterministic rule "
                f"dominate the model (concept paper 3.3)."
            )

    total = sum(RULE_WEIGHTS.values())
    if total > MAX_TOTAL_RULE_SHIFT + 1e-9:
        raise FeatureContractViolation(
            f"Weights sum to {total:.3f}, exceeding MAX_TOTAL_RULE_SHIFT="
            f"{MAX_TOTAL_RULE_SHIFT}."
        )


def weight_vector() -> list[float]:
    """Weights ordered by RULE_FEATURE_ORDER, for dot-product with a feature vector."""
    return [RULE_WEIGHTS[name] for name in RULE_FEATURE_ORDER]


assert_weights_sane()
