"""Feature-availability contract — see MVP_PLAN.md section 1.1.

THIS IS THE MOST IMPORTANT MODULE IN THE PROJECT.

EMSCAD ships metadata columns that are strongly predictive of fraud but that simply
do not exist when a user pastes a WhatsApp message into TELITI. `has_company_logo`
is the classic example: it is one of the most predictive columns in the dataset, and
it is unobtainable at inference time for our actual product.

If a model learns to lean on those columns, offline metrics look excellent and the
live product is garbage. That failure is silent — nothing crashes, the numbers just
lie. So the contract is enforced by assertions that raise, not by convention.

Both the training code (`ml/`) and the serving code (`api/`) import from here so that
there is exactly one definition of "what the model is allowed to see" and exactly one
definition of the rule-feature vector ordering.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# EMSCAD raw schema
# ---------------------------------------------------------------------------

LABEL_COLUMN = "fraudulent"

#: All 18 columns of `fake_job_postings.csv` (EMSCAD). Used to validate the
#: download in Gate 0.3 — if this doesn't match, you have the wrong file.
EMSCAD_COLUMNS: tuple[str, ...] = (
    "job_id",
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "fraudulent",
)

EXPECTED_ROW_COUNT = 17_880
EXPECTED_FRAUD_COUNT = 866


# ---------------------------------------------------------------------------
# Inference profiles
# ---------------------------------------------------------------------------

PROFILE_TEXT_ONLY = "text_only"
PROFILE_STRUCTURED = "structured"

PROFILES: tuple[str, ...] = (PROFILE_TEXT_ONLY, PROFILE_STRUCTURED)

#: Fields concatenated into the single document handed to the text model.
#:
#: `text_only` deliberately excludes `company_profile` and `benefits`:
#:   - company_profile: a WhatsApp job ad never carries a structured company
#:     description, and in EMSCAD its *emptiness* correlates with fraud. Training
#:     on it teaches the model a correlation it can never observe in production.
#:   - benefits: borderline. Real pasted ads sometimes list benefits, often not.
#:     Excluded to stay conservative about train/serve skew.
#:
#: `benefits` is the designated ABLATION CANDIDATE — if Day 6 has slack, measure
#: whether adding it helps on the Indonesian holdout (not just on EMSCAD).
TEXT_DOCUMENT_FIELDS: dict[str, tuple[str, ...]] = {
    PROFILE_TEXT_ONLY: ("title", "description", "requirements"),
    PROFILE_STRUCTURED: ("title", "company_profile", "description", "requirements", "benefits"),
}

#: Columns the model must NEVER see for a given profile, because they are not
#: recoverable from the user's input at inference time.
FORBIDDEN_MODEL_COLUMNS: dict[str, frozenset[str]] = {
    PROFILE_TEXT_ONLY: frozenset(
        {
            # Platform-side metadata — unobtainable from pasted text.
            "has_company_logo",
            "has_questions",
            "telecommuting",
            # Job-board taxonomy fields — a chat message has no taxonomy.
            "department",
            "industry",
            "function",
            "required_education",
            "required_experience",
            "employment_type",
            # Structured salary column: mostly empty in EMSCAD, denominated in USD,
            # and irrelevant to us — salary is extracted from raw text by the rule
            # layer and compared against Indonesian UMK instead.
            "salary_range",
            # Not recoverable from a chat paste.
            "company_profile",
            "benefits",
            # Identifier — pure leakage risk, zero semantic content.
            "job_id",
        }
    ),
    PROFILE_STRUCTURED: frozenset({"job_id", "salary_range"}),
}

#: `location` is extracted and used by the RULE layer (to pick the right UMK) but is
#: never fed to the text model — EMSCAD is US/UK-heavy and the model would learn
#: geography as a fraud proxy, which does not transfer to Indonesia at all.
RULE_ONLY_COLUMNS: frozenset[str] = frozenset({"location"})


# ---------------------------------------------------------------------------
# Rule feature vector
# ---------------------------------------------------------------------------
#
# ORDER IS LOAD-BEARING. The fusion meta-model (ml/train_fusion.py) is a plain
# LogisticRegression over a positional numpy array. If serving builds this vector
# in a different order than training did, every coefficient is applied to the wrong
# feature and the product breaks in a way that no test of either half will catch.
# Both sides import RULE_FEATURE_ORDER from here. Never reorder in place — append.

#: Rule signals we EXPECT to be learnable from EMSCAD, so the fusion model
#: estimates their weights from data rather than us picking numbers by hand.
#:
#: ⚠️ VERIFY ON DAY 1 (step 1.2). EMSCAD anonymises emails and URLs inside the
#: description text, replacing them with `#EMAIL_<hash>#` and `#URL_<hash>#`
#: placeholders. If that is true of this copy of the data, then email-domain
#: signals are NOT derivable from EMSCAD, and these features must be demoted to
#: the bounded-penalty bucket below. `verify_emscad_derivability()` measures this
#: instead of assuming it — do not skip that check.
LEARNED_RULE_FEATURES: tuple[str, ...] = (
    "email_free_provider",
    "email_absent",
    "email_domain_mismatch",
    "contact_messaging_only",
    "qualification_conflict",
    "url_shortener",
)

#: Indonesia-specific signals with no EMSCAD support. Per concept paper section 3.3
#: these are applied as BOUNDED ADDITIVE PENALTIES on top of the fused probability,
#: with a hard cap each, so that no deterministic rule can ever dominate the model.
PENALTY_RULE_FEATURES: tuple[str, ...] = (
    "salary_implausible_vs_umk",
    "risk_phrase_score_id",
    "payment_request_id",
)

#: Per-feature caps, in probability units added to p_fusi.
PENALTY_CAPS: dict[str, float] = {
    "salary_implausible_vs_umk": 0.05,
    "risk_phrase_score_id": 0.05,
    "payment_request_id": 0.05,
}

#: Paper section 3.3: "tidak ada satu aturan deterministik pun yang dapat
#: mendominasi keputusan model". 0.15 is at most a 15-point swing on a 0-100 score.
MAX_TOTAL_PENALTY = 0.15

RULE_FEATURE_ORDER: tuple[str, ...] = LEARNED_RULE_FEATURES + PENALTY_RULE_FEATURES


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


class FeatureContractViolation(AssertionError):
    """Raised when code tries to feed the model something it cannot have at inference."""


def assert_valid_profile(profile: str) -> None:
    if profile not in PROFILES:
        raise FeatureContractViolation(
            f"Unknown inference profile {profile!r}. Expected one of {PROFILES}."
        )


def assert_no_forbidden_columns(columns: Iterable[str], profile: str) -> None:
    """Fail loudly if any column outside the profile's allowance reaches the model.

    Call this in `ml/prepare_data.py` immediately before writing the training
    matrix, and in `ml/train_fusion.py` before fitting. It is cheap and it is the
    only thing standing between us and a model that scores 0.97 offline and
    guesses at random in production.
    """
    assert_valid_profile(profile)
    forbidden = FORBIDDEN_MODEL_COLUMNS[profile] | RULE_ONLY_COLUMNS
    offending = sorted(set(columns) & forbidden)
    if offending:
        raise FeatureContractViolation(
            f"Columns {offending} are forbidden for profile {profile!r} because they "
            f"cannot be recovered from user input at inference time. "
            f"See MVP_PLAN.md section 1.1."
        )


def assert_rule_vector(names: Sequence[str]) -> None:
    """Guard the positional contract between training and serving."""
    if tuple(names) != RULE_FEATURE_ORDER:
        raise FeatureContractViolation(
            "Rule feature vector order mismatch.\n"
            f"  expected: {RULE_FEATURE_ORDER}\n"
            f"  got:      {tuple(names)}\n"
            "The fusion model indexes this vector positionally; a mismatch silently "
            "applies every coefficient to the wrong feature."
        )


def assert_penalty_caps_sane() -> None:
    """The bounded-penalty guarantee from paper section 3.3, checked at import time."""
    if set(PENALTY_CAPS) != set(PENALTY_RULE_FEATURES):
        raise FeatureContractViolation("PENALTY_CAPS must cover exactly PENALTY_RULE_FEATURES.")
    total = sum(PENALTY_CAPS.values())
    if total > MAX_TOTAL_PENALTY + 1e-9:
        raise FeatureContractViolation(
            f"Penalty caps sum to {total:.3f}, exceeding MAX_TOTAL_PENALTY="
            f"{MAX_TOTAL_PENALTY}. This would let deterministic rules dominate the "
            f"model, contradicting concept paper section 3.3."
        )


def text_document_columns(profile: str) -> tuple[str, ...]:
    assert_valid_profile(profile)
    return TEXT_DOCUMENT_FIELDS[profile]


# Fail at import time rather than three days later during a demo.
assert_penalty_caps_sane()
