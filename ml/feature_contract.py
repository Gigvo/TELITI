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

#: Bookkeeping columns a prepared frame legitimately carries. They are NOT model
#: inputs: `job_id` identifies a row for split tracking and reproducibility,
#: `fraudulent` is the target, `text` is the vetted document, `n_words` is a stat.
#: `assert_no_forbidden_columns` is about what reaches the MODEL, so these are
#: excluded before the check — otherwise the identifier trips the guard meant to
#: catch platform metadata.
BOOKKEEPING_COLUMNS: frozenset[str] = frozenset({"job_id", "text", "n_words", LABEL_COLUMN})


# ---------------------------------------------------------------------------
# Rule feature vector
# ---------------------------------------------------------------------------
#
# ORDER IS LOAD-BEARING. The fusion meta-model (ml/train_fusion.py) is a plain
# LogisticRegression over a positional numpy array. If serving builds this vector
# in a different order than training did, every coefficient is applied to the wrong
# feature and the product breaks in a way that no test of either half will catch.
# Both sides import RULE_FEATURE_ORDER from here. Never reorder in place — append.

#: ORDER IS LOAD-BEARING and is declared explicitly rather than derived from the
#: buckets below, so that reclassifying a feature never silently permutes the
#: vector the fusion model indexes positionally. Append only; never reorder.
RULE_FEATURE_ORDER: tuple[str, ...] = (
    "email_free_provider",
    "email_absent",
    "email_domain_mismatch",
    "contact_messaging_only",
    "qualification_conflict",
    "url_shortener",
    "salary_implausible_vs_umk",
    "risk_phrase_score_id",
    "payment_request_id",
)

# ---------------------------------------------------------------------------
# MEASURED 2026-07-31 — ml/verify_derivability.py, eval/derivability_report.md
# ---------------------------------------------------------------------------
#
# The concept paper (section 3.3) assumed some rule signals could be learned from
# EMSCAD metadata: "Sebagian sinyal aturan (domain email, keberadaan profil
# perusahaan) tersedia langsung pada metadata EMSCAD sehingga dapat dipelajari
# model meta". Measurement contradicts this for the `text_only` profile.
#
#   real email address present in document : 0.0%   (1 row in 3,000)
#   real URL present in document           : 0.3%
#   visible `#EMAIL_x#` / `#URL_x#` marker : 24.8%
#
# EMSCAD removed contact details before publication. Only about a quarter left a
# placeholder behind; the rest were stripped SILENTLY, so a contact rule cannot even
# reliably detect that it is looking at redacted text.
#
# Worse, the redaction rate differs by class — 32.1% of fraudulent rows carry a
# placeholder versus 25.2% of real ones. A model given those features would learn
# "placeholder present -> fraudulent", score well offline, and rely on an artefact
# that cannot exist in a user's WhatsApp paste.
#
# Consequence: only ONE rule feature is learnable from EMSCAD.

#: Derivable from EMSCAD text. `qualification_conflict` survives because it is a
#: property of the prose itself ("fresh graduate" alongside "5 years experience"),
#: not of the contact metadata. The rule must therefore carry ENGLISH patterns as
#: well as Indonesian ones, or it is unlearnable here too (step 2.4).
EMSCAD_DERIVABLE_FEATURES: tuple[str, ...] = ("qualification_conflict",)

#: Everything else. Weights are fitted on a dedicated Indonesian fusion TRAINING
#: set (`eval/indonesian_fusion_train.jsonl`), which is separate from and disjoint
#: with the held-out evaluation set. See MVP_PLAN.md step 3.1.
INDONESIAN_FITTED_FEATURES: tuple[str, ...] = tuple(
    f for f in RULE_FEATURE_ORDER if f not in EMSCAD_DERIVABLE_FEATURES
)

#: Where the fusion meta-model's training data comes from. Recorded here because it
#: is a deviation from the concept paper worth stating plainly in the write-up: the
#: paper trains fusion on an EMSCAD validation partition, we cannot.
FUSION_TRAINING_SOURCE = "indonesian_fusion_train"

# ---------------------------------------------------------------------------
# Bounded contribution — the concept paper's section 3.3 guarantee
# ---------------------------------------------------------------------------
#
# Weights are now LEARNED rather than hand-set, but the caps remain as a safety
# rail: the Indonesian fusion set is small (order 10^2) and 9 features fitted on it
# will produce noisy coefficients. The cap bounds how far any single noisy weight
# can move a user's score.
#
# NOTE FOR THE WRITE-UP: section 3.3 currently states an aggregate cap of 0.15.
# That figure was sized for 3 penalty rules. With 8 rules sharing it each would be
# worth ~2 score points and the rule layer would be decorative. The guarantee is
# therefore expressed PER RULE, which is what "tidak ada satu aturan deterministik
# pun yang dapat mendominasi" actually asserts.

#: No single rule may shift the fused probability by more than this — at most a
#: 10-point swing on the 0-100 score.
PER_RULE_CONTRIBUTION_CAP = 0.10

#: Aggregate ceiling across all rules combined.
MAX_TOTAL_RULE_SHIFT = 0.45

#: Backwards-compatible alias; prefer MAX_TOTAL_RULE_SHIFT in new code.
MAX_TOTAL_PENALTY = MAX_TOTAL_RULE_SHIFT


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


#: A rule allowed to shift p(scam) by more than this is, by any reasonable reading,
#: dominating the model's own judgement.
MAX_SAFE_PER_RULE_CAP = 0.5


def assert_penalty_caps_sane() -> None:
    """The bounded-contribution guarantee from paper section 3.3, checked at import.

    Order matters: the dominance check runs BEFORE the relational one, so that an
    absurd cap reports the reason that actually matters rather than tripping the
    weaker "exceeds the aggregate" message on its way past.
    """
    if PER_RULE_CONTRIBUTION_CAP <= 0.0:
        raise FeatureContractViolation(
            f"PER_RULE_CONTRIBUTION_CAP={PER_RULE_CONTRIBUTION_CAP} must be positive."
        )
    if PER_RULE_CONTRIBUTION_CAP > MAX_SAFE_PER_RULE_CAP:
        raise FeatureContractViolation(
            f"PER_RULE_CONTRIBUTION_CAP={PER_RULE_CONTRIBUTION_CAP} lets one "
            f"deterministic rule dominate the model, contradicting section 3.3."
        )
    if PER_RULE_CONTRIBUTION_CAP > MAX_TOTAL_RULE_SHIFT:
        raise FeatureContractViolation(
            f"PER_RULE_CONTRIBUTION_CAP={PER_RULE_CONTRIBUTION_CAP} exceeds the "
            f"aggregate ceiling MAX_TOTAL_RULE_SHIFT={MAX_TOTAL_RULE_SHIFT}."
        )


def clip_rule_shift(total_shift: float) -> float:
    """Apply the aggregate ceiling to the combined rule contribution."""
    return max(-MAX_TOTAL_RULE_SHIFT, min(total_shift, MAX_TOTAL_RULE_SHIFT))


def assert_features_partitioned() -> None:
    """Every feature belongs to exactly one weight-source bucket."""
    overlap = set(EMSCAD_DERIVABLE_FEATURES) & set(INDONESIAN_FITTED_FEATURES)
    if overlap:
        raise FeatureContractViolation(f"Features in both buckets: {sorted(overlap)}")
    union = set(EMSCAD_DERIVABLE_FEATURES) | set(INDONESIAN_FITTED_FEATURES)
    if union != set(RULE_FEATURE_ORDER):
        raise FeatureContractViolation(
            f"Bucket union does not cover RULE_FEATURE_ORDER. "
            f"Missing: {sorted(set(RULE_FEATURE_ORDER) - union)}"
        )


def text_document_columns(profile: str) -> tuple[str, ...]:
    assert_valid_profile(profile)
    return TEXT_DOCUMENT_FIELDS[profile]


# Fail at import time rather than three days later during a demo.
assert_penalty_caps_sane()
assert_features_partitioned()
