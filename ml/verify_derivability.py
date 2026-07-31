"""Measure which rule features EMSCAD can actually teach — MVP_PLAN.md step 1.2.

The concept paper (section 3.3) assumes the fusion meta-model can learn weights for
rule signals like email domain from EMSCAD metadata. This script tests that claim
instead of trusting it, and writes `eval/derivability_report.md`.

Run it whenever the corpus or the rule set changes:

    python ml/verify_derivability.py

A feature is DERIVABLE only if the signal it depends on is actually present in the
corpus text. A feature that is absent almost everywhere cannot be fitted here, and
a feature whose *availability* differs by class is worse than useless — the model
will learn the redaction artefact rather than the fraud.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# See the equivalent note in ml/prepare_data.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from api.ingest import REDACTION_RE, ingest
from api.rules.engine import default_engine
from ml.feature_contract import (
    EMSCAD_DERIVABLE_FEATURES,
    LABEL_COLUMN,
    PROFILE_TEXT_ONLY,
    RULE_FEATURE_ORDER,
    text_document_columns,
)
from ml.text_cleaning import build_document

#: A feature must fire on a usable FRACTION of the corpus to be fittable.
#:
#: Both bounds matter. Below the floor there is nothing to learn from. Above the
#: ceiling the feature is effectively constant and equally useless — and in this
#: corpus a near-100% fire rate is itself the redaction artefact, since EMSCAD
#: stripped the contact details that would otherwise have cleared the rule.
PRESENCE_FLOOR = 0.05
PRESENCE_CEILING = 0.95

#: Above this absolute difference in availability between classes, the feature leaks
#: the redaction process rather than the fraud.
CLASS_SKEW_THRESHOLD = 0.03


def build_documents(df: pd.DataFrame, profile: str) -> pd.Series:
    fields = text_document_columns(profile)
    frame = df[list(fields)].fillna("")
    return frame.apply(lambda row: build_document(row.to_dict(), fields), axis=1)


def measure(df: pd.DataFrame, docs: pd.Series) -> tuple[pd.DataFrame, dict]:
    engine = default_engine()
    labels = df[LABEL_COLUMN].to_numpy()

    availability: dict[str, list[bool]] = {f: [] for f in RULE_FEATURE_ORDER}
    fired: dict[str, list[bool]] = {f: [] for f in RULE_FEATURE_ORDER}

    has_email, has_url, has_placeholder = [], [], []

    for text in docs:
        ctx = ingest(text)
        has_email.append(bool(ctx.emails))
        has_url.append(bool(ctx.urls))
        has_placeholder.append(bool(REDACTION_RE.search(text)))

        evaluation = engine.evaluate(ctx)
        for feature in RULE_FEATURE_ORDER:
            outcome = evaluation.outcomes.get(feature)
            availability[feature].append(bool(outcome and outcome.available))
            fired[feature].append(bool(outcome and outcome.fired))

    rows = []
    for feature in RULE_FEATURE_ORDER:
        avail = pd.Series(availability[feature])
        fire = pd.Series(fired[feature])
        avail_real = avail[labels == 0].mean()
        avail_fraud = avail[labels == 1].mean()
        skew = abs(avail_fraud - avail_real)

        implemented = feature not in engine.pending_features
        if not implemented:
            verdict = "NOT IMPLEMENTED"
        elif fire.mean() < PRESENCE_FLOOR:
            verdict = "NOT DERIVABLE (signal absent)"
        elif fire.mean() > PRESENCE_CEILING:
            verdict = "NOT DERIVABLE (fires almost always)"
        elif skew > CLASS_SKEW_THRESHOLD:
            verdict = "UNSAFE (availability differs by class)"
        else:
            verdict = "DERIVABLE"

        rows.append(
            {
                "feature": feature,
                "implemented": implemented,
                "available_%": round(avail.mean() * 100, 1),
                "fired_%": round(fire.mean() * 100, 1),
                "avail_real_%": round(avail_real * 100, 1),
                "avail_fraud_%": round(avail_fraud * 100, 1),
                "class_skew": round(skew, 3),
                "verdict": verdict,
            }
        )

    corpus = {
        "documents": len(docs),
        "real_email_present_%": round(pd.Series(has_email).mean() * 100, 2),
        "real_url_present_%": round(pd.Series(has_url).mean() * 100, 2),
        "redaction_placeholder_%": round(pd.Series(has_placeholder).mean() * 100, 2),
    }
    return pd.DataFrame(rows), corpus


def _markdown_table(table: pd.DataFrame) -> str:
    """Render without pandas.to_markdown, which needs the `tabulate` package.

    Not worth an extra dependency for one table in one report.
    """
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def render_report(table: pd.DataFrame, corpus: dict, sample_size: int) -> str:
    derivable = sorted(table.loc[table.verdict == "DERIVABLE", "feature"])
    lines = [
        "# EMSCAD derivability report",
        "",
        "Generated by `ml/verify_derivability.py`. Tests the concept paper's section",
        "3.3 assumption that rule weights can be fitted on EMSCAD.",
        "",
        f"Sample: **{sample_size}** documents, `text_only` profile.",
        "",
        "## Corpus-level contact signal",
        "",
        "| property | rate |",
        "| --- | ---: |",
        f"| real email address present | {corpus['real_email_present_%']}% |",
        f"| real URL present | {corpus['real_url_present_%']}% |",
        f"| visible redaction placeholder | {corpus['redaction_placeholder_%']}% |",
        "",
        "EMSCAD removed contact details before publication. Only a minority left a",
        "`#EMAIL_x#` / `#URL_x#` marker behind; the rest were stripped silently, so a",
        "contact rule cannot even reliably tell that it is looking at redacted text.",
        "",
        "## Per-feature verdicts",
        "",
        _markdown_table(table),
        "",
        "## Conclusion",
        "",
        f"Derivable from EMSCAD: **{', '.join(derivable) if derivable else 'none'}**.",
        "",
        "Everything else must have its weight fitted on Indonesian data",
        "(`eval/indonesian_fusion_train.jsonl`), per the decision recorded in",
        "`ml/feature_contract.py::FUSION_TRAINING_SOURCE`.",
        "",
        "`class_skew` is the difference in feature AVAILABILITY between fraudulent and",
        "real postings. A non-trivial skew means the feature partly encodes how the",
        "corpus was anonymised. Since that artefact cannot exist in a user's pasted",
        "text, a model fitted on it would score well offline and fail in production.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/fake_job_postings.csv"))
    parser.add_argument("--out", type=Path, default=Path("eval/derivability_report.md"))
    parser.add_argument("--sample", type=int, default=0, help="0 = whole corpus")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=args.seed)
    df = df.reset_index(drop=True)

    docs = build_documents(df, PROFILE_TEXT_ONLY)
    table, corpus = measure(df, docs)

    print(f"\ncorpus: {corpus}\n")
    print(table.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(table, corpus, len(df)), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # Compare only where measurement can actually speak. A feature with no rule
    # implemented yet is unmeasurable, not contradicted — reporting it as a contract
    # violation would train the team to ignore this warning.
    # ASCII only: the Windows console is cp1252 and cannot encode emoji.
    assessable = table[table.implemented]
    measured_derivable = set(assessable.loc[assessable.verdict == "DERIVABLE", "feature"])
    measured_not = set(assessable.loc[assessable.verdict != "DERIVABLE", "feature"])
    declared = set(EMSCAD_DERIVABLE_FEATURES)

    pending = set(table.loc[~table.implemented, "feature"])
    if pending:
        print(f"\nNOTE: not yet implemented, so not assessable: {sorted(pending)}")

    wrongly_declared = declared & measured_not
    could_be_promoted = measured_derivable - declared
    if wrongly_declared:
        print(
            f"\nWARNING: feature_contract declares {sorted(wrongly_declared)} derivable "
            f"from EMSCAD, but measurement disagrees. Update the contract."
        )
    if could_be_promoted:
        print(
            f"\nNOTE: {sorted(could_be_promoted)} measure as derivable but are declared "
            f"Indonesian-fitted. Promoting them would let EMSCAD supply their weights."
        )
    if not wrongly_declared and not could_be_promoted:
        print("\nContract agrees with measurement for every implemented feature.")


if __name__ == "__main__":
    main()
