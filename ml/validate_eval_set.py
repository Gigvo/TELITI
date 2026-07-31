"""Validate an Indonesian evaluation file — MVP_PLAN.md step 1.5.

Run this while annotating, not at the end. It catches malformed rows, duplicate
ids, duplicated ad text, and missing independent labels as they are entered, and
reports Cohen's kappa so the team can see agreement drifting before 200 items are
done.

    python ml/validate_eval_set.py
    python ml/validate_eval_set.py --path eval/synthetic_fixture.jsonl
    python ml/validate_eval_set.py --target-items 200 --target-scam-rate 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.eval_set import (  # noqa: E402
    DEFAULT_HOLDOUT,
    EvalSetError,
    annotation_report,
    load_eval_set,
)

KAPPA_SUBSTANTIAL = 0.60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--target-items", type=int, default=200)
    parser.add_argument(
        "--target-scam-rate",
        type=float,
        default=0.35,
        help="Holdout should lean legitimate: false positives are the error that matters.",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Report problems without exiting non-zero. Useful mid-annotation.",
    )
    args = parser.parse_args()

    try:
        eval_set = load_eval_set(args.path, strict=not args.lenient)
    except EvalSetError as exc:
        print(f"FAILED\n{exc}")
        return 1

    report = annotation_report(eval_set)

    print(f"file          : {report['path']}")
    if report["is_synthetic"]:
        print("provenance    : SYNTHETIC — exercises the pipeline, measures nothing")
    else:
        print("provenance    : real annotations")

    print(f"items         : {report['n_items']} / {args.target_items} target")
    print(
        f"labels        : {report['n_scam']} scam, {report['n_legit']} legitimate "
        f"({report['scam_rate'] * 100:.1f}% scam)"
    )
    print(f"double-labeled: {report['n_double_annotated']}")
    print(f"disagreements : {report['n_disagreements']}")

    kappa = report["cohens_kappa"]
    print(f"Cohen's kappa : {kappa:.3f}" if kappa == kappa else "Cohen's kappa : n/a")

    print(f"channels      : {report['channels']}")
    print(f"source types  : {report['source_types']}")

    advice: list[str] = []
    for warning in report["warnings"]:
        advice.append(f"duplicate text: {warning}")

    if report["n_items"] < args.target_items:
        advice.append(
            f"{args.target_items - report['n_items']} more items to reach target"
        )
    if kappa == kappa and report["n_double_annotated"] >= 30 and kappa < KAPPA_SUBSTANTIAL:
        advice.append(
            f"kappa {kappa:.2f} is below {KAPPA_SUBSTANTIAL} (substantial). The two "
            f"annotators are applying different criteria — reconcile the definition "
            f"of 'scam' now, before the remaining items are labelled."
        )
    if not report["is_synthetic"] and report["n_double_annotated"] < report["n_items"]:
        advice.append(
            f"{report['n_items'] - report['n_double_annotated']} item(s) lack "
            f"independent label_a/label_b"
        )
    if report["n_items"] and abs(report["scam_rate"] - args.target_scam_rate) > 0.15:
        advice.append(
            f"scam rate {report['scam_rate'] * 100:.0f}% is far from the "
            f"{args.target_scam_rate * 100:.0f}% target; a holdout that leans "
            f"legitimate is what surfaces false positives"
        )

    if advice:
        print("\nto do:")
        for line in advice:
            print(f"  - {line}")
    else:
        print("\nlooks good.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
