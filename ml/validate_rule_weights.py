"""Does the rule layer actually help? — MVP_PLAN.md step 3.1 (revised).

Because the weights in `ml/rule_weights.py` are set a priori rather than fitted, the
burden of proof sits here: measure on the held-out Indonesian set whether adding the
rule layer improves over the text model alone.

    python ml/validate_rule_weights.py                      # real holdout
    python ml/validate_rule_weights.py --path eval/synthetic_fixture.jsonl \
        --allow-synthetic                                   # plumbing check only

## The rule that keeps this honest

Do NOT adjust the weights in response to what this reports. The moment they are tuned
against the holdout, the holdout has become training data and the headline number is
gone. Run it, record the answer, and report it — including if the answer is "the rule
layer did not help".

Until the text model exists (step 2.1), `--rules-only` evaluates the rule layer as a
standalone classifier, which is a useful early read on whether the severities are
sane.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from api.ingest import ingest  # noqa: E402
from api.rules.engine import default_engine  # noqa: E402
from api.scoring import compute_score, rule_contributions  # noqa: E402
from ml.eval_set import DEFAULT_HOLDOUT, SYNTHETIC_BANNER, load_eval_set  # noqa: E402
from ml.rule_weights import RULE_WEIGHTS, WEIGHTS_VERSION  # noqa: E402


def _safe_import_metrics():
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    return average_precision_score, precision_recall_fscore_support, roc_auc_score


def evaluate(eval_set, model_probabilities: np.ndarray | None) -> dict:
    """Score every item with and without the rule layer."""
    average_precision, prf, roc_auc = _safe_import_metrics()

    engine = default_engine()
    labels = np.array(eval_set.labels)

    if model_probabilities is None:
        # No text model yet: treat it as maximally uninformative so that whatever
        # separation appears comes purely from the rules.
        model_probabilities = np.full(len(eval_set), 0.5)

    fused, shifts, fired_counts = [], [], []
    per_rule_fires = {name: 0 for name in RULE_WEIGHTS}

    for item, p_text in zip(eval_set, model_probabilities):
        evaluation = engine.evaluate(ingest(item.text))
        breakdown = compute_score(float(p_text), evaluation)
        fused.append(breakdown.fused_probability)
        shifts.append(breakdown.rule_shift_applied)

        contributions = rule_contributions(evaluation)
        fired_counts.append(len(contributions))
        for name in contributions:
            per_rule_fires[name] += 1

    fused = np.array(fused)
    shifts = np.array(shifts)

    def metrics(scores: np.ndarray) -> dict:
        # Degenerate input (one class only, or no variation) makes these undefined.
        if len(set(labels)) < 2 or float(np.ptp(scores)) == 0.0:
            return {"pr_auc": float("nan"), "roc_auc": float("nan"),
                    "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
        precision, recall, f1, _ = prf(
            labels, (scores >= 0.5).astype(int), average="binary", zero_division=0
        )
        return {
            "pr_auc": float(average_precision(labels, scores)),
            "roc_auc": float(roc_auc(labels, scores)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    scam = labels == 1
    return {
        "n_items": len(eval_set),
        "n_scam": int(scam.sum()),
        "n_legit": int((~scam).sum()),
        "weights_version": WEIGHTS_VERSION,
        "text_only": metrics(np.asarray(model_probabilities, dtype=float)),
        "fused": metrics(fused),
        "mean_shift_scam": float(shifts[scam].mean()) if scam.any() else float("nan"),
        "mean_shift_legit": float(shifts[~scam].mean()) if (~scam).any() else float("nan"),
        "mean_rules_fired_scam": float(np.array(fired_counts)[scam].mean()) if scam.any() else 0.0,
        "mean_rules_fired_legit": float(np.array(fired_counts)[~scam].mean()) if (~scam).any() else 0.0,
        "legit_items_with_any_rule": int((np.array(fired_counts)[~scam] > 0).sum()) if (~scam).any() else 0,
        "per_rule_fires": per_rule_fires,
    }


def render(report: dict, *, synthetic: bool) -> str:
    lines: list[str] = ["# Rule-weight validation", ""]
    if synthetic:
        lines += [SYNTHETIC_BANNER, ""]

    lines += [
        f"Weights: `{report['weights_version']}` (set a priori, see ml/rule_weights.py)",
        "",
        f"Items: {report['n_items']} ({report['n_scam']} scam, {report['n_legit']} legitimate)",
        "",
        "## Does the rule layer help?",
        "",
        "| metric | text only | + rules | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
        before, after = report["text_only"][key], report["fused"][key]
        delta = after - before
        lines.append(f"| {key} | {before:.4f} | {after:.4f} | {delta:+.4f} |")

    lines += [
        "",
        "## Separation",
        "",
        f"- mean probability shift on scam ads: **{report['mean_shift_scam']:.4f}**",
        f"- mean probability shift on legitimate ads: **{report['mean_shift_legit']:.4f}**",
        f"- mean rules fired, scam: {report['mean_rules_fired_scam']:.2f}",
        f"- mean rules fired, legitimate: {report['mean_rules_fired_legit']:.2f}",
        f"- legitimate ads triggering **any** rule: "
        f"{report['legit_items_with_any_rule']} / {report['n_legit']}",
        "",
        "The shift on scam ads must exceed the shift on legitimate ones. If it does",
        "not, the rule layer is adding noise and should be reported as such.",
        "",
        "The false-positive count is the number section 3.6 cares about most.",
        "",
        "## Rule fire counts",
        "",
        "| rule | weight | times fired |",
        "| --- | ---: | ---: |",
    ]
    for name, count in sorted(report["per_rule_fires"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {RULE_WEIGHTS[name]:.2f} | {count} |")

    lines += [
        "",
        "A rule that never fires contributes nothing and should be cut or fixed.",
        "A rule that fires on almost everything is not discriminating.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--probabilities", type=Path, default=None,
                        help="JSON list of calibrated p(scam), aligned with the file order.")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("eval/rule_weight_validation.md"))
    args = parser.parse_args()

    eval_set = load_eval_set(args.path)
    eval_set.require_real("validate rule weights", allow_synthetic=args.allow_synthetic)

    probabilities = None
    if args.probabilities:
        values = json.loads(args.probabilities.read_text(encoding="utf-8"))
        if len(values) != len(eval_set):
            raise SystemExit(
                f"{args.probabilities} has {len(values)} probabilities but "
                f"{args.path} has {len(eval_set)} items."
            )
        probabilities = np.asarray(values, dtype=float)
    else:
        print("NOTE: no text-model probabilities supplied; evaluating rules alone "
              "against a constant 0.5 baseline.\n")

    report = evaluate(eval_set, probabilities)

    if eval_set.is_synthetic:
        print("*** SYNTHETIC FIXTURE — these numbers measure nothing. ***\n")

    print(f"items                : {report['n_items']} "
          f"({report['n_scam']} scam, {report['n_legit']} legitimate)")
    print(f"mean shift, scam     : {report['mean_shift_scam']:.4f}")
    print(f"mean shift, legit    : {report['mean_shift_legit']:.4f}")
    print(f"legit firing any rule: {report['legit_items_with_any_rule']} / {report['n_legit']}")
    print(f"\nPR-AUC text only     : {report['text_only']['pr_auc']:.4f}")
    print(f"PR-AUC with rules    : {report['fused']['pr_auc']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(report, synthetic=eval_set.is_synthetic), encoding="utf-8")
    print(f"\nwrote {args.out}")

    if report["mean_shift_scam"] <= report["mean_shift_legit"]:
        print("\nWARNING: rules shift legitimate ads as much as scams. "
              "The layer is not discriminating on this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
