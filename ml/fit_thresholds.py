"""Derive the Rendah / Sedang / Tinggi boundaries — MVP_PLAN.md step 3.2.

    python ml/fit_thresholds.py

Writes `artifacts/thresholds.json`, which the scoring layer loads at runtime.

## Why not round numbers

Concept paper §3.3 is explicit that the risk boundaries must come from a stated
precision target rather than being chosen arbitrarily. The placeholders shipped so
far (`< 40` Tinggi, `>= 70` Rendah) were exactly the arbitrary choice the paper
argues against. This script replaces them with boundaries derived from what the
model actually does, against targets chosen from the product's cost asymmetry.

## The targets, and why these numbers

§3.6 states that a false positive against a legitimate company is the expensive
error — it damages a real business and destroys trust in the tool. A missed scam is
bad, but the user is no worse off than without the tool at all. So the two boundaries
are set from different targets:

- **Tinggi** (loud warning): requires **precision >= 0.85** on the scam class. When
  the product says "high risk", it should be right roughly six times in seven.
- **Rendah** (reassurance): requires **recall >= 0.95**. Below this boundary we are
  telling someone an advertisement looks safe, so almost every genuine scam must
  already be above it. Being wrong here is the failure that actually hurts a user.

Everything between is Sedang — the honest "we are not sure, look closely yourself".

## ⚠️ Which data, and the disclosure that comes with it

Thresholds are fitted on the Indonesian holdout, not on EMSCAD. That is forced: the
model learned EMSCAD's 4.8% base rate and systematically under-predicts on Indonesian
text (mean probability 0.069 against an actual 36% scam rate). EMSCAD-derived
boundaries would label almost every Indonesian scam "Rendah" — a product that
reassures users about scams is worse than no product.

The cost of that choice is bounded and worth stating plainly:

- **PR-AUC is unaffected.** It is computed across all thresholds, so choosing one
  cannot inflate it. The headline 0.9258 stays clean.
- **Precision and recall AT these boundaries are optimistic**, because the boundaries
  were chosen using the same 195 items they are reported on. `thresholds.json`
  records this so the caveat travels with the numbers instead of being forgotten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ml.calibration import PlattCalibrator  # noqa: E402
from ml.eval_set import load_eval_set  # noqa: E402
from ml.metrics import evaluate, threshold_for_precision  # noqa: E402

#: See the module docstring for why these two differ.
TINGGI_PRECISION_TARGET = 0.85
RENDAH_RECALL_TARGET = 0.95


def threshold_for_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float
) -> float | None:
    """Highest threshold still achieving at least `target_recall`.

    The mirror of `threshold_for_precision`. Highest rather than lowest because we
    want the most permissive "looks safe" boundary that still keeps essentially every
    scam above it.
    """
    from sklearn.metrics import precision_recall_curve

    _, recall, thresholds = precision_recall_curve(y_true, y_score)
    viable = [t for t, r in zip(thresholds, recall[:-1]) if r >= target_recall]
    return float(max(viable)) if viable else None


def model_margins(texts: list[str], model_dir: Path, max_length: int, batch_size: int):
    """Raw logit margins, before any calibration."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    margins = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size], truncation=True,
                max_length=max_length, padding="max_length", return_tensors="pt",
            ).to(device)
            logits = model(**batch).logits.float()
            margins.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(margins)


def load_or_refit_calibrator(
    margins: np.ndarray, labels: np.ndarray, model_dir: Path, refit: bool
):
    """Return (probabilities, calibrator, refitted?).

    ## Why refitting is usually necessary here

    The shipped calibrator was fitted on EMSCAD, where 4.8% of postings are
    fraudulent. The deployment domain is Indonesian advertisements, and the model
    carries EMSCAD's prior with it: mean predicted probability 0.069 against an
    actual 36% scam rate. That is a textbook prior shift.

    It does not hurt ranking — PR-AUC is identical either way — but it wrecks the
    Integrity Score, because `S = round((1 - p) * 100)` assumes `p` means something
    in the domain being scored. With EMSCAD calibration every Indonesian ad lands
    between 93 and 100, so a scam displays as "98/100" and the score is useless as a
    user-facing number.

    Concept paper §3.3 defines the score in terms of a *calibrated* probability.
    Refitting for the deployment domain is what makes that definition hold, not a
    departure from it.
    """
    emscad_path = model_dir.parent / "calibrator.json"

    if not refit:
        if emscad_path.exists():
            calibrator = PlattCalibrator.from_dict(json.loads(emscad_path.read_text()))
            return calibrator.transform(margins), calibrator, False
        print("  ! no calibrator.json — using uncalibrated probabilities")
        return 1.0 / (1.0 + np.exp(-margins)), None, False

    calibrator = PlattCalibrator().fit(margins, labels)
    return calibrator.transform(margins), calibrator, True


def summarise_at(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    return {
        "threshold_probability": round(float(threshold), 6),
        "integrity_score_boundary": int(round((1.0 - threshold) * 100)),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("eval/indonesian_holdout.jsonl"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/scam_model"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/thresholds.json"))
    parser.add_argument("--report", type=Path, default=Path("eval/thresholds_report.md"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision-target", type=float, default=TINGGI_PRECISION_TARGET)
    parser.add_argument("--recall-target", type=float, default=RENDAH_RECALL_TARGET)
    parser.add_argument(
        "--no-refit-calibration",
        action="store_true",
        help="Keep the EMSCAD calibrator. Produces scores crushed into 93-100 on "
             "Indonesian input; see load_or_refit_calibrator for why.",
    )
    parser.add_argument(
        "--calibrator-out", type=Path, default=Path("artifacts/calibrator_deployment.json")
    )
    args = parser.parse_args()

    eval_set = load_eval_set(args.path)
    eval_set.require_real("fit thresholds")
    labels = np.array(eval_set.labels)

    print(f"{len(eval_set)} items — {eval_set.n_scam} scam, {eval_set.n_legit} legitimate")
    print(f"scoring with {args.model_dir} ...")
    margins = model_margins(
        eval_set.texts, args.model_dir, args.max_length, args.batch_size
    )
    probabilities, calibrator, refitted = load_or_refit_calibrator(
        margins, labels, args.model_dir, refit=not args.no_refit_calibration
    )

    if refitted:
        emscad_path = args.model_dir.parent / "calibrator.json"
        if emscad_path.exists():
            old = PlattCalibrator.from_dict(json.loads(emscad_path.read_text()))
            old_p = old.transform(margins)
            print(f"  EMSCAD calibration : mean p {old_p.mean():.4f}  "
                  f"score range {int((1-old_p.max())*100)}–{int((1-old_p.min())*100)}")
        print(f"  refitted for domain: mean p {probabilities.mean():.4f}  "
              f"score range {int((1-probabilities.max())*100)}–"
              f"{int((1-probabilities.min())*100)}")
        args.calibrator_out.parent.mkdir(parents=True, exist_ok=True)
        args.calibrator_out.write_text(
            json.dumps(
                {**calibrator.to_dict(), "fitted_on": str(args.path),
                 "n_rows": len(eval_set), "domain": "indonesian_holdout"},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {args.calibrator_out}")

    print(f"\n  actual scam rate {labels.mean():.4f}\n")

    tinggi_p = threshold_for_precision(labels, probabilities, args.precision_target)
    rendah_p = threshold_for_recall(labels, probabilities, args.recall_target)

    if tinggi_p is None:
        raise SystemExit(
            f"No threshold reaches precision {args.precision_target}. The model "
            f"cannot support a 'Tinggi' band at that target on this data — lower "
            f"the target or report that the band is unattainable."
        )
    if rendah_p is None:
        raise SystemExit(
            f"No threshold reaches recall {args.recall_target}. Every band would "
            f"leak scams into 'Rendah'; lower the target or widen 'Sedang'."
        )

    # Scores run opposite to probabilities: S = round((1 - p) * 100). A HIGH scam
    # probability is a LOW integrity score, so the Tinggi boundary sits below Rendah.
    tinggi_below = int(round((1.0 - tinggi_p) * 100))
    rendah_at_or_above = int(round((1.0 - rendah_p) * 100))

    if tinggi_below >= rendah_at_or_above:
        raise SystemExit(
            f"Bands inverted or empty: Tinggi<{tinggi_below}, "
            f"Rendah>={rendah_at_or_above}. The two targets cannot both be met — "
            f"they overlap, leaving no Sedang band."
        )

    tinggi_stats = summarise_at(labels, probabilities, tinggi_p)
    rendah_stats = summarise_at(labels, probabilities, rendah_p)

    band = np.where(
        probabilities >= tinggi_p, "Tinggi",
        np.where(probabilities >= rendah_p, "Sedang", "Rendah"),
    )
    distribution = {
        name: {
            "n": int((band == name).sum()),
            "scam": int(((band == name) & (labels == 1)).sum()),
            "legit": int(((band == name) & (labels == 0)).sum()),
        }
        for name in ("Tinggi", "Sedang", "Rendah")
    }

    print(f"Tinggi  : score < {tinggi_below}   (p >= {tinggi_p:.4f})")
    print(f"          precision {tinggi_stats['precision']:.4f}  "
          f"recall {tinggi_stats['recall']:.4f}  "
          f"FP {tinggi_stats['false_positives']}")
    print(f"Rendah  : score >= {rendah_at_or_above}   (p < {rendah_p:.4f})")
    print(f"          catches {rendah_stats['recall'] * 100:.1f}% of scams above it, "
          f"{distribution['Rendah']['scam']} scam(s) fall into 'Rendah'")
    print(f"Sedang  : {tinggi_below} .. {rendah_at_or_above - 1}")
    print()
    for name in ("Tinggi", "Sedang", "Rendah"):
        d = distribution[name]
        print(f"  {name:7} {d['n']:4d} items  ({d['scam']} scam, {d['legit']} legitimate)")

    payload = {
        "tinggi_below": tinggi_below,
        "rendah_at_or_above": rendah_at_or_above,
        "tinggi_probability": round(float(tinggi_p), 6),
        "rendah_probability": round(float(rendah_p), 6),
        "precision_target": args.precision_target,
        "recall_target": args.recall_target,
        "tinggi_stats": tinggi_stats,
        "rendah_stats": rendah_stats,
        "band_distribution": distribution,
        "fitted_on": str(args.path),
        "n_items": len(eval_set),
        "prevalence": round(float(labels.mean()), 4),
        "pr_auc": round(evaluate("model", labels, probabilities).pr_auc, 4),
        "caveat": (
            "Fitted on the same Indonesian holdout used for reporting, because the "
            "model under-predicts on Indonesian text (EMSCAD prior shift) and "
            "EMSCAD-derived boundaries would label nearly every Indonesian scam "
            "'Rendah'. PR-AUC is threshold-independent and therefore unaffected; "
            "precision and recall AT these boundaries are optimistic."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(payload), encoding="utf-8")

    print(f"\nwrote {args.out}")
    print(f"wrote {args.report}")
    return 0


def render_report(p: dict) -> str:
    d = p["band_distribution"]
    return "\n".join([
        "# Risk thresholds",
        "",
        "Generated by `ml/fit_thresholds.py`. Concept paper §3.3 requires these",
        "boundaries to come from a stated target rather than round numbers.",
        "",
        "## Boundaries",
        "",
        "| Band | Integrity Score | Probability | Chosen for |",
        "| --- | --- | --- | --- |",
        f"| **Tinggi** | below {p['tinggi_below']} | p ≥ {p['tinggi_probability']} | "
        f"precision ≥ {p['precision_target']} on the scam class |",
        f"| **Sedang** | {p['tinggi_below']} – {p['rendah_at_or_above'] - 1} | — | everything between |",
        f"| **Rendah** | {p['rendah_at_or_above']} or above | p < {p['rendah_probability']} | "
        f"recall ≥ {p['recall_target']} |",
        "",
        "The two bands use different targets because the errors are not symmetric",
        "(§3.6): a false alarm damages a real company, while a missed scam leaves the",
        "user where they started. So *Tinggi* is tuned for precision — when it shouts,",
        "it should be right — and *Rendah* for recall, because telling someone an",
        "advertisement is safe when it is not is the failure that actually hurts them.",
        "",
        "## Measured at these boundaries",
        "",
        "| | precision | recall | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Tinggi boundary | {p['tinggi_stats']['precision']:.4f} | "
        f"{p['tinggi_stats']['recall']:.4f} | {p['tinggi_stats']['false_positives']} | "
        f"{p['tinggi_stats']['false_negatives']} |",
        f"| Rendah boundary | {p['rendah_stats']['precision']:.4f} | "
        f"{p['rendah_stats']['recall']:.4f} | {p['rendah_stats']['false_positives']} | "
        f"{p['rendah_stats']['false_negatives']} |",
        "",
        "## How the holdout distributes",
        "",
        "| Band | items | scam | legitimate |",
        "| --- | ---: | ---: | ---: |",
        f"| Tinggi | {d['Tinggi']['n']} | {d['Tinggi']['scam']} | {d['Tinggi']['legit']} |",
        f"| Sedang | {d['Sedang']['n']} | {d['Sedang']['scam']} | {d['Sedang']['legit']} |",
        f"| Rendah | {d['Rendah']['n']} | {d['Rendah']['scam']} | {d['Rendah']['legit']} |",
        "",
        f"**{d['Rendah']['scam']} scam(s) fall into Rendah** — advertisements the",
        "product would call safe while they are not. That count is the one to watch:",
        "it is the failure mode with a real victim.",
        "",
        "## ⚠️ Caveat",
        "",
        p["caveat"],
        "",
        f"Fitted on `{p['fitted_on']}` — {p['n_items']} items, "
        f"prevalence {p['prevalence']}, model PR-AUC {p['pr_auc']}.",
    ]) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
