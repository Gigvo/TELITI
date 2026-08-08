"""Cross-language evaluation on the Indonesian holdout — MVP_PLAN.md step 2.2.

    python ml/evaluate_holdout.py

This produces the project's headline number: does a model trained on 2017 English
job-board postings (EMSCAD) actually work on 2026 Indonesian job scams circulating
on WhatsApp and Telegram?

The holdout is never trained on. This is a genuine zero-shot cross-language test.

## Why a length baseline is always reported

An earlier version of the holdout had a severe confound: scam items were forwarded
WhatsApp messages (median 767 chars) while "legitimate" items were one-line job-board
index rows (median 125 chars). Text length alone scored PR-AUC 0.8847 on it — higher
than the trained transformer managed on EMSCAD. Any model score from that file would
have been uninterpretable.

That was fixed by re-collecting the full advertisement text, and length-only dropped
to 0.3734 against a 0.3641 prevalence floor. But the check stays in permanently:
every run reports what length alone achieves, so a future regression in data
collection cannot silently inflate the headline number again.

## What gets compared

- **length only** — the confound check. Should sit near the prevalence floor.
- **rules only** — the deterministic Indonesian layer with no model at all.
- **model only** — the calibrated transformer, zero-shot.
- **model + rules** — the full fused Integrity Score.

The ablation is the point: it shows whether the transformer transfers across
languages, whether the hand-written rules carry their weight, and whether combining
them helps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from api.ingest import ingest  # noqa: E402
from api.locale import LOCALE_ID, load_registry  # noqa: E402
from api.rules.engine import default_engine  # noqa: E402
from api.scoring import compute_score, rule_contributions  # noqa: E402
from ml.calibration import PlattCalibrator, expected_calibration_error  # noqa: E402
from ml.eval_set import SYNTHETIC_BANNER, load_eval_set  # noqa: E402
from ml.metrics import Metrics, evaluate, metrics_table  # noqa: E402

DEFAULT_HOLDOUT = Path("eval/indonesian_holdout.jsonl")
EMSCAD_BASELINE_NOTE = "TF-IDF on EMSCAD val: 0.8769 · transformer on EMSCAD val: 0.8763"


def score_with_model(texts: list[str], model_dir: Path, max_length: int, batch_size: int):
    """Return (calibrated p_scam, raw p_scam). Calibrator applied when available."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    raw, margins = [], []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            batch = tokenizer(
                chunk, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt",
            ).to(device)
            logits = model(**batch).logits.float()
            raw.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            margins.append((logits[:, 1] - logits[:, 0]).cpu().numpy())

    raw = np.concatenate(raw)
    margins = np.concatenate(margins)

    calibrator_path = model_dir.parent / "calibrator.json"
    if calibrator_path.exists():
        calibrator = PlattCalibrator.from_dict(json.loads(calibrator_path.read_text()))
        return calibrator.transform(margins), raw

    print("  ! no calibrator.json found — using uncalibrated probabilities")
    return raw, raw


@contextmanager
def rules_forced_on():
    """Measure the rule layer regardless of whether the product ships it.

    `api/scoring.py::RULE_LAYER_ENABLED` is a PRODUCT decision — currently False,
    because the rules degraded the score on this very holdout. The ABLATION is a
    separate question: "what do the rules do?" Without forcing them on here, the
    ablation rows would just report the consequences of the flag, and the evidence
    justifying that flag would quietly disappear from the report.
    """
    import api.scoring as scoring

    previous = scoring.RULE_LAYER_ENABLED
    scoring.RULE_LAYER_ENABLED = True
    try:
        yield
    finally:
        scoring.RULE_LAYER_ENABLED = previous


def score_with_rules(texts: list[str]) -> tuple[np.ndarray, list[dict]]:
    """Rule layer alone, as a standalone classifier.

    The locale is pinned to Indonesian rather than auto-detected: the corpus language
    is known, and detection on a short fragment would add avoidable noise.
    """
    locale = load_registry().get(LOCALE_ID)
    engine = default_engine(locale)

    shifts, detail = [], []
    with rules_forced_on():
        for text in texts:
            evaluation = engine.evaluate(ingest(text))
            breakdown = compute_score(0.0, evaluation)  # 0.0 => shift only, no model
            shifts.append(breakdown.rule_shift_applied)
            detail.append(
                {
                    "fired": sorted(rule_contributions(evaluation)),
                    "shift": breakdown.rule_shift_applied,
                    "unassessed": list(evaluation.unavailable_features),
                }
            )
    return np.array(shifts), detail


def score_fused(texts: list[str], model_probabilities: np.ndarray) -> np.ndarray:
    locale = load_registry().get(LOCALE_ID)
    engine = default_engine(locale)
    fused = []
    with rules_forced_on():
        for text, probability in zip(texts, model_probabilities):
            evaluation = engine.evaluate(ingest(text))
            fused.append(compute_score(float(probability), evaluation).fused_probability)
    return np.array(fused)


def render_report(
    results: list[Metrics],
    labels: np.ndarray,
    detail: list[dict],
    calibration,
    meta: dict,
    synthetic: bool,
) -> str:
    prevalence = float(labels.mean())
    length_metrics = next((m for m in results if m.name == "length only"), None)
    model_metrics = next((m for m in results if m.name == "model only"), None)

    lines = ["# Indonesian holdout — cross-language evaluation", ""]
    if synthetic:
        lines += [SYNTHETIC_BANNER, ""]

    lines += [
        f"Generated by `ml/evaluate_holdout.py` from `{meta['path']}`.",
        "",
        f"**{meta['n']} items** — {meta['n_scam']} scam, {meta['n_legit']} legitimate "
        f"({prevalence * 100:.1f}% scam).",
        "",
        "This set was never trained on. The transformer was fine-tuned on EMSCAD,",
        "which is English-language job-board text from 2017; this is a genuine",
        "zero-shot cross-language test.",
        "",
        "## Results",
        "",
        metrics_table(results),
        "",
        "## How to read this",
        "",
        f"- **Prevalence floor: {prevalence:.4f}.** A classifier that guesses randomly",
        "  scores this on PR-AUC. Anything at or below it has learned nothing.",
    ]

    if length_metrics is not None:
        gap = length_metrics.pr_auc - prevalence
        verdict = (
            "clean — length carries no usable signal"
            if gap < 0.10
            else "⚠️ CONFOUNDED — length alone is predictive, see below"
        )
        lines += [
            f"- **Length only: {length_metrics.pr_auc:.4f}** ({verdict}).",
            "  This is the data-quality check. If it rises well above the prevalence",
            "  floor, the two classes differ in length for reasons unrelated to fraud",
            "  and every other number here becomes hard to interpret.",
        ]

    if model_metrics is not None and length_metrics is not None:
        lift = model_metrics.pr_auc - length_metrics.pr_auc
        lines.append(
            f"- **Model beyond length: {lift:+.4f}.** This is the honest measure of "
            f"what the transformer contributes over the trivial baseline."
        )

    lines += [
        "",
        f"For reference on the training domain — {EMSCAD_BASELINE_NOTE}.",
        "",
        "## Rule layer behaviour",
        "",
    ]

    scam_mask = labels == 1
    fired_counts = np.array([len(d["fired"]) for d in detail])
    lines += [
        f"- rules fired on scam items: {fired_counts[scam_mask].mean():.2f} on average",
        f"- rules fired on legitimate items: {fired_counts[~scam_mask].mean():.2f} on average",
        f"- legitimate items triggering **any** rule: "
        f"{int((fired_counts[~scam_mask] > 0).sum())} / {int((~scam_mask).sum())}",
        "",
        "That last line is the false-positive count the concept paper (section 3.6)",
        "treats as the expensive error.",
        "",
    ]

    per_rule: dict[str, int] = {}
    for d in detail:
        for name in d["fired"]:
            per_rule[name] = per_rule.get(name, 0) + 1
    if per_rule:
        lines += ["| rule | times fired |", "| --- | ---: |"]
        lines += [f"| `{k}` | {v} |" for k, v in sorted(per_rule.items(), key=lambda kv: -kv[1])]
        lines += [
            "",
            "A rule that never fires contributes nothing on this data; one that fires",
            "on almost everything is not discriminating.",
            "",
        ]

    if calibration is not None:
        lines += [
            "## Calibration on this set",
            "",
            f"- ECE **{calibration.ece:.4f}**, Brier **{calibration.brier:.4f}**",
            "",
            "Calibration was fitted on EMSCAD. Degradation here is expected and worth",
            "reporting: a model can rank correctly across a domain shift while its",
            "probabilities drift.",
            "",
        ]

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/scam_model"))
    parser.add_argument("--out", type=Path, default=Path("eval/indonesian_results.md"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--rules-only", action="store_true",
                        help="Skip the model. Useful before the trained weights arrive.")
    args = parser.parse_args()

    eval_set = load_eval_set(args.path)
    eval_set.require_real("evaluate the holdout", allow_synthetic=args.allow_synthetic)

    texts = eval_set.texts
    labels = np.array(eval_set.labels)
    lengths = np.array([len(t) for t in texts], dtype=float)

    print(f"{len(eval_set)} items — {eval_set.n_scam} scam, {eval_set.n_legit} legitimate")
    print(f"prevalence floor: {labels.mean():.4f}\n")

    results: list[Metrics] = []

    # Data-quality check first: if this is high, nothing below is interpretable.
    results.append(evaluate("length only", labels, lengths))
    print(f"  length only   PR-AUC {results[-1].pr_auc:.4f}")

    rule_shifts, detail = score_with_rules(texts)
    results.append(evaluate("rules only", labels, rule_shifts))
    print(f"  rules only    PR-AUC {results[-1].pr_auc:.4f}")

    calibration = None
    if not args.rules_only:
        if not (args.model_dir / "config.json").exists():
            raise SystemExit(
                f"No model at {args.model_dir}. Copy artifacts/ from the training "
                f"machine, or pass --rules-only to evaluate the rule layer alone."
            )
        print(f"\nscoring with {args.model_dir} ...")
        calibrated, _raw = score_with_model(
            texts, args.model_dir, args.max_length, args.batch_size
        )
        results.append(evaluate("model only", labels, calibrated))
        print(f"  model only    PR-AUC {results[-1].pr_auc:.4f}")

        fused = score_fused(texts, calibrated)
        results.append(evaluate("model + rules", labels, fused))
        print(f"  model + rules PR-AUC {results[-1].pr_auc:.4f}")

        calibration = expected_calibration_error(labels, calibrated)
        print(f"\n  calibration on this set: ECE {calibration.ece:.4f}")

    meta = {
        "path": str(args.path),
        "n": len(eval_set),
        "n_scam": eval_set.n_scam,
        "n_legit": eval_set.n_legit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render_report(results, labels, detail, calibration, meta, eval_set.is_synthetic),
        encoding="utf-8",
    )

    print("\n" + metrics_table(results))
    print(f"\nwrote {args.out}")

    length_pr = results[0].pr_auc
    if length_pr - float(labels.mean()) > 0.10:
        print(
            f"\n⚠️  Length alone scores {length_pr:.4f} against a "
            f"{labels.mean():.4f} floor. The classes differ in length for reasons "
            f"unrelated to fraud, so treat every number above with caution."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
