"""Publish the trained model to a Hugging Face model repository.

Datathon 2026 semifinal requires trained weights to be hosted on Hugging Face and
submitted as a separate link. This uploads them, writes a model card, and prints
the two lines you need afterwards.

    python scripts/publish_to_hf.py --repo <username>/teliti-job-scam-mdistilbert

Dry run first if you want to see what would be uploaded:

    python scripts/publish_to_hf.py --repo <username>/teliti --dry-run

Layout matters. `from_pretrained` requires config.json and the weights at the
repository ROOT, so everything is uploaded flat — including the calibrators and
threshold table, which sit in `artifacts/` locally. api/artifacts.py knows about
both layouts.

The training checkpoint is never uploaded: 1.6 GB of optimiser state that only
exists to resume an interrupted run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "scam_model"

#: (source, name-in-repo). Flat, because from_pretrained needs the weights at the
#: root of the repo.
FILES: list[tuple[Path, str]] = [
    (MODEL_DIR / "config.json", "config.json"),
    (MODEL_DIR / "model.safetensors", "model.safetensors"),
    (MODEL_DIR / "tokenizer.json", "tokenizer.json"),
    (MODEL_DIR / "tokenizer_config.json", "tokenizer_config.json"),
    (MODEL_DIR / "training_summary.json", "training_summary.json"),
    (ARTIFACTS / "calibrator.json", "calibrator.json"),
    (ARTIFACTS / "calibrator_deployment.json", "calibrator_deployment.json"),
    (ARTIFACTS / "thresholds.json", "thresholds.json"),
]

#: Without these the repo is not a usable model.
REQUIRED = {"config.json", "model.safetensors", "tokenizer.json"}


def build_model_card(repo: str) -> str:
    """Write the card from the real metrics, never from memory.

    Everything quantitative here is read out of training_summary.json and the
    evaluation reports. A model card with numbers typed in by hand drifts from
    the artefact it describes, and the drift is invisible.
    """
    summary = json.loads((MODEL_DIR / "training_summary.json").read_text(encoding="utf-8"))
    final = summary["final"]
    config = summary["config"]

    # Held-out test results, when ml/evaluate.py has produced them. Read from the
    # JSON twin rather than the markdown so the card cannot quote a superseded
    # figure: the first version of this card led with the validation number,
    # which the test run later contradicted.
    results_path = ROOT / "eval" / "results.json"
    held_out = None
    if results_path.is_file():
        data = json.loads(results_path.read_text(encoding="utf-8"))
        if data.get("split") == "test":
            held_out = data

    if held_out:
        transformer = held_out["transformer"]
        baseline = held_out["baseline"]
        test_section = f"""### EMSCAD **test** split — the honest number

Scored **once**, on a split untouched by any earlier decision.

| Metric | mDistilBERT | TF-IDF + LinearSVC |
|---|---:|---:|
| **PR-AUC** | **{transformer["pr_auc"]:.4f}** | **{baseline["pr_auc"]:.4f}** |
| ROC-AUC | {transformer["roc_auc"]:.4f} | {baseline["roc_auc"]:.4f} |
| Precision | {transformer["precision"]:.4f} | {baseline["precision"]:.4f} |
| Recall | {transformer["recall"]:.4f} | {baseline["recall"]:.4f} |
| F1 | {transformer["f1"]:.4f} | {baseline["f1"]:.4f} |

**Quote {transformer["pr_auc"]:.4f}, not the validation figure.** The validation
score was {final["pr_auc"]:.4f}, but the checkpoint was *selected* on validation,
which makes that split an optimistic estimate. The transformer drops
{final["pr_auc"] - transformer["pr_auc"]:.4f} PR-AUC from validation to test; the
baseline, which involved no such selection, drops only
{summary["baseline_pr_auc"] - baseline["pr_auc"]:.4f}.

The transformer **does not beat a TF-IDF baseline on English EMSCAD.** It is used
because TF-IDF trained on English cannot process Indonesian at all, and Indonesian
is the deployment domain — not because of this number.

"""
    else:
        test_section = ""

    return f"""---
license: mit
language:
  - id
  - en
base_model: {summary["model"]}
pipeline_tag: text-classification
tags:
  - job-scam-detection
  - fraud-detection
  - indonesian
  - text-classification
---

# TELITI — Job Advertisement Integrity Scoring

Fine-tuned `{summary["model"]}` that scores how likely an Indonesian job
advertisement is to be a scam, so a jobseeker can check it **before** sending a
CV and personal data.

Part of **TELITI** (Teknologi Evaluasi Lowongan dan Integritas). Full system —
API, rule engine, explanations, web app:
<https://github.com/Gigvo/TELITI>

## What is in this repository

| File | Purpose |
|---|---|
| `model.safetensors`, `config.json`, `tokenizer*.json` | The fine-tuned classifier |
| `calibrator_deployment.json` | Platt scaling recalibrated for Indonesian deployment — **use this one** |
| `calibrator.json` | Platt scaling fitted on EMSCAD, kept for reference |
| `thresholds.json` | Risk-label boundaries (Rendah / Sedang / Tinggi) |
| `training_summary.json` | Full training configuration and metrics |

## Results, stated honestly

Two evaluations on two datasets. **The numbers are not comparable to each
other** — PR-AUC scales with how common the positive class is, so the higher
number is not evidence of a better model.

{test_section}### EMSCAD validation (English, {final["n"]:,} items, {final["n_positive"] / final["n"]:.1%} scams)

Reported for completeness. The checkpoint was selected here, so these are
optimistic — see the test figures above.

| Metric | Value |
|---|---|
| PR-AUC | {final["pr_auc"]:.4f} |
| TF-IDF baseline | {summary["baseline_pr_auc"]:.4f} |
| ROC-AUC | {final["roc_auc"]:.4f} |
| Precision | {final["precision"]:.4f} |
| Recall | {final["recall"]:.4f} |
| F1 | {final["f1"]:.4f} |
| Brier score | {final["brier"]:.4f} |

Accuracy is {final["accuracy"]:.1%} and is **misleading** — at
{final["n_positive"] / final["n"]:.1%} prevalence, predicting "not a scam" for
everything scores {1 - final["n_positive"] / final["n"]:.1%}.

### Indonesian holdout (195 items, 36.4% scams)

| Configuration | PR-AUC | False positives |
|---|---|---|
| **model only** | **0.9258** | 5 |
| model + rule layer | 0.8617 | 28 |
| rule layer only | 0.4167 | 93 |

The rule layer made things **worse** and is disabled in the product. Concept
paper §3.6 names false positives against real businesses as the expensive error,
and the rules multiplied them by five.

## Usage

```python
import json
import math

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "{repo}"

tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

# Raw logits are NOT probabilities here. Apply the deployment calibrator.
with open(hf_hub_download(REPO, "calibrator_deployment.json")) as fh:
    calibrator = json.load(fh)

text = (
    "Dibutuhkan admin online, gaji Rp9.000.000, tanpa pengalaman. "
    "Wajib transfer biaya administrasi Rp250.000."
)

with torch.no_grad():
    encoded = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    logits = model(**encoded).logits

# The calibrator is fitted on the LOGIT MARGIN, not on softmax output.
margin = (logits[0, 1] - logits[0, 0]).item()
p_scam = 1 / (1 + math.exp(-(calibrator["a"] * margin + calibrator["b"])))

print(f"p(scam) = {{p_scam:.3f}}")
print(f"integrity score = {{round((1 - p_scam) * 100)}} / 100")
```

Verified output — scam example above scores **0**, and a legitimate posting
scores **91**.

Note `max_length=256`. The model was fine-tuned at
{config["max_length"]}, but the product serves at 256 to keep occlusion-based
explanation within its latency budget; use 256 to reproduce the product's
numbers exactly.

### Calibration is not optional

Raw `softmax(logits)` is **not** a usable probability here. The model was
trained where scams are {final["n_positive"] / final["n"]:.1%} of postings and
deployed where they are ~36%. Without recalibration every Indonesian ad scored
93–100 out of 100 — technically ranked correctly, useless as a number a person
can act on.

`calibrator_deployment.json` corrects that shift. Use it.

## Training

| | |
|---|---|
| Base model | `{summary["model"]}` |
| Max sequence length | {config["max_length"]} |
| Batch size | {config["batch_size"]} |
| Learning rate | {config["lr"]} |
| Epochs | {config["epochs"]} |
| Warmup ratio | {config["warmup_ratio"]} |
| Class weighting | pos_weight {summary["pos_weight"]:.2f} |
| Seed | {config["seed"]} |
| Device | {summary["device"]} |

Trained on [EMSCAD](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
(Employment Scam Aegean Dataset), then recalibrated on a manually annotated
Indonesian holdout.

## Intended use

**For:** helping a jobseeker decide whether to look more carefully at an
advertisement before applying.

**Not for:** automated rejection of postings, moderation without human review, or
any decision about a company or a person taken on the score alone.

## Limitations

- **Trained on English data.** EMSCAD is English; Indonesian performance is
  estimated from a 195-item holdout. Indicative, not established.
- **Did not beat a TF-IDF baseline** on EMSCAD validation.
- **EMSCAD is redacted.** Emails, URLs and phone numbers were replaced with
  placeholders before publication, so the model never learned from real contact
  details — the exact signal that matters most in a real scam advertisement.
- **Not a verdict.** The score is a risk indicator. A high score is not proof of
  fraud and a low score is not a guarantee of safety.
- **Ghost jobs are out of scope.** Detecting advertisements for roles that do not
  exist is a separate problem and is not implemented.

## Ethics

Concept paper §3.6 treats false positives against legitimate businesses as the
expensive error, which is why the rule layer was disabled after measurement
rather than kept because it was built. The product ships a visible disclaimer and
an appeal route; appeals are reviewed by a person and are **not** used to retrain.

## Citation

```bibtex
@misc{{teliti2026,
  title  = {{TELITI: Teknologi Evaluasi Lowongan dan Integritas}},
  author = {{Tim SateManaAjaDah}},
  year   = {{2026}},
  note   = {{Datathon 2026, Ristek Fasilkom UI}},
  url    = {{https://huggingface.co/{repo}}}
}}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. username/teliti-job-scam-mdistilbert")
    parser.add_argument("--private", action="store_true", help="create as private")
    parser.add_argument("--dry-run", action="store_true", help="show the plan, upload nothing")
    args = parser.parse_args()

    print(f"\nPublishing to {args.repo}\n")

    present: list[tuple[Path, str]] = []
    total = 0
    for source, name in FILES:
        if source.is_file():
            size = source.stat().st_size
            total += size
            present.append((source, name))
            print(f"  {name:<32} {size / 1_048_576:>8.1f} MB")
        elif name in REQUIRED:
            print(f"\nERROR: {source} is missing and is required.")
            return 1
        else:
            print(f"  {name:<32} {'MISSING':>8}  (skipped)")

    print(f"\n  {'total':<32} {total / 1_048_576:>8.1f} MB")

    if not (MODEL_DIR / "training_summary.json").is_file():
        print("\nERROR: training_summary.json is missing — the model card is built from it.")
        return 1

    card = build_model_card(args.repo)
    card_path = ROOT / "artifacts" / "README_model_card.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"\nModel card written to {card_path.relative_to(ROOT)} ({len(card):,} chars)")

    if args.dry_run:
        print("\nDry run — nothing uploaded.")
        print("Re-run without --dry-run to publish.\n")
        return 0

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    try:
        whoami = api.whoami()
        print(f"Authenticated as {whoami['name']}")
    except Exception:
        print("\nERROR: not logged in. Run:  hf auth login")
        print("Create a WRITE token at https://huggingface.co/settings/tokens\n")
        return 1

    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"Repository ready: https://huggingface.co/{args.repo}")

    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo,
        commit_message="Add model card",
    )
    print("  uploaded README.md (model card)")

    for source, name in present:
        print(f"  uploading {name} ({source.stat().st_size / 1_048_576:.1f} MB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=name,
            repo_id=args.repo,
            commit_message=f"Add {name}",
        )

    print(f"\nDone: https://huggingface.co/{args.repo}\n")
    print("Two things left:")
    print(f"  1. Set DEFAULT_MODEL_REPO in api/artifacts.py to  {args.repo!r}")
    print("     so a fresh clone downloads the weights automatically.")
    print("  2. Submit that URL as your model link.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
