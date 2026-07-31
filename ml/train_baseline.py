"""TF-IDF baselines — MVP_PLAN.md step 1.3.

    python ml/train_baseline.py

Trains classical models on the EMSCAD splits and reports on `val`. Three jobs:

1. **Gate 1.3** — establish a PR-AUC floor (target >= 0.80). If the transformer
   cannot beat this, something is wrong with the transformer, not with the problem.
2. **Fallback** — a shippable model that needs no GPU, in case step 2.1 goes badly.
3. **Ablation row** — the concept paper (section 3.3) promises a comparison between
   classical ML and the transformer. This produces it.

## On SMOTE

Section 3.2 of the paper mentions SMOTE. It is applied HERE and nowhere else.

SMOTE synthesises new minority examples by interpolating between existing feature
vectors. That is meaningful for TF-IDF, where a vector is a point in a continuous
space. It is meaningless for token sequences — you cannot average two sentences into
a valid sentence — so the transformer uses class weighting instead
(MVP_PLAN.md section 1.3). The `--smote` variant here measures whether it helps at
all, so the paper can report a decision rather than an assumption.

## Test discipline

`test` is never touched. Model selection happens on `val`; `test` is opened once, at
step 4.2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from ml.feature_contract import LABEL_COLUMN  # noqa: E402
from ml.metrics import Metrics, evaluate, metrics_table  # noqa: E402

GATE_PR_AUC = 0.80

#: Word 1-2 grams. `min_df=3` drops hapax noise; `sublinear_tf` dampens the effect of
#: a term repeated many times in one long posting, which matters here because
#: document lengths vary by an order of magnitude (p50 242 words, p99 772).
VECTORIZER_KWARGS = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_features=50_000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
)


def load_split(processed: Path, name: str) -> pd.DataFrame:
    path = processed / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python ml/prepare_data.py")
    frame = pd.read_csv(path)
    frame["text"] = frame["text"].fillna("")
    return frame


def build_models(pos_weight: float) -> dict[str, Pipeline]:
    """Each model must expose a continuous score, not just a hard label.

    PR-AUC is computed over the full threshold range, so a bare `predict()` would
    make every model look far worse than it is. LinearSVC has no `predict_proba`,
    so its decision function is calibrated into one.
    """
    models: dict[str, Pipeline] = {
        "tfidf+logreg": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**VECTORIZER_KWARGS)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", C=1.0, solver="liblinear"
                    ),
                ),
            ]
        ),
        "tfidf+linearsvc": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**VECTORIZER_KWARGS)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced", C=0.5, dual=True, max_iter=5000),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        # Gradient boosting on a dense, reduced feature space. HistGradientBoosting
        # is scikit-learn's answer to XGBoost/LightGBM and needs no extra dependency;
        # it cannot consume a 50k-wide sparse matrix, hence the smaller vectorizer.
        "tfidf+histgb": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        **{**VECTORIZER_KWARGS, "max_features": 3_000, "ngram_range": (1, 1)}
                    ),
                ),
                ("densify", _Densify()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.1,
                        class_weight={0: 1.0, 1: pos_weight},
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    return models


class _Densify:
    """Sparse -> dense for estimators that cannot take sparse input."""

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    def fit_transform(self, X, y=None):  # noqa: N803
        return self.transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def build_smote_model() -> Pipeline | None:
    """TF-IDF + SMOTE + logistic regression, if imbalanced-learn is available."""
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError:
        return None

    return ImbPipeline(
        [
            ("tfidf", TfidfVectorizer(**VECTORIZER_KWARGS)),
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
        ]
    )


def score_of(model: Pipeline, texts: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


def top_terms(model: Pipeline, k: int = 20) -> list[tuple[str, float]]:
    """Highest-weighted fraud terms, for the sanity check below."""
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["clf"]
        coefficients = classifier.coef_[0]
    except (AttributeError, KeyError):
        return []
    names = vectorizer.get_feature_names_out()
    order = np.argsort(coefficients)[::-1][:k]
    return [(str(names[i]), float(coefficients[i])) for i in order]


def render_report(results: list[Metrics], terms: list[tuple[str, float]], meta: dict) -> str:
    best = max(results, key=lambda m: m.pr_auc)
    passed = best.pr_auc >= GATE_PR_AUC

    lines = [
        "# TF-IDF baseline results",
        "",
        f"Generated by `ml/train_baseline.py` in {meta['elapsed_s']:.1f}s.",
        f"Evaluated on **val** ({meta['n_val']} rows, {meta['n_val_fraud']} fraud). "
        f"`test` untouched.",
        "",
        "## Gate 1.3",
        "",
        f"Target PR-AUC >= {GATE_PR_AUC:.2f}. Best: **{best.name}** at "
        f"**{best.pr_auc:.4f}** — {'PASS' if passed else 'FAIL'}.",
        "",
        "## Results",
        "",
        metrics_table(results),
        "",
        "Thresholds maximise F1 on the fraud class. The shipped Rendah/Sedang/Tinggi",
        "boundaries are chosen differently — from a precision target (step 3.2) —",
        "because a false positive against a real company costs more than a miss",
        "(concept paper section 3.6).",
        "",
        "## Why accuracy is absent from the table",
        "",
        f"The val split is {meta['n_val_fraud']}/{meta['n_val']} fraud "
        f"({100 * meta['n_val_fraud'] / meta['n_val']:.2f}%). Predicting \"real\" for",
        f"every posting scores **{100 * (1 - meta['n_val_fraud'] / meta['n_val']):.1f}%",
        "accuracy** and catches nothing. For reference:",
        "",
        "| model | accuracy | caught fraud |",
        "| --- | ---: | ---: |",
    ]
    for m in sorted(results, key=lambda m: m.pr_auc, reverse=True):
        lines.append(
            f"| {m.name} | {m.accuracy * 100:.2f}% | {m.true_positives}/{m.n_positive} |"
        )

    if terms:
        lines += [
            "",
            "## Top fraud-weighted terms (tfidf+logreg)",
            "",
            "| term | coefficient |",
            "| --- | ---: |",
        ]
        lines += [f"| `{term}` | {weight:.3f} |" for term, weight in terms]
        lines += [
            "",
            "**Read this list before trusting any number above.** If the model is",
            "keying on a corpus artefact rather than on scam language, it shows up",
            "here first. Ten minutes of reading has saved entire projects.",
            "",
            "### What the first run showed (2026-07-31)",
            "",
            "Mixed, and worth carrying forward.",
            "",
            "**Genuine scam language**, which should transfer: `money`, `from home`,",
            "`earn`, `income`, `high school`, `data entry`, `administrative assistant`,",
            "`clerk`. Low-barrier roles promising easy income — the pattern the concept",
            "paper describes in section 1.1.",
            "",
            "**Corpus artefacts**, which will not transfer:",
            "",
            "| term | train docs | % fraud |",
            "| --- | ---: | ---: |",
            "| `subsea` | 28 | 96.4% |",
            "| `below link` | 27 | 96.3% |",
            "| `apply using` | 23 | 95.7% |",
            "",
            "`subsea` and `offshore` are not properties of job scams; they are one",
            "oil-and-gas scam campaign present in EMSCAD's 2017 collection. `below",
            "link` and `apply using` are one scammer's template phrasing. The model is",
            "partly memorising specific campaigns.",
            "",
            "**Consequence:** expect the zero-shot Indonesian number (step 2.2) to sit",
            "well below this PR-AUC. Memorised 2017 English campaign vocabulary cannot",
            "help on Indonesian WhatsApp ads in 2026. This is a concrete reason to",
            "report the Indonesian figure openly rather than leading with EMSCAD",
            "cross-validation, exactly as section 3.2 commits to.",
        ]

    warnings = [f"- {m.name}: {note}" for m in results for note in m.notes]
    if warnings:
        lines += ["", "## Warnings", ""] + warnings

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("eval/baseline_results.md"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-smote", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()

    train = load_split(args.processed, "train")
    val = load_split(args.processed, "val")

    x_train, y_train = train["text"].tolist(), train[LABEL_COLUMN].to_numpy()
    x_val, y_val = val["text"].tolist(), val[LABEL_COLUMN].to_numpy()

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"train {len(x_train)} rows ({y_train.sum()} fraud), "
          f"val {len(x_val)} rows ({y_val.sum()} fraud), pos_weight {pos_weight:.2f}\n")

    models = build_models(pos_weight)
    if not args.skip_smote:
        smote = build_smote_model()
        if smote is not None:
            models["tfidf+smote+logreg"] = smote
        else:
            print("imbalanced-learn not installed; skipping the SMOTE variant.")
            print("  install with: pip install imbalanced-learn\n")

    results: list[Metrics] = []
    fitted: dict[str, Pipeline] = {}

    for name, model in models.items():
        begin = time.perf_counter()
        model.fit(x_train, y_train)
        metrics = evaluate(name, y_val, score_of(model, x_val))
        results.append(metrics)
        fitted[name] = model
        print(
            f"{name:22} PR-AUC {metrics.pr_auc:.4f}  F1 {metrics.f1:.4f}  "
            f"P {metrics.precision:.4f}  R {metrics.recall:.4f}  "
            f"({time.perf_counter() - begin:.1f}s)"
        )

    best = max(results, key=lambda m: m.pr_auc)
    meta = {
        "elapsed_s": time.perf_counter() - started,
        "n_val": len(x_val),
        "n_val_fraud": int(y_val.sum()),
    }

    terms = top_terms(fitted.get("tfidf+logreg"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(results, terms, meta), encoding="utf-8")

    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "baseline_metrics.json").write_text(
        json.dumps([m.as_dict() for m in results], indent=2), encoding="utf-8"
    )

    print(f"\nbest: {best.name} PR-AUC {best.pr_auc:.4f}")
    print(f"Gate 1.3 (>= {GATE_PR_AUC:.2f}): {'PASS' if best.pr_auc >= GATE_PR_AUC else 'FAIL'}")
    print(f"wrote {args.out}")

    if terms:
        print("\ntop fraud terms — check these are scam language, not artefacts:")
        print("  " + ", ".join(t for t, _ in terms[:12]))

    return 0 if best.pr_auc >= GATE_PR_AUC else 1


if __name__ == "__main__":
    raise SystemExit(main())
