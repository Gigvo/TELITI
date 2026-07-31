"""EMSCAD data preparation — MVP_PLAN.md step 1.2.

Loads `fake_job_postings.csv`, builds one cleaned text document per posting under
the `text_only` feature contract, and writes four stratified, disjoint, reproducible
splits.

    python ml/prepare_data.py

## Why four splits and not three

    train  70%   fits the transformer
    val    10%   selects the checkpoint and the operating threshold
    calib  10%   fits Platt scaling ONLY
    test   10%   touched exactly once, at step 4.2

Calibration must be fitted on data the text model never saw *and* never selected
against. Reusing `val` produces probabilities that look calibrated on `val` and are
overconfident everywhere else, which quietly invalidates the "Integrity Score is a
calibrated probability" claim the whole product rests on.

Note the fusion meta-model does NOT train on `calib`, unlike the concept paper's
design. Measurement showed EMSCAD cannot teach rule weights (see
`eval/derivability_report.md`), so fusion is fitted on Indonesian data instead —
`ml/feature_contract.py::FUSION_TRAINING_SOURCE`.

## Reproducibility

`split_manifest.json` records the seed, the source file's SHA-256, the cleaning
version, and a checksum of each split's `job_id` list. Two people running this
command get byte-identical splits, and any model artefact can be traced to the exact
rows and the exact cleaning that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python ml/prepare_data.py` as well as `python -m ml.prepare_data`.
# Running a script by path puts ml/ on sys.path, not the repo root, so the
# absolute `ml.*` imports below would fail. pytest injects this via pyproject.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from ml.feature_contract import (
    BOOKKEEPING_COLUMNS,
    EMSCAD_COLUMNS,
    EXPECTED_FRAUD_COUNT,
    EXPECTED_ROW_COUNT,
    LABEL_COLUMN,
    PROFILE_TEXT_ONLY,
    PROFILES,
    assert_no_forbidden_columns,
    assert_valid_profile,
    text_document_columns,
)
from ml.text_cleaning import CLEANING_VERSION, build_document

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.10, "calib": 0.10, "test": 0.10}

#: Documents shorter than this carry no usable signal — a title and nothing else.
MIN_DOCUMENT_WORDS = 5


# ---------------------------------------------------------------------------
# Gate 0.3
# ---------------------------------------------------------------------------


def validate_source(df: pd.DataFrame, *, strict: bool = True) -> list[str]:
    """Confirm this is really EMSCAD and not some other job-postings CSV."""
    problems: list[str] = []

    if tuple(df.columns) != EMSCAD_COLUMNS:
        missing = sorted(set(EMSCAD_COLUMNS) - set(df.columns))
        extra = sorted(set(df.columns) - set(EMSCAD_COLUMNS))
        problems.append(f"column mismatch (missing={missing}, unexpected={extra})")
    if len(df) != EXPECTED_ROW_COUNT:
        problems.append(f"expected {EXPECTED_ROW_COUNT} rows, found {len(df)}")
    if LABEL_COLUMN in df:
        fraud = int(df[LABEL_COLUMN].sum())
        if fraud != EXPECTED_FRAUD_COUNT:
            problems.append(f"expected {EXPECTED_FRAUD_COUNT} fraud rows, found {fraud}")
        if not df[LABEL_COLUMN].isin([0, 1]).all():
            problems.append("label column contains values outside {0, 1}")

    if problems and strict:
        raise ValueError(
            "Source file failed Gate 0.3:\n  - " + "\n  - ".join(problems)
        )
    return problems


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_checksum(job_ids: pd.Series) -> str:
    joined = ",".join(str(v) for v in sorted(job_ids.tolist()))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def build_frame(df: pd.DataFrame, profile: str, *, desegment: bool) -> pd.DataFrame:
    """Produce the modelling frame: identifier, document, label. Nothing else."""
    assert_valid_profile(profile)
    fields = text_document_columns(profile)

    # The contract guard. Everything the model sees must be recoverable from a
    # user's pasted text at inference time; see MVP_PLAN.md section 1.1.
    assert_no_forbidden_columns(fields, profile)

    source = df[list(fields)].fillna("")
    documents = source.apply(
        lambda row: build_document(row.to_dict(), fields, apply_desegment=desegment), axis=1
    )

    frame = pd.DataFrame(
        {
            "job_id": df["job_id"].to_numpy(),
            "text": documents.to_numpy(),
            LABEL_COLUMN: df[LABEL_COLUMN].astype(int).to_numpy(),
        }
    )
    frame["n_words"] = frame["text"].str.split().str.len().fillna(0).astype(int)

    # Final assertion: no forbidden column reached the frame by any route.
    # Bookkeeping columns are excluded — `job_id` is an identifier for split
    # tracking, not a model input, and the guard is about what the MODEL sees.
    assert_no_forbidden_columns(set(frame.columns) - BOOKKEEPING_COLUMNS, profile)
    return frame


def drop_degenerate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = frame["n_words"] >= MIN_DOCUMENT_WORDS
    return frame[keep].reset_index(drop=True), frame[~keep].reset_index(drop=True)


def stratified_splits(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """70/10/10/10, stratified on the label, via two nested splits."""
    labels = frame[LABEL_COLUMN]

    train, rest = train_test_split(
        frame, train_size=SPLIT_FRACTIONS["train"], stratify=labels, random_state=seed
    )
    # `rest` is 30%: val takes a third of it (10% overall), the remainder splits evenly.
    val, calib_test = train_test_split(
        rest,
        train_size=SPLIT_FRACTIONS["val"] / (1 - SPLIT_FRACTIONS["train"]),
        stratify=rest[LABEL_COLUMN],
        random_state=seed,
    )
    calib, test = train_test_split(
        calib_test, train_size=0.5, stratify=calib_test[LABEL_COLUMN], random_state=seed
    )

    return {
        name: part.sort_values("job_id").reset_index(drop=True)
        for name, part in (("train", train), ("val", val), ("calib", calib), ("test", test))
    }


def assert_splits_sound(splits: dict[str, pd.DataFrame], total: int, base_rate: float) -> None:
    seen: dict[int, str] = {}
    for name, part in splits.items():
        for job_id in part["job_id"]:
            if job_id in seen:
                raise AssertionError(
                    f"job_id {job_id} appears in both {seen[job_id]} and {name}. "
                    f"Overlapping splits invalidate every metric downstream."
                )
            seen[job_id] = name

    if len(seen) != total:
        raise AssertionError(f"splits cover {len(seen)} rows, expected {total}")

    for name, part in splits.items():
        rate = part[LABEL_COLUMN].mean()
        if abs(rate - base_rate) > 0.005:
            raise AssertionError(
                f"{name} fraud rate {rate:.4f} deviates from base {base_rate:.4f} "
                f"by more than 0.5pp; stratification failed."
            )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_report(
    splits: dict[str, pd.DataFrame], dropped: pd.DataFrame, manifest: dict
) -> str:
    lines = [
        "# EMSCAD preparation report",
        "",
        f"Generated {manifest['generated_at']} by `ml/prepare_data.py`.",
        "",
        f"- profile: `{manifest['profile']}`",
        f"- fields: `{'`, `'.join(manifest['text_fields'])}`",
        f"- seed: `{manifest['seed']}`  |  cleaning: `{manifest['cleaning_version']}`"
        f"  |  desegment: `{manifest['desegment']}`",
        f"- source sha256: `{manifest['source_sha256'][:16]}...`",
        "",
        "## Splits",
        "",
        "| split | rows | fraud | fraud % | median words | p95 words | checksum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, part in splits.items():
        lines.append(
            f"| {name} | {len(part)} | {int(part[LABEL_COLUMN].sum())} | "
            f"{part[LABEL_COLUMN].mean() * 100:.2f}% | "
            f"{int(part['n_words'].median())} | {int(part['n_words'].quantile(0.95))} | "
            f"`{manifest['splits'][name]['checksum']}` |"
        )

    words = pd.concat([p["n_words"] for p in splits.values()])
    lines += [
        "",
        f"Dropped as degenerate (<{MIN_DOCUMENT_WORDS} words): **{len(dropped)}**",
        "",
        "## Document length",
        "",
        "| percentile | words | est. subword tokens |",
        "| --- | ---: | ---: |",
    ]
    for q in (0.50, 0.75, 0.90, 0.95, 0.99):
        w = int(words.quantile(q))
        lines.append(f"| p{int(q * 100)} | {w} | ~{int(w * 1.35)} |")

    lines += [
        "",
        "Subword estimate uses a 1.35 tokens/word ratio for multilingual WordPiece on",
        "English. **Measure it properly on the training box** before fixing",
        "`max_length` — this estimate is not a substitute (step 2.1).",
        "",
        "## Class imbalance",
        "",
        f"Base fraud rate **{manifest['base_fraud_rate'] * 100:.2f}%** "
        f"(pos_weight ≈ {manifest['pos_weight']:.1f}).",
        "",
        "Accuracy is meaningless at this ratio — a model predicting \"all real\" scores",
        f"{(1 - manifest['base_fraud_rate']) * 100:.1f}%. Report PR-AUC and F1 on the",
        "fraud class instead.",
        "",
        "Use `pos_weight` in a weighted loss for the transformer. Do NOT use SMOTE",
        "there: it interpolates feature vectors, which is meaningful for TF-IDF and",
        "meaningless for token sequences (MVP_PLAN.md section 1.3).",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EMSCAD splits for TELITI.")
    parser.add_argument("--csv", type=Path, default=Path("data/raw/fake_job_postings.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE_TEXT_ONLY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-desegment",
        action="store_true",
        help="Skip glued-word repair. 78%% of documents are affected; see ml/text_cleaning.py.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"{args.csv} not found. Download EMSCAD ('Real / Fake Job Posting "
            f"Prediction' on Kaggle) and place fake_job_postings.csv there."
        )

    print(f"reading {args.csv} ...")
    raw = pd.read_csv(args.csv)
    validate_source(raw)
    print(f"  Gate 0.3 OK: {raw.shape}, {int(raw[LABEL_COLUMN].sum())} fraud")

    frame = build_frame(raw, args.profile, desegment=not args.no_desegment)
    frame, dropped = drop_degenerate(frame)
    if len(dropped):
        print(f"  dropped {len(dropped)} degenerate rows (<{MIN_DOCUMENT_WORDS} words)")

    base_rate = float(frame[LABEL_COLUMN].mean())
    splits = stratified_splits(frame, args.seed)
    assert_splits_sound(splits, len(frame), base_rate)
    print("  splits disjoint and stratified")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": args.profile,
        "text_fields": list(text_document_columns(args.profile)),
        "seed": args.seed,
        "cleaning_version": CLEANING_VERSION,
        "desegment": not args.no_desegment,
        "source_csv": str(args.csv),
        "source_sha256": file_sha256(args.csv),
        "rows_used": len(frame),
        "rows_dropped": len(dropped),
        "base_fraud_rate": base_rate,
        "pos_weight": (1 - base_rate) / base_rate,
        "splits": {},
    }

    for name, part in splits.items():
        path = args.out / f"{name}.csv"
        part.to_csv(path, index=False)
        manifest["splits"][name] = {
            "path": str(path),
            "rows": len(part),
            "fraud": int(part[LABEL_COLUMN].sum()),
            "fraud_rate": float(part[LABEL_COLUMN].mean()),
            "checksum": split_checksum(part["job_id"]),
        }
        print(f"  wrote {path}  ({len(part)} rows, {int(part[LABEL_COLUMN].sum())} fraud)")

    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "prepare_report.md").write_text(
        render_report(splits, dropped, manifest), encoding="utf-8"
    )
    print(f"  wrote {args.out / 'split_manifest.json'}")
    print(f"  wrote {args.out / 'prepare_report.md'}")
    print(f"\npos_weight for the weighted loss: {manifest['pos_weight']:.2f}")


if __name__ == "__main__":
    main()
