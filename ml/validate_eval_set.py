"""Validate the Indonesian evaluation file (single-pass / provenance labels).

This is the revised validator for a dataset that is labelled ONCE per item
(the `label` field), rather than by two independent annotators. The original
version computed Cohen's kappa from `label_a`/`label_b` to measure
inter-annotator agreement; with single-pass labels there is no second opinion
to compare against, so that check has been removed rather than faked.

What it still checks (the parts that remain meaningful):
  - every row is well-formed JSON with the required fields
  - ids are unique
  - no duplicated ad `text`
  - every item actually has a label (0 or 1) — nothing left unlabelled
  - `source_type` / `channel` are within the allowed vocab
  - scam/legit balance is near target (a holdout should lean legitimate so that
    false positives are what gets surfaced)

Honest note printed at runtime: provenance labels are NOT a verified reliability
check. If items were mis-sourced, nothing here will catch it — you are trusting
the sourcing, not two humans agreeing.

    python ml/validate_eval_set.py
    python ml/validate_eval_set.py --path eval/indonesian_holdout.jsonl
    python ml/validate_eval_set.py --target-items 200 --target-scam-rate 0.35
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_HOLDOUT = Path("eval/indonesian_holdout.jsonl")

REQUIRED_FIELDS = ("id", "text", "label", "source_url", "source_type", "channel")

# spec vocab + the local `job_board` extension used for Bosshire/JakartaKerja/Glints boards
SOURCE_TYPES = {
    "bareskrim", "kominfo", "media", "watchdog_account", "community_report",
    "jobstreet", "glints", "kalibrr", "karir_com", "campus_career",
    "company_official", "job_board",
}
CHANNELS = {"whatsapp", "telegram", "instagram", "facebook", "job_board", "other"}

MIN_TEXT_LEN = 30


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"FAILED\nline {lineno}: invalid JSON — {exc}")
    return rows


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


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

    if not args.path.exists():
        print(f"FAILED\nno such file: {args.path}")
        return 1

    rows = load_rows(args.path)

    errors: list[str] = []   # hard problems -> non-zero exit (unless --lenient)
    advice: list[str] = []   # soft nudges

    # --- structural checks ---
    ids = Counter()
    text_sigs: dict[str, str] = {}
    n_scam = n_legit = n_unlabelled = 0

    for i, r in enumerate(rows):
        where = r.get("id", f"row#{i}")
        for field in REQUIRED_FIELDS:
            if field not in r:
                errors.append(f"{where}: missing required field '{field}'")
        ids[r.get("id")] += 1

        label = r.get("label")
        if label == 1:
            n_scam += 1
        elif label == 0:
            n_legit += 1
        else:
            n_unlabelled += 1
            errors.append(f"{where}: label is not 0 or 1 (got {label!r})")

        text = r.get("text") or ""
        if len(text.strip()) < MIN_TEXT_LEN:
            advice.append(f"{where}: text shorter than {MIN_TEXT_LEN} chars")
        sig = norm(text)
        if sig in text_sigs:
            advice.append(f"duplicate text: {where} == {text_sigs[sig]}")
        else:
            text_sigs[sig] = where

        st = r.get("source_type")
        if st is not None and st not in SOURCE_TYPES:
            advice.append(f"{where}: source_type '{st}' not in allowed vocab")
        ch = r.get("channel")
        if ch is not None and ch not in CHANNELS:
            advice.append(f"{where}: channel '{ch}' not in allowed vocab")

    for _id, count in ids.items():
        if count > 1:
            errors.append(f"duplicate id: {_id} appears {count}x")

    n = len(rows)
    scam_rate = (n_scam / n) if n else 0.0

    # --- report ---
    print(f"file          : {args.path}")
    print("provenance    : single-pass (provenance labels) — no inter-annotator agreement computed")
    print(f"items         : {n} / {args.target_items} target")
    print(f"labels        : {n_scam} scam, {n_legit} legitimate ({scam_rate * 100:.1f}% scam)")
    if n_unlabelled:
        print(f"unlabelled    : {n_unlabelled}  <-- must be 0")
    print(f"channels      : {dict(Counter(r.get('channel') for r in rows))}")
    print(f"source types  : {dict(Counter(r.get('source_type') for r in rows))}")

    # --- balance / progress nudges ---
    if n < args.target_items:
        advice.append(f"{args.target_items - n} more items to reach target")
    if n and abs(scam_rate - args.target_scam_rate) > 0.15:
        advice.append(
            f"scam rate {scam_rate * 100:.0f}% is far from the "
            f"{args.target_scam_rate * 100:.0f}% target; a holdout that leans "
            f"legitimate is what surfaces false positives"
        )

    if errors:
        print("\nERRORS:")
        for line in errors:
            print(f"  - {line}")
    if advice:
        print("\nto do:")
        for line in advice:
            print(f"  - {line}")
    if not errors and not advice:
        print("\nlooks good.")

    if errors and not args.lenient:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())