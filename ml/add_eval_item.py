"""Add an annotated item to the evaluation set — MVP_PLAN.md step 1.5.

Annotation is the critical path, and hand-writing JSONL is slow and error-prone.
This prompts for each field, validates as it goes, and appends a correct line.

    python ml/add_eval_item.py

Paste the advertisement, answer a few questions, done. The id is allocated
automatically and the file is re-validated after every append, so a malformed entry
is caught immediately rather than on the day you need the metric.

To review progress at any time:

    python ml/validate_eval_set.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.eval_set import (  # noqa: E402
    DEFAULT_HOLDOUT,
    VALID_CHANNELS,
    VALID_SOURCE_TYPES,
    EvalSetError,
    annotation_report,
    load_eval_set,
)

END_MARKER = "."


def prompt_text() -> str:
    print("\nPaste the advertisement VERBATIM — keep emoji, typos, ALL-CAPS and line")
    print(f"breaks exactly as they appear. Finish with a line containing only '{END_MARKER}'.\n")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == END_MARKER:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def prompt_choice(label: str, options: set[str], default: str | None = None) -> str:
    ordered = sorted(options)
    print(f"\n{label}:")
    for index, option in enumerate(ordered, start=1):
        marker = "  (default)" if option == default else ""
        print(f"  {index:2}. {option}{marker}")
    while True:
        raw = input("> ").strip()
        if not raw and default:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(ordered):
            return ordered[int(raw) - 1]
        if raw in options:
            return raw
        print(f"  Enter 1-{len(ordered)} or the name itself.")


def prompt_label(who: str) -> int:
    while True:
        raw = input(f"{who}'s label — 1 = scam, 0 = legitimate: ").strip()
        if raw in ("0", "1"):
            return int(raw)
        print("  Enter 0 or 1.")


def prompt_required(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print("  Required.")


def next_id(path: Path) -> str:
    """Allocate the next sequential id, skipping any already used."""
    highest = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item_id = json.loads(line).get("id", "")
            except json.JSONDecodeError:
                continue
            if item_id.startswith("id-holdout-"):
                try:
                    highest = max(highest, int(item_id.rsplit("-", 1)[1]))
                except ValueError:
                    pass
    return f"id-holdout-{highest + 1:04d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_HOLDOUT)
    args = parser.parse_args()

    print("=" * 68)
    print("TELITI — add an evaluation item")
    print("=" * 68)
    print(f"file: {args.path}")

    text = prompt_text()
    if len(text) < 30:
        print(f"\nOnly {len(text)} characters — too short to be a usable item. Aborted.")
        return 1

    print(f"\n{len(text)} characters captured.")

    # Independent labels first, BEFORE any discussion, or Cohen's kappa is
    # meaningless — it measures agreement between independent judgements.
    print("\nBoth annotators label INDEPENDENTLY, before discussing.")
    annotator_a = prompt_required("Annotator A name")
    label_a = prompt_label("Annotator A")
    annotator_b = prompt_required("Annotator B name")
    label_b = prompt_label("Annotator B")

    resolved_by = None
    if label_a != label_b:
        print("\nThe annotators disagree. A third person decides.")
        resolved_by = prompt_required("Resolver name")
        final_label = prompt_label("Resolver")
    else:
        final_label = label_a

    source_url = prompt_required("\nSource URL (required — this is what makes it evidence)")
    source_type = prompt_choice(
        "Source type", VALID_SOURCE_TYPES - {"synthetic"}, default="media"
    )
    channel = prompt_choice("Channel it circulated on", VALID_CHANNELS, default="job_board")

    campaign = input("\nCampaign / company grouping key (optional, Enter to skip): ").strip()
    notes = input("Notes — why you judged it this way (optional): ").strip()

    item = {
        "id": next_id(args.path),
        "text": text,
        "label": final_label,
        "source_url": source_url,
        "source_type": source_type,
        "channel": channel,
        "annotator_a": annotator_a,
        "annotator_b": annotator_b,
        "label_a": label_a,
        "label_b": label_b,
        "resolved_by": resolved_by,
        "collected_at": date.today().isoformat(),
        "campaign": campaign or None,
        "notes": notes,
    }

    args.path.parent.mkdir(parents=True, exist_ok=True)
    existing = args.path.read_text(encoding="utf-8") if args.path.exists() else ""
    separator = "" if not existing or existing.endswith("\n") else "\n"
    with args.path.open("a", encoding="utf-8") as handle:
        handle.write(separator + json.dumps(item, ensure_ascii=False) + "\n")

    # Re-validate the whole file so a bad entry surfaces now, not on the day the
    # metric is needed.
    try:
        eval_set = load_eval_set(args.path)
    except EvalSetError as exc:
        print(f"\nWARNING: the file no longer validates after this append:\n{exc}")
        return 1

    report = annotation_report(eval_set)
    print(f"\nAdded {item['id']}.")
    print(
        f"  {report['n_items']} items — {report['n_scam']} scam, "
        f"{report['n_legit']} legitimate ({report['scam_rate'] * 100:.0f}% scam)"
    )
    kappa = report["cohens_kappa"]
    if kappa == kappa:
        print(f"  Cohen's kappa {kappa:.3f} over {report['n_double_annotated']} items")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"  ! {warning}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAborted; nothing written.")
        raise SystemExit(130) from None
