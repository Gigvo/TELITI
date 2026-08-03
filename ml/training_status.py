"""Check on a running (or finished) training job — MVP_PLAN.md step 2.1.

    python ml/training_status.py

Safe to run at any time: it only reads files, never touches the training process.

Reads `artifacts/scam_model/checkpoint/state.pt`, which the trainer rewrites at every
evaluation. Reports progress, the PR-AUC history, the measured step rate, and an ETA
derived from actual elapsed time rather than the pre-run benchmark — the benchmark was
taken on an idle machine, and anything else running competes for the same cores.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOTAL_STEPS_DEFAULT = 2349


def process_running(pid: int) -> bool | None:
    """True/False, or None if we cannot tell."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return str(pid) in out
    except Exception:
        return None


def human(minutes: float) -> str:
    if minutes < 1:
        return "under a minute"
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/scam_model"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS_DEFAULT)
    parser.add_argument("--pid", type=int, default=0, help="Training PID, if known.")
    args = parser.parse_args()

    import torch  # local import: slow, and irrelevant if the paths are missing

    print("=" * 62)
    print("TELITI — training status")
    print("=" * 62)

    # --- when did it start? ------------------------------------------------
    # The token cache is written once at startup, so its mtime is the start time.
    started_at = None
    cache = args.processed / "token_cache" / "train_256.pt"
    if cache.exists():
        started_at = dt.datetime.fromtimestamp(cache.stat().st_mtime)
        elapsed_min = (dt.datetime.now() - started_at).total_seconds() / 60
        print(f"started      {started_at:%H:%M:%S}  ({human(elapsed_min)} ago)")

    if args.pid:
        alive = process_running(args.pid)
        state = {True: "RUNNING", False: "NOT RUNNING", None: "unknown"}[alive]
        print(f"process      pid {args.pid}: {state}")

    # --- finished? ---------------------------------------------------------
    summary = args.model_dir / "training_summary.json"
    if summary.exists():
        import json

        data = json.loads(summary.read_text(encoding="utf-8"))
        final = data["final"]
        print("\nTRAINING COMPLETE")
        print(f"  PR-AUC    {final['pr_auc']:.4f}")
        print(f"  F1        {final['f1']:.4f}")
        print(f"  precision {final['precision']:.4f}   recall {final['recall']:.4f}")
        print(f"  gate 2.1  {'PASS' if data['gate_passed'] else 'FAIL'} "
              f"(needs >= 0.88; TF-IDF baseline {data['baseline_pr_auc']:.4f})")
        print("\nnext: python ml/calibrate.py")
        return 0

    # --- in progress -------------------------------------------------------
    checkpoint = args.model_dir / "checkpoint" / "state.pt"
    if not checkpoint.exists():
        print("\nNo checkpoint written yet.")
        print("  The first evaluation is at step 400. Until then there is nothing")
        print("  to report — this is normal, not a stall.")
        if started_at:
            print(f"  Elapsed: {human((dt.datetime.now()-started_at).total_seconds()/60)}")
        print("\n  A stall would look like: process gone, or checkpoint mtime")
        print("  frozen for far longer than the gap between evaluations.")
        return 0

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = state["step"]
    saved_at = dt.datetime.fromtimestamp(checkpoint.stat().st_mtime)

    print(f"last saved   {saved_at:%H:%M:%S}")
    pct = 100 * step / args.total_steps
    bar = "#" * int(pct / 4) + "." * (25 - int(pct / 4))
    print(f"progress     [{bar}] {step}/{args.total_steps}  ({pct:.0f}%)")

    if started_at and step:
        # Measured from real elapsed time, so it already includes evaluation
        # pauses and any CPU contention from other work on this machine.
        rate = (saved_at - started_at).total_seconds() / step
        remaining = (args.total_steps - step) * rate / 60
        print(f"rate         {rate:.2f} s/step (measured, incl. eval pauses)")
        print(f"ETA          ~{human(remaining)} remaining "
              f"(~{(dt.datetime.now() + dt.timedelta(minutes=remaining)):%H:%M})")

    print(f"\nbest PR-AUC  {state['best_pr_auc']:.4f}   (gate: 0.88, baseline: 0.8769)")

    history = state.get("history", [])
    if history:
        print("\n  step   PR-AUC      F1   precision  recall")
        for h in history:
            mark = "  <- best" if h["pr_auc"] == state["best_pr_auc"] else ""
            print(f"  {h['step']:5d}  {h['pr_auc']:.4f}  {h['f1']:.4f}     "
                  f"{h['precision']:.4f}  {h['recall']:.4f}{mark}")

    print("\nIf interrupted, resume with:")
    print("  python ml/train_transformer.py --resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
