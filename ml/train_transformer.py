"""Fine-tune the scam classifier — MVP_PLAN.md step 2.1.

    python ml/train_transformer.py

Trains `distilbert-base-multilingual-cased` on the EMSCAD splits produced by
`ml/prepare_data.py`, and writes the best checkpoint to `artifacts/scam_model/`.

Gate: val PR-AUC >= 0.88 (the TF-IDF baseline reached 0.8769, so the transformer
must beat a model that needs no GPU at all to justify itself).

## Why a manual loop instead of `Trainer`

Three things here are non-standard and easier to get right explicitly: a weighted
loss over an imbalanced binary task, checkpoint selection by PR-AUC rather than
loss, and resumability tuned for a multi-hour CPU run. The HF `Trainer` can do all
of it, but through configuration indirection that is harder to audit.

## CPU adaptations

This was written to run on the machine that serves the API — no CUDA device. Two
compromises, both measured before being chosen:

- **Frozen embeddings + bottom 3 transformer layers.** The multilingual embedding
  matrix is 92M of the model's 135M parameters, so freezing it removes most of the
  backward pass. Measured 7.44 s/step -> 4.45 s/step. Lower layers encode general
  syntax; the task-specific work happens in the upper layers, so this costs little
  on a binary classification task with 12.5k examples.
- **max_length 256.** Covers roughly the first 190 words. Documents are longer
  (median 242 words) but truncation keeps the HEAD, where the title, the salary
  promise and the opening pitch live — the TF-IDF baseline's most predictive terms
  (`data entry`, `money`, `from home`, `earn`) all appear early.

The script auto-detects CUDA and uses it when present, so the same command produces
a better model on a GPU box with `--no-freeze --epochs 3` and no code change.

## Resumability

A multi-hour CPU run must survive a crash or a closed laptop lid. Model, optimizer
and scheduler state are written to `artifacts/scam_model/checkpoint/` at every
evaluation; `--resume` picks up from there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from ml.feature_contract import LABEL_COLUMN  # noqa: E402
from ml.metrics import evaluate, metrics_table  # noqa: E402

DEFAULT_MODEL = "distilbert-base-multilingual-cased"
GATE_PR_AUC = 0.88
BASELINE_PR_AUC = 0.8769  # eval/baseline_results.md — what we must beat


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_split(processed: Path, name: str) -> pd.DataFrame:
    path = processed / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python ml/prepare_data.py")
    frame = pd.read_csv(path)
    frame["text"] = frame["text"].fillna("")
    return frame


def encode(tokenizer, frame: pd.DataFrame, max_length: int, cache: Path | None = None):
    """Tokenise once. Cached so a resumed run does not repeat the work."""
    if cache is not None and cache.exists():
        blob = torch.load(cache)
        if blob["max_length"] == max_length and blob["n"] == len(frame):
            return TensorDataset(blob["ids"], blob["mask"], blob["labels"])

    encoded = tokenizer(
        frame["text"].tolist(),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = torch.tensor(frame[LABEL_COLUMN].to_numpy(), dtype=torch.long)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ids": encoded["input_ids"],
                "mask": encoded["attention_mask"],
                "labels": labels,
                "max_length": max_length,
                "n": len(frame),
            },
            cache,
        )
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def apply_freezing(model, n_frozen_layers: int, freeze_embeddings: bool) -> tuple[int, int]:
    """Freeze the lower stack. Returns (trainable, total) parameter counts."""
    if freeze_embeddings:
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False

    for index, layer in enumerate(model.distilbert.transformer.layer):
        if index < n_frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, p_fraud) over a loader."""
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for ids, mask, y in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        probabilities.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/scam_model"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=400, help="Steps between evaluations.")
    parser.add_argument("--freeze-layers", type=int, default=3)
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Full fine-tune. ~1.7x slower on CPU; use on a GPU box.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0, help="0 = all logical cores.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full run. For smoke tests.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads or os.cpu_count() or 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  threads: {torch.get_num_threads()}")

    from transformers import (  # imported late: heavy, and only needed here
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    train_df = load_split(args.processed, "train")
    val_df = load_split(args.processed, "val")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cache_dir = args.processed / "token_cache"
    train_ds = encode(tokenizer, train_df, args.max_length, cache_dir / f"train_{args.max_length}.pt")
    val_ds = encode(tokenizer, val_df, args.max_length, cache_dir / f"val_{args.max_length}.pt")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    trainable, total = apply_freezing(
        model, 0 if args.no_freeze else args.freeze_layers, not args.no_freeze
    )
    model.to(device)
    print(f"params: {trainable/1e6:.1f}M trainable of {total/1e6:.1f}M")

    # Weighted cross-entropy, NOT SMOTE. SMOTE interpolates feature vectors, which
    # is meaningful for TF-IDF and undefined for token sequences — you cannot
    # average two sentences into a valid sentence. See MVP_PLAN.md section 1.3.
    n_pos = int(train_df[LABEL_COLUMN].sum())
    pos_weight = (len(train_df) - n_pos) / max(n_pos, 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"train {len(train_df)} ({n_pos} fraud)  val {len(val_df)}  pos_weight {pos_weight:.2f}")

    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out / "checkpoint"
    best_pr_auc, start_step = 0.0, 0
    history: list[dict] = []

    if args.resume and (checkpoint_dir / "state.pt").exists():
        state = torch.load(checkpoint_dir / "state.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step, best_pr_auc = state["step"], state["best_pr_auc"]
        history = state.get("history", [])
        print(f"resumed at step {start_step}, best PR-AUC {best_pr_auc:.4f}")

    print(
        f"\n{total_steps} steps ({steps_per_epoch}/epoch), evaluating every "
        f"{args.eval_every}\n"
    )

    def run_eval(step: int) -> float:
        nonlocal best_pr_auc
        began = time.perf_counter()
        y_true, y_score = predict(model, val_loader, device)
        metrics = evaluate(f"step-{step}", y_true, y_score)
        history.append({"step": step, **metrics.as_dict()})

        flag = ""
        if metrics.pr_auc > best_pr_auc:
            best_pr_auc = metrics.pr_auc
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            flag = "  <- best, saved"

        print(
            f"  eval @{step:5d}  PR-AUC {metrics.pr_auc:.4f}  F1 {metrics.f1:.4f}  "
            f"P {metrics.precision:.4f}  R {metrics.recall:.4f}  "
            f"({time.perf_counter()-began:.0f}s){flag}",
            flush=True,
        )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_pr_auc": best_pr_auc,
                "history": history,
            },
            checkpoint_dir / "state.pt",
        )
        model.train()
        return metrics.pr_auc

    model.train()
    step = start_step
    started = time.perf_counter()
    running_loss, loss_count = 0.0, 0

    while step < total_steps:
        for ids, mask, labels in train_loader:
            if step >= total_steps:
                break

            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            loss_count += 1
            step += 1

            if step % 50 == 0:
                elapsed = time.perf_counter() - started
                done = step - start_step
                eta = (total_steps - step) * elapsed / max(done, 1) / 60
                print(
                    f"  step {step:5d}/{total_steps}  loss {running_loss/loss_count:.4f}  "
                    f"{elapsed/done:.2f}s/step  ETA {eta:.0f} min",
                    flush=True,
                )
                running_loss, loss_count = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                run_eval(step)

    if not history or history[-1]["step"] != step:
        run_eval(step)

    # ---- final report -----------------------------------------------------
    print(f"\ntraining finished in {(time.perf_counter()-started)/60:.0f} min")

    model = AutoModelForSequenceClassification.from_pretrained(args.out).to(device)
    y_true, y_score = predict(model, val_loader, device)
    final = evaluate("mdistilbert (best checkpoint)", y_true, y_score)

    print("\n" + metrics_table([final]))
    passed = final.pr_auc >= GATE_PR_AUC
    print(f"\nGate 2.1 (val PR-AUC >= {GATE_PR_AUC}): {'PASS' if passed else 'FAIL'}")
    print(f"TF-IDF baseline was {BASELINE_PR_AUC:.4f} -> {final.pr_auc - BASELINE_PR_AUC:+.4f}")

    (args.out / "training_summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "config": {
                    k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                },
                "device": str(device),
                "pos_weight": pos_weight,
                "final": final.as_dict(),
                "history": history,
                "gate_passed": bool(passed),
                "baseline_pr_auc": BASELINE_PR_AUC,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out / 'training_summary.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
