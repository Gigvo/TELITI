# GPU handoff — running training on the RTX 5050 machine

Training is currently running on CPU on another machine because this GPU box
wasn't available earlier. This doc gets a GPU run started here instead, which
will be much faster (minutes instead of hours).

**Before you start:** ping whoever is running the CPU job so they can stop it once
your GPU run is confirmed working — otherwise you'll end up with two competing
model checkpoints and it won't be obvious which one is "the" model.

---

## 1. Clone the repo

```bash
git clone https://github.com/Gigvo/TELITI.git
cd TELITI
```

## 2. Python environment

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 3. Install PyTorch — read this before running pip install torch

⚠️ **Do not run a plain `pip install torch`.** The RTX 5050 is a Blackwell-generation
GPU (compute capability **sm_120**). The default PyTorch wheels do not include
sm_120 kernels — the install succeeds, `torch.cuda.is_available()` returns `True`,
and then the **first real GPU operation fails** with something like `no kernel
image is available for execution on the device`. This is a known trap and it wastes
a lot of time if you don't check for it up front.

Install the CUDA 12.8 build explicitly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.44"
```

### Verify before doing anything else

```bash
python -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
a = torch.randn(4096, 4096, device='cuda')
b = torch.randn(4096, 4096, device='cuda')
c = a @ b
torch.cuda.synchronize()
print('matmul OK, result sum:', c.sum().item())
"
```

If this prints `matmul OK` with no error, the GPU is genuinely usable. If it raises
a kernel error, the wheel is wrong — double check you installed from the `cu128`
index and not a cached/default wheel.

## 4. Get the dataset

The raw EMSCAD dataset (~48MB) is **not in git** (gitignored — too large, and it's a
static public dataset, not something that should live in version control).

Two options:

- **Download fresh:** it's the "Real / Fake Job Posting Prediction" dataset on
  Kaggle. Save it as `data/raw/fake_job_postings.csv`.
- **Get it directly:** ask for the file to be sent over (Discord/Drive/USB —
  whatever's easiest) rather than re-downloading. It's the same file either way;
  what matters is it ends up at exactly `data/raw/fake_job_postings.csv`.

Then generate the splits:

```bash
python ml/prepare_data.py
```

This writes `data/processed/{train,val,calib,test}.csv` plus a manifest. Takes a
few seconds — it's just splitting, no training happens here.

**Sanity check:** the script prints row counts for each split. `train` should have
roughly 12,500 rows. If it errors about missing columns, the CSV isn't the right
file — re-download rather than debugging the parser.

## 5. Run training — the actual point of all this

```bash
python ml/train_transformer.py --no-freeze --epochs 3
```

`--no-freeze` matters: the CPU run had to freeze the embedding layer and the bottom
3 transformer layers to make training tractable on a CPU (see
`ml/train_transformer.py` docstring for the reasoning). On a GPU there's no need for
that shortcut — full fine-tuning is both faster and better, so drop the flag.

Expect this to take on the order of **10-20 minutes** on an RTX 5050, versus the
~3+ hours the CPU run needs. Progress prints every 50 steps with a running loss and
ETA. Evaluation happens automatically every 400 steps and the best checkpoint (by
PR-AUC, not loss) is saved to `artifacts/scam_model/`.

**Gate to watch for:** val PR-AUC ≥ 0.88. The TF-IDF baseline (no GPU needed at all)
already gets 0.8769, so the transformer needs to clear that meaningfully to justify
existing. If it lands below 0.88, that's useful information, not a failure to hide —
report the number as-is.

If the run gets interrupted for any reason:

```bash
python ml/train_transformer.py --no-freeze --epochs 3 --resume
```

It picks up from the last checkpoint (saved every 400 steps), not from scratch.

## 6. Check on progress mid-run (from a second terminal)

```bash
python ml/training_status.py --pid <the training process PID>
```

Read-only — safe to run anytime without disturbing the actual training. Shows a
progress bar, PR-AUC history so far, a measured (not estimated) step rate, and an
ETA based on actual elapsed time.

## 7. After training finishes

```bash
python ml/calibrate.py
```

This fits Platt scaling on the `calib` split (a split the model never trained on
*or* was selected against — using `val` for this would produce overconfident
probabilities, since `val` already influenced which checkpoint got saved). Gate:
ECE ≤ 0.05.

## 8. Send the result back

The trained model lives in `artifacts/scam_model/` (a few hundred MB — too big for
a normal git push, and it's gitignored anyway). After training + calibration:

- `artifacts/scam_model/` — the model weights
- `artifacts/calibrator.json` — the Platt scaling parameters
- `artifacts/scam_model/training_summary.json` — the metrics, for the record

Zip `artifacts/` and send it back (Drive/Discord/USB), or push it somewhere both
machines can pull from. Whoever has it next runs `python ml/verify_derivability.py`
and the rest of the pipeline against these artifacts rather than retraining.

---

## Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| `no kernel image is available for execution on the device` | Wrong PyTorch wheel — reinstall from the `cu128` index (step 3) |
| `cuda.is_available()` is `False` | NVIDIA driver issue, unrelated to PyTorch — check `nvidia-smi` works first |
| `data/raw/fake_job_postings.csv not found` | Dataset not downloaded yet, or saved to the wrong path — see step 4 |
| Training loss is `nan` | Extremely unlikely with the defaults, but if it happens, lower `--lr` (default `3e-5`) |
| Out of GPU memory | Lower `--batch-size` (default `16`) |

If something doesn't match this doc, check `MVP_PLAN.md` (Gate 0.2, section on
CUDA/sm_120) and `STATUS.md` for the latest state of the project.
