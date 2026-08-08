# Publishing the model to Hugging Face

Datathon 2026 semifinal, Implementation deliverable:

> If your product uses trained model weights or datasets, these must be hosted on
> Hugging Face and submitted as a separate link.

This is a **model repository**, not a Space. Model and dataset repos are free for
everyone — only Spaces run on compute and require PRO. Nothing here costs money.

**Time: about 20 minutes**, nearly all of it uploading 516 MB.

---

## Step 1 — Log in

```bash
.venv/Scripts/python.exe -m pip install -U "huggingface_hub[cli]"
hf auth login
```

Paste a token from <https://huggingface.co/settings/tokens>. On the token page,
choose the **Write** preset.

Not "Full Access". The script only calls `whoami`, `create_repo` and
`upload_file`, all of which Write covers. Full Access adds billing, org settings,
repo deletion and account settings — none of it used here. The token is stored in
plaintext at `~/.cache/huggingface/token` and will be sitting on the machine while
you record the demo video, so the smaller scope is worth the ten seconds.

**Read-Only** fails at the upload step, after the wait rather than before it.

If you publish under a team **organization** rather than your personal account, a
fine-grained token also needs that organization selected explicitly in its
permissions — account-level Write does not imply it.

---

## Step 2 — Dry run

Check what would be uploaded before sending half a gigabyte:

```bash
.venv/Scripts/python.exe scripts/publish_to_hf.py \
  --repo <your-username>/teliti-job-scam-mdistilbert --dry-run
```

Expect:

```
  config.json                           0.0 MB
  model.safetensors                   516.2 MB
  tokenizer.json                        2.8 MB
  tokenizer_config.json                 0.0 MB
  training_summary.json                 0.0 MB
  calibrator.json                       0.0 MB
  calibrator_deployment.json            0.0 MB
  thresholds.json                       0.0 MB

  total                               519.0 MB
```

519 MB, not 2.1 GB. The 1.6 GB training checkpoint is deliberately excluded —
optimiser and scheduler state that only exists to resume an interrupted run, and
that inference never reads.

The dry run also writes the model card to `artifacts/README_model_card.md`.
**Read it before publishing.** Every number in it is generated from
`training_summary.json` and the evaluation reports rather than typed by hand, so
it cannot drift from the artefact — but you are the one publishing it.

---

## Step 3 — Publish

```bash
.venv/Scripts/python.exe scripts/publish_to_hf.py \
  --repo <your-username>/teliti-job-scam-mdistilbert
```

The script creates the repo, uploads the card first, then the files. `model.safetensors`
is the slow one.

Result: `https://huggingface.co/<your-username>/teliti-job-scam-mdistilbert`

---

## Step 4 — Point the code at it

Open [`api/artifacts.py`](../api/artifacts.py) and set:

```python
DEFAULT_MODEL_REPO = "<your-username>/teliti-job-scam-mdistilbert"
```

This is what makes the submission verifiable. With it set, a judge who clones the
repository and runs the app gets the weights downloaded automatically. Without
it, they get a service that starts, reports `model_loaded: false`, and returns
503 from every analysis — which looks like a broken product rather than a missing
541 MB file.

Verify by hiding your local copy and starting clean:

```bash
mv artifacts/scam_model artifacts/scam_model.bak
.venv/Scripts/python.exe -m uvicorn api.main:app --port 8000
# expect: "No local model at ... — loading <repo> from the Hugging Face Hub"

.venv/Scripts/python.exe scripts/smoke_test.py http://localhost:8000   # 18/18

mv artifacts/scam_model.bak artifacts/scam_model
```

Do this once. It is the only way to know the clone-and-run path actually works,
and it is the path every judge will take.

---

## Step 5 — The dataset (optional but worth it)

The Indonesian holdout is 195 items you annotated yourselves. Hosting it as a
dataset repo strengthens *Reproducibility & Access* and gives you a second link.

```bash
hf repo create <your-username>/teliti-indonesian-holdout --repo-type dataset
hf upload <your-username>/teliti-indonesian-holdout \
  eval/indonesian_holdout.jsonl --repo-type dataset
```

Two things to settle first:

- **Provenance.** If these were scraped, check the source's terms before
  republishing.
- **Personal data.** The postings may contain real company emails and phone
  numbers. Redact them, or keep the dataset private and invite the committee as
  collaborators — the rules explicitly allow that:
  *"If any data cannot be made public, inform the committee so we can be invited
  as collaborators instead."*

EMSCAD does not need hosting. It is public and already cited.

---

## What to submit

| Deliverable | Link |
|---|---|
| Code | `https://github.com/<you>/TELITI` — public after the deadline |
| Model weights | `https://huggingface.co/<you>/teliti-job-scam-mdistilbert` |
| Dataset (optional) | `https://huggingface.co/datasets/<you>/teliti-indonesian-holdout` |

---

## Updating after a change

```bash
.venv/Scripts/python.exe scripts/publish_to_hf.py --repo <same-repo>
```

Re-uploads everything and rewrites the card. Fine for small files; if only the
calibrator changed, upload that one file with `hf upload` instead of re-sending
516 MB.
