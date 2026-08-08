"""Locating model artefacts, locally or on the Hugging Face Hub.

The weights are 541 MB and therefore gitignored. Anyone who clones this
repository gets the code and none of the model, and until now that meant the
service started, reported `model_loaded: false`, and returned 503 from every
analysis — a failure that looks like a bug rather than a missing download.

So resolution happens in two steps:

1. **`artifacts/` on disk.** What you have after training locally, and what the
   Docker image bakes in. Always preferred: no network, no surprises.
2. **The Hugging Face Hub.** If the local copy is absent, fetch from
   `TELITI_MODEL_REPO`. Files are cached by `huggingface_hub`, so the download
   happens once per machine rather than once per start.

The point of step 2 is that `git clone && pip install && uvicorn` works for
someone who has never seen this project. That is the difference between a
reviewer verifying the claims and a reviewer filing an issue.

Set `TELITI_MODEL_REPO=""` to disable the Hub entirely — appropriate for an
air-gapped or fully reproducible build where a silent network fetch would be
worse than a loud failure.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("teliti.artifacts")

#: Hugging Face repository holding the trained weights, calibrators and
#: thresholds.
#:
#: NOTE: update this after creating the model repo (scripts/publish_to_hf.py
#: prints the exact value). It is a default rather than a required environment
#: variable so that a fresh clone runs with no configuration at all.
DEFAULT_MODEL_REPO = "Gigvo/teliti-job-scam-mdistilbert"

MODEL_REPO = os.environ.get("TELITI_MODEL_REPO", DEFAULT_MODEL_REPO).strip()

ARTIFACTS_DIR = Path(os.environ.get("TELITI_ARTIFACTS", "artifacts"))
MODEL_DIR = Path(os.environ.get("TELITI_MODEL_DIR", str(ARTIFACTS_DIR / "scam_model")))


def _hub_enabled() -> bool:
    """False when the repo is unset or still the placeholder.

    Downloading from a repository literally named CHANGE-ME would 404 after a
    confusing delay, so treat the placeholder as "not configured".
    """
    return bool(MODEL_REPO) and not MODEL_REPO.startswith("CHANGE-ME")


def resolve_model_dir() -> tuple[str, str]:
    """Return ``(source, origin)`` for ``from_pretrained``.

    ``source`` is either a local path or a Hub repo id — ``from_pretrained``
    accepts both, so callers do not branch. ``origin`` is ``"local"`` or
    ``"hub"``, purely so /health can say which one is in use; a model that
    silently came from somewhere unexpected is a debugging trap.
    """
    if (MODEL_DIR / "config.json").is_file():
        return str(MODEL_DIR), "local"

    if _hub_enabled():
        log.info(
            "No local model at %s — loading %s from the Hugging Face Hub. "
            "The first run downloads ~541 MB and caches it.",
            MODEL_DIR, MODEL_REPO,
        )
        return MODEL_REPO, "hub"

    raise FileNotFoundError(
        f"No model at {MODEL_DIR} and no Hugging Face repo configured. "
        f"Either copy the trained artefacts into {ARTIFACTS_DIR}/, or set "
        f"TELITI_MODEL_REPO to the model repository."
    )


def resolve_model_file(name: str) -> Path | None:
    """Find a file that sits beside the weights, e.g. training_summary.json.

    Locally these live in ``artifacts/scam_model/``; on the Hub they are at the
    repository root, because ``from_pretrained`` requires config.json and the
    weights to be at the root and everything ships together. Same file, two
    layouts — hence a resolver rather than a path constant.
    """
    local = MODEL_DIR / name
    if local.is_file():
        return local

    if not _hub_enabled():
        return None

    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=MODEL_REPO, filename=name))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not fetch %s from %s: %s", name, MODEL_REPO, exc)
        return None


def resolve_file(name: str) -> Path | None:
    """Find a small artefact — a calibrator or the threshold table.

    Returns None rather than raising when the file cannot be found anywhere:
    every caller has a documented degraded mode (an uncalibrated probability, a
    placeholder threshold table) and reports it. A hard failure here would take
    down analysis over a file that only affects presentation.
    """
    local = ARTIFACTS_DIR / name
    if local.is_file():
        return local

    if not _hub_enabled():
        return None

    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=MODEL_REPO, filename=name))
    except Exception as exc:  # noqa: BLE001 - offline, 404, auth: all non-fatal
        log.warning("Could not fetch %s from %s: %s", name, MODEL_REPO, exc)
        return None
