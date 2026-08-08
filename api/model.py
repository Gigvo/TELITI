"""Text-model inference for serving — MVP_PLAN.md step 2.5.

Loads the fine-tuned transformer and the deployment calibrator once at start-up, and
scores one advertisement per request.

## Loading is lazy and failure is survivable

The model is ~540MB and takes a few seconds to load. It is loaded on first use rather
than at import, so the test suite and any tooling that only touches the rule layer
does not pay for it.

If the artefacts are missing or fail to load, the service does NOT crash — it falls
back to reporting itself unloaded, and `/health` shows `model_loaded: false`. A demo
that starts and honestly says "no model" is far better than one that will not start
at all, and the failure is visible rather than silent.

## Which calibrator

Two exist, and the distinction matters:

- `calibrator.json` — fitted on the EMSCAD `calib` split. Correct for English
  job-board text resembling the training distribution.
- `calibrator_deployment.json` — refitted on Indonesian advertisements. Correct for
  what this product actually receives.

The deployment calibrator is preferred when present. Without it, the model's
probabilities carry EMSCAD's 4.8% base rate into a domain where scams are far more
common: every score lands between 93 and 100, and a scam displays as "98/100". See
`ml/fit_thresholds.py` for the measurements.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("teliti.model")

MODEL_DIR = Path(os.environ.get("TELITI_MODEL_DIR", "artifacts/scam_model"))
DEPLOYMENT_CALIBRATOR = Path(
    os.environ.get("TELITI_CALIBRATOR", "artifacts/calibrator_deployment.json")
)
FALLBACK_CALIBRATOR = Path("artifacts/calibrator.json")

#: Must match the `max_length` the model was fine-tuned with, or the input
#: distribution at serving time differs from training in a way nothing will flag.
MAX_LENGTH = int(os.environ.get("TELITI_MAX_LENGTH", "256"))


@dataclass(frozen=True)
class ModelInfo:
    loaded: bool
    version: str
    calibrator: str
    max_length: int
    device: str
    error: str | None = None


class ScamModel:
    """Thread-safe lazy wrapper around the transformer and its calibrator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._calibrator = None
        self._info: ModelInfo | None = None
        self._attempted = False

    # -- loading ------------------------------------------------------------

    def _load(self) -> None:
        """Load artefacts. Records the failure rather than raising."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from ml.calibration import PlattCalibrator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(os.cpu_count() or 4)

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
        )

        calibrator_path, calibrator_name = None, "none"
        if DEPLOYMENT_CALIBRATOR.is_file():
            calibrator_path, calibrator_name = DEPLOYMENT_CALIBRATOR, "deployment"
        elif FALLBACK_CALIBRATOR.is_file():
            calibrator_path, calibrator_name = FALLBACK_CALIBRATOR, "emscad"
            log.warning(
                "Using the EMSCAD calibrator: scores will cluster near 100 on "
                "Indonesian input. Run `python ml/fit_thresholds.py` to produce %s.",
                DEPLOYMENT_CALIBRATOR,
            )

        if calibrator_path is not None:
            self._calibrator = PlattCalibrator.from_dict(
                json.loads(calibrator_path.read_text(encoding="utf-8"))
            )
        else:
            log.warning("No calibrator found — probabilities will be uncalibrated.")

        version = f"mdistilbert-{MAX_LENGTH}"
        summary = MODEL_DIR / "training_summary.json"
        if summary.is_file():
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
                pr_auc = data.get("final", {}).get("pr_auc")
                if pr_auc is not None:
                    version = f"mdistilbert-{MAX_LENGTH}-pr{pr_auc:.4f}"
            except (json.JSONDecodeError, TypeError):
                pass

        self._info = ModelInfo(
            loaded=True,
            version=version,
            calibrator=calibrator_name,
            max_length=MAX_LENGTH,
            device=device,
        )
        log.info("Loaded %s on %s (calibrator: %s)", version, device, calibrator_name)

    def _ensure(self) -> None:
        if self._attempted:
            return
        with self._lock:
            if self._attempted:  # another thread won the race
                return
            self._attempted = True
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
                log.exception("Failed to load the text model from %s", MODEL_DIR)
                self._info = ModelInfo(
                    loaded=False, version="unloaded", calibrator="none",
                    max_length=MAX_LENGTH, device="none", error=f"{type(exc).__name__}: {exc}",
                )

    # -- inference ----------------------------------------------------------

    @property
    def info(self) -> ModelInfo:
        self._ensure()
        assert self._info is not None
        return self._info

    @property
    def is_loaded(self) -> bool:
        return self.info.loaded

    def probability(self, text: str) -> float:
        """Calibrated p(scam) for one advertisement."""
        return self.probabilities([text])[0]

    def margins(self, texts: list[str]) -> list[float]:
        """Raw logit margins (logit_scam − logit_real), before calibration.

        Unbounded, unlike a probability. That matters for explanation: when the model
        is confident the probability saturates near 1.0, and removing a sentence moves
        it by ~0.001 even when the effect is large. The margin keeps moving, so
        `api/explain.py` measures influence here rather than in probability space.
        """
        self._ensure()
        if not self.info.loaded:
            raise RuntimeError("Text model is not loaded.")
        if not texts:
            return []

        import torch

        batch = self._tokenizer(
            texts, truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            logits = self._model(**batch).logits.float()

        return [float(m) for m in (logits[:, 1] - logits[:, 0]).cpu().numpy()]

    def calibrate_margin(self, margin: float) -> float:
        """Turn one logit margin into a calibrated probability.

        Lets a caller reuse a margin it already has — the analyse endpoint needs both
        the margin (for occlusion) and the probability (for the score), and computing
        them from one forward pass rather than two halves the work.
        """
        self._ensure()
        if self._calibrator is not None:
            return float(self._calibrator.transform([margin])[0])
        import math

        return 1.0 / (1.0 + math.exp(-margin))

    def probabilities(self, texts: list[str]) -> list[float]:
        """Calibrated p(scam) for several advertisements in one forward pass.

        Batched because the explanation module scores one variant per sentence.
        Twenty separate calls would pay the per-call overhead twenty times; one batch
        of twenty does not.
        """
        if not texts:
            return []

        import numpy as np

        margins = np.asarray(self.margins(texts))
        # The calibrator was fitted on the logit MARGIN, not on a softmax output —
        # applying it to a probability would squash through a sigmoid twice.
        if self._calibrator is not None:
            return [float(p) for p in self._calibrator.transform(margins)]
        return [float(p) for p in 1.0 / (1.0 + np.exp(-margins))]


#: Process-wide instance. Loading is lazy, so importing this is cheap.
scam_model = ScamModel()
