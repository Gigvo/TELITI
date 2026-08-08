"""Appeal and label-correction intake — concept paper §3.6.

§3.6 commits to three things. Two were already true: the score is presented as a
risk indicator rather than a verdict, and analysis persists nothing. This module
supplies the third — *"terdapat mekanisme banding/koreksi label"* — a route for
someone to say the system got it wrong.

The case that matters is a legitimate company scored *Tinggi*. §3.6 names false
positives against real businesses as the error to suppress, and a system that can
be wrong about a company with no way to say so is worse than one that admits it.

## The privacy tension, and how it is resolved

§3.6 also promises *"tidak ada data pribadi pengguna yang disimpan dari lowongan
yang dianalisis"*. An appeal is worthless unless the disputed advertisement is
stored, so the two commitments appear to collide.

They do not, because the distinction is CONSENT:

- **Analysis** stores nothing. Unchanged. A user who pastes an advertisement leaves
  no trace.
- **A report** stores only what the person deliberately submits, having chosen to
  file it. Contact details are optional and requested only for follow-up.

That is a different act, and the promise in §3.6 refers to the first.

## ⚠️ Reports are NOT training data

Every report lands in a quarantine file and stops there. Nothing here feeds the
model, the thresholds or the rule lexicons.

The paper's own Tahap 3 explains why: crowd-sourced labels may only enter training
after inter-reporter agreement and reviewer confirmation, *"guna mencegah peracunan
data (data poisoning)"*. An endpoint that accepted a label and retrained on it would
be an invitation to move any score in either direction — including a scammer
reporting their own advertisement as legitimate.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPORTS_PATH = Path(os.environ.get("TELITI_REPORTS", "data/reports/corrections.jsonl"))

#: Enough for context; short of an essay. Long submissions are almost always pasted
#: logs rather than explanation.
MAX_COMMENT_LENGTH = 2000
MAX_CONTACT_LENGTH = 200

#: Serialises appends. uvicorn runs handlers in a threadpool, and two interleaved
#: writes would produce a corrupt line that breaks the whole JSONL file on read.
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class StoredReport:
    report_id: str
    received_at: str
    correction: str
    text: str
    reported_score: int | None = None
    reported_label: str | None = None
    request_id: str | None = None
    #: Which model produced the disputed score. Without it a report cannot be
    #: interpreted later — a complaint about a model two retrains ago may say
    #: nothing about the one currently deployed.
    model_version: str | None = None
    comment: str = ""
    contact: str | None = None
    #: Never set by the API. A human sets it after review, so the file records what
    #: was decided rather than only what was claimed.
    reviewed: bool = False
    review_note: str = ""
    tags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def store_report(
    *,
    correction: str,
    text: str,
    reported_score: int | None = None,
    reported_label: str | None = None,
    request_id: str | None = None,
    model_version: str | None = None,
    comment: str = "",
    contact: str | None = None,
    path: Path | None = None,
) -> StoredReport:
    """Append one report to the quarantine file. Returns the stored record.

    Append-only JSONL rather than a database: it is auditable by hand, survives
    without a running service, and a corrupted line loses one report instead of all
    of them.
    """
    target = path or REPORTS_PATH

    report = StoredReport(
        report_id=f"rep-{uuid.uuid4().hex[:12]}",
        received_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        correction=correction,
        text=text,
        reported_score=reported_score,
        reported_label=reported_label,
        request_id=request_id,
        model_version=model_version,
        comment=comment[:MAX_COMMENT_LENGTH],
        contact=(contact or "").strip()[:MAX_CONTACT_LENGTH] or None,
    )

    with _WRITE_LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(report.to_json() + "\n")

    return report


def load_reports(path: Path | None = None) -> list[dict]:
    """Read every report. For the reviewer, not for the service.

    Malformed lines are skipped rather than raising: one bad line must not make the
    rest of the queue unreadable.
    """
    target = path or REPORTS_PATH
    if not target.is_file():
        return []

    reports: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            reports.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return reports


def summarise(path: Path | None = None) -> dict:
    """Counts for the review queue."""
    reports = load_reports(path)
    by_type: dict[str, int] = {}
    for report in reports:
        key = report.get("correction", "unknown")
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "total": len(reports),
        "unreviewed": sum(1 for r in reports if not r.get("reviewed")),
        "by_correction": dict(sorted(by_type.items())),
    }
