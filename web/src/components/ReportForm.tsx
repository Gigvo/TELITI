/**
 * Appeal / label correction — concept paper §3.6.
 *
 * The case this exists for is a legitimate company scored *Tinggi*. §3.6 names
 * false positives against real businesses as the error to suppress, and a system
 * that can be wrong about a company with no way to say so is worse than one that
 * admits it.
 *
 * Two things this form is deliberate about:
 *
 * 1. **It says the advertisement will be stored, before it is sent.** Analysis
 *    stores nothing; filing a report is a separate act, and the user should know
 *    which one they are performing.
 * 2. **It says reports are not retrained on.** Otherwise "report this" reads as
 *    "teach the model", which is both untrue and the thing the paper's Tahap 3
 *    warns about (data poisoning).
 *
 * Collapsed by default — an appeal route must be discoverable without competing
 * with the result for attention.
 */

import { useState } from "react";

import { ApiError, report } from "../api/client";
import type { AnalyzeResponse, CorrectionType } from "../api/types";

const OPTIONS: { value: CorrectionType; label: string }[] = [
  { value: "false_positive", label: "This is a real job — it should not be flagged" },
  { value: "false_negative", label: "This is a scam — it was scored too safe" },
  { value: "wrong_evidence", label: "The score seems right, the reasoning does not" },
  { value: "other", label: "Something else" },
];

export function ReportForm({ result }: { result: AnalyzeResponse }) {
  const [open, setOpen] = useState(false);
  const [correction, setCorrection] = useState<CorrectionType>("false_positive");
  const [comment, setComment] = useState("");
  const [contact, setContact] = useState("");
  const [sending, setSending] = useState(false);
  const [sentId, setSentId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setSending(true);
    setError(null);
    try {
      const response = await report({
        correction,
        text: result.analysed_text,
        reported_score: result.integrity_score,
        reported_label: result.risk_label,
        request_id: result.request_id,
        comment: comment.trim(),
        contact: contact.trim() || undefined,
      });
      setSentId(response.report_id);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Could not send the report.",
      );
    } finally {
      setSending(false);
    }
  }

  if (sentId) {
    return (
      <div className="banner banner--info report-done">
        <strong>Thank you — report {sentId} received.</strong> A person will review
        it. Your report is not used to retrain the model.
      </div>
    );
  }

  if (!open) {
    return (
      <button type="button" className="report-open" onClick={() => setOpen(true)}>
        Disagree with this result? Report it
      </button>
    );
  }

  return (
    <div className="report-form">
      <h3>Report an incorrect result</h3>

      <fieldset>
        <legend className="sr-only">What went wrong</legend>
        {OPTIONS.map((option) => (
          <label key={option.value} className="report-option">
            <input
              type="radio"
              name="correction"
              value={option.value}
              checked={correction === option.value}
              onChange={() => setCorrection(option.value)}
            />
            <span>{option.label}</span>
          </label>
        ))}
      </fieldset>

      <label htmlFor="report-comment" className="sr-only">
        Anything else we should know
      </label>
      <textarea
        id="report-comment"
        className="report-comment"
        value={comment}
        onChange={(event) => setComment(event.target.value)}
        placeholder="Anything else we should know? (optional)"
        maxLength={2000}
      />

      <label htmlFor="report-contact" className="sr-only">
        Email, if you would like a reply
      </label>
      <input
        id="report-contact"
        className="url-input"
        value={contact}
        onChange={(event) => setContact(event.target.value)}
        placeholder="Email, if you would like a reply (optional)"
        maxLength={200}
      />

      {/* Stated before sending, not after. Analysis stores nothing, so a user has
          every reason to assume this does not either. */}
      <p className="hint">
        Sending this <strong>stores the advertisement text</strong> so a person can
        review it — analysing does not. Reports are never used to retrain the model.
      </p>

      {error && <div className="banner banner--error">{error}</div>}

      <div className="controls">
        <button
          type="button"
          className="btn-primary"
          onClick={() => void submit()}
          disabled={sending}
        >
          {sending && <span className="spinner" aria-hidden="true" />}
          {sending ? "Sending…" : "Send report"}
        </button>
        <button type="button" className="btn-ghost" onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
    </div>
  );
}
