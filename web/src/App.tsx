/**
 * TELITI — MVP_PLAN.md step 1.6.
 *
 * Single screen: paste a job ad, get an Integrity Score with per-item evidence.
 *
 * Two things this UI is deliberate about:
 *
 * 1. **It never hides that the model is stubbed.** While `/health` reports
 *    `model_loaded: false`, a warning banner stays up. A screenshot of this page
 *    must never be mistakable for a real result.
 *
 * 2. **It surfaces unassessed rules.** A rule that could not run is not a rule
 *    that found nothing. Showing only the fired rules would let a partial analysis
 *    read as a complete one.
 */

import { useEffect, useMemo, useState } from "react";

import { ApiError, analyze, health } from "./api/client";
import type { AnalyzeResponse, HealthResponse } from "./api/types";
import { HighlightedText } from "./components/HighlightedText";
import { ReportForm } from "./components/ReportForm";
import { RuleCard } from "./components/RuleCard";
import { ScoreGauge } from "./components/ScoreGauge";

// Mirrors MIN/MAX_TEXT_LENGTH in api/constants.py. Validating here too gives
// instant feedback instead of a 422 round-trip.
const MIN_LENGTH = 30;
const MAX_LENGTH = 20000;

type InputMode = "text" | "url";

export default function App() {
  const [mode, setMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<{ message: string; detail?: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [serverHealth, setServerHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);

  // Highlighting slices `result.analysed_text` — the server's sanitised copy —
  // never the textarea. Two reasons: the user may edit the box after analysing,
  // and sanitisation strips bidirectional overrides, so rendering the raw
  // submission could display text that differs from what was scored.

  useEffect(() => {
    health()
      .then(setServerHealth)
      .catch(() => setHealthError(true));
  }, []);

  const trimmed = text.trim();
  const trimmedUrl = url.trim();
  const tooShort = trimmed.length > 0 && trimmed.length < MIN_LENGTH;
  const tooLong = trimmed.length > MAX_LENGTH;
  const urlLooksValid = /^https?:\/\/.+\..+/i.test(trimmedUrl);

  const canSubmit = loading
    ? false
    : mode === "text"
      ? trimmed.length >= MIN_LENGTH && !tooLong
      : urlLooksValid;

  async function handleAnalyze() {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    try {
      // Exactly one field — the server rejects both-or-neither.
      setResult(await analyze(mode === "text" ? { text: trimmed } : { url: trimmedUrl }));
    } catch (err) {
      setResult(null);
      if (err instanceof ApiError) {
        setError({ message: err.message, detail: err.detail });
      } else {
        setError({ message: "Something went wrong while analysing." });
      }
    } finally {
      setLoading(false);
    }
  }

  // Shared by the textarea and the URL input, so the element type must cover both.
  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement | HTMLInputElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      void handleAnalyze();
    }
  }

  const sortedHits = useMemo(
    () => (result ? [...result.rule_hits].sort((a, b) => b.contribution - a.contribution) : []),
    [result],
  );

  const stubbed = serverHealth !== null && !serverHealth.model_loaded;

  return (
    <div className="app">
      <header className="masthead">
        <h1>TELITI</h1>
        <span className="tagline">
          Teknologi Evaluasi Lowongan dan Integritas — job-ad integrity scoring
        </span>
      </header>

      {healthError && (
        <div className="banner banner--error">
          <strong>API unreachable.</strong> Start it with{" "}
          <code>uvicorn api.main:app --reload --port 8000</code> from the repository
          root.
        </div>
      )}

      {stubbed && (
        <div className="banner banner--warn">
          <strong>Text model is stubbed.</strong> The rule layer and extracted fields
          are real; <code>model_probability</code> is synthetic, so the score itself
          means nothing yet. Do not use this view as a result.
        </div>
      )}

      <div className="columns">
        {/* ---------------- input ---------------- */}
        <section className="panel">
          <h2>Job advertisement</h2>

          <div className="mode-tabs" role="tablist" aria-label="Input method">
            <button
              type="button"
              role="tab"
              aria-selected={mode === "text"}
              className={`mode-tab ${mode === "text" ? "is-active" : ""}`}
              onClick={() => { setMode("text"); setError(null); }}
            >
              Paste text
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "url"}
              className={`mode-tab ${mode === "url" ? "is-active" : ""}`}
              onClick={() => { setMode("url"); setError(null); }}
            >
              From a link
            </button>
          </div>

          {mode === "text" ? (
            <>
              <label htmlFor="ad-text" className="sr-only">
                Paste the job advertisement text
              </label>
              <textarea
                id="ad-text"
                value={text}
                onChange={(event) => setText(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Paste a job advertisement here — from WhatsApp, Telegram, Instagram, or a job board…"
                spellCheck={false}
              />
            </>
          ) : (
            <>
              <label htmlFor="ad-url" className="sr-only">
                Link to the job posting
              </label>
              <input
                id="ad-url"
                type="url"
                className="url-input"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="https://example.com/lowongan/…"
                spellCheck={false}
              />
              {/* Setting expectations honestly: measured on real dataset URLs, about
                  three in ten could be read. Most job-scam messages arrive on
                  WhatsApp or Telegram and have no link at all. */}
              <p className="hint">
                Works on live job-board postings. Expired links, forum threads and
                pages we cannot read will be declined — paste the text instead.
                Messages from WhatsApp or Telegram have no link, so use{" "}
                <strong>Paste text</strong> for those.
              </p>
            </>
          )}

          <div className="controls">
            <button
              type="button"
              className="btn-primary"
              onClick={() => void handleAnalyze()}
              disabled={!canSubmit}
            >
              {loading && <span className="spinner" aria-hidden="true" />}
              {loading ? "Analysing…" : "Analyse"}
            </button>

            {(mode === "text" ? text.length : url.length) > 0 && (
              <button
                type="button"
                className="btn-ghost"
                onClick={() => {
                  if (mode === "text") setText("");
                  else setUrl("");
                  setResult(null);
                  setError(null);
                }}
              >
                Clear
              </button>
            )}

            <span
              className={`charcount ${
                mode === "text" && (tooShort || tooLong) ? "is-invalid" : ""
              }`}
              aria-live="polite"
            >
              {mode === "url"
                ? trimmedUrl.length === 0
                  ? "Paste a link"
                  : urlLooksValid
                    ? "Link looks valid"
                    : "Needs a full http(s) address"
                : tooShort
                  ? `${MIN_LENGTH - trimmed.length} more characters needed`
                  : tooLong
                    ? `${trimmed.length - MAX_LENGTH} characters over the limit`
                    : `${trimmed.length.toLocaleString()} characters`}
            </span>
          </div>
        </section>

        {/* ---------------- result ---------------- */}
        {/* One continuous right-hand column.
            Previously the results lived in a SECOND `.columns` grid row, which only
            began once the tallest cell of the first row ended. The input panel is
            tall and the Analysis panel is short, so that left a large dead gap above
            "Evidence in context". Stacking them in one column removes it. */}
        <div className="results-column">
        <section className="panel" aria-live="polite">
          <h2>Analysis</h2>

          {error && (
            <div className="banner banner--error">
              <strong>{error.message}</strong>
              {error.detail && <div style={{ marginTop: 4 }}>{error.detail}</div>}
            </div>
          )}

          {!result && !error && (
            <p className="empty">
              Paste an advertisement and select <strong>Analyse</strong> to see the
              Integrity Score and the evidence behind it.
            </p>
          )}

          {result && (
            <>
              <div className="score-head">
                <ScoreGauge score={result.integrity_score} label={result.risk_label} />
                <div>
                  <span className={`risk-pill risk-${result.risk_label}`}>
                    {result.risk_label}
                  </span>
                  <p className="summary">{result.summary}</p>
                </div>
              </div>

              <div className="meta-line">
                locale {result.locale}
                {result.locale !== result.locale_detected &&
                  ` (detected ${result.locale_detected} — fell back)`}{" "}
                · model {result.model_version} · {result.latency_ms} ms
              </div>
            </>
          )}
        </section>

          {result && (
            <>
          <section className="panel">
            <h2>{result.rule_layer_enabled ? "Why this score" : "Additional observations"}</h2>

            {!result.rule_layer_enabled && sortedHits.length > 0 && (
              <div className="banner banner--info">
                These checks are shown as <strong>context only</strong> — they did not
                affect the score. Measured against real Indonesian advertisements they
                made the result worse, so the score comes from the model alone.
              </div>
            )}

            {sortedHits.length === 0 ? (
              <p className="empty">No deterministic rules were triggered.</p>
            ) : (
              sortedHits.map((hit) => (
                <RuleCard
                  key={hit.rule_id}
                  hit={hit}
                  locale={result.locale}
                  affectsScore={result.rule_layer_enabled}
                />
              ))
            )}

            {result.unassessed_rules.length > 0 && (
              <div className="banner banner--info" style={{ marginTop: 12 }}>
                <strong>Not assessed:</strong> {result.unassessed_rules.join(", ")}.
                These checks could not run — the language resources are missing, or
                the source removed the evidence. They are <em>not</em> clean results.
              </div>
            )}
          </section>

          <section className="panel">
            <h2>Evidence in context</h2>

            {result.source_url && (
              <p className="meta-line" style={{ marginTop: 0, marginBottom: 10 }}>
                fetched from <code>{result.source_url}</code>
              </p>
            )}

            <HighlightedText
              text={result.analysed_text}
              ruleHits={result.rule_hits}
              sentences={result.sentence_evidence}
            />

            {result.sentence_evidence_approximate && (
              <p className="hint" style={{ marginTop: 8 }}>
                Highlighted sentences are matched by keyword, not chosen by the model.
                Treat them as a reading aid rather than as the model's reasoning.
              </p>
            )}

            {(result.extracted_fields.title ||
              result.extracted_fields.emails.length > 0 ||
              result.extracted_fields.urls.length > 0 ||
              result.extracted_fields.phones.length > 0) && (
              <>
                <h2 style={{ marginTop: 18 }}>Extracted fields</h2>
                <dl className="fields">
                  {result.extracted_fields.title && (
                    <>
                      <dt>Title</dt>
                      <dd>{result.extracted_fields.title}</dd>
                    </>
                  )}
                  {result.extracted_fields.company && (
                    <>
                      <dt>Company</dt>
                      <dd>{result.extracted_fields.company}</dd>
                    </>
                  )}
                  {result.extracted_fields.emails.length > 0 && (
                    <>
                      <dt>Email</dt>
                      <dd>{result.extracted_fields.emails.join(", ")}</dd>
                    </>
                  )}
                  {result.extracted_fields.phones.length > 0 && (
                    <>
                      <dt>Phone</dt>
                      <dd>{result.extracted_fields.phones.join(", ")}</dd>
                    </>
                  )}
                  {result.extracted_fields.urls.length > 0 && (
                    <>
                      <dt>Links</dt>
                      <dd>{result.extracted_fields.urls.join(", ")}</dd>
                    </>
                  )}
                </dl>
                <p className="meta-line">
                  Shown so you can check our parsing. If a field is misread, discount
                  the rule that relied on it.
                </p>
              </>
            )}
          </section>

          <ReportForm key={result.request_id} result={result} />
            </>
          )}
        </div>
      </div>

      <footer className="footer-note">
        {result ? (
          <>
            {result.disclaimer} {result.privacy_note}
          </>
        ) : (
          <>
            This tool reports a risk indicator, not a verdict. It can be wrong. The
            text you analyse is not stored.
          </>
        )}
      </footer>
    </div>
  );
}
