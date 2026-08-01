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
import { RuleCard } from "./components/RuleCard";
import { ScoreGauge } from "./components/ScoreGauge";
import { EXAMPLES } from "./examples";

// Mirrors MIN/MAX_TEXT_LENGTH in api/constants.py. Validating here too gives
// instant feedback instead of a 422 round-trip.
const MIN_LENGTH = 30;
const MAX_LENGTH = 20000;

export default function App() {
  const [text, setText] = useState("");
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
  const tooShort = trimmed.length > 0 && trimmed.length < MIN_LENGTH;
  const tooLong = trimmed.length > MAX_LENGTH;
  const canSubmit = trimmed.length >= MIN_LENGTH && !tooLong && !loading;

  async function handleAnalyze() {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await analyze({ text: trimmed }));
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

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
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

          <div className="examples">
            {EXAMPLES.map((example) => (
              <button
                key={example.id}
                type="button"
                className="btn-ghost"
                onClick={() => {
                  setText(example.text);
                  setResult(null);
                  setError(null);
                }}
              >
                {example.label}
              </button>
            ))}
          </div>

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

            {text.length > 0 && (
              <button
                type="button"
                className="btn-ghost"
                onClick={() => {
                  setText("");
                  setResult(null);
                  setError(null);
                }}
              >
                Clear
              </button>
            )}

            <span
              className={`charcount ${tooShort || tooLong ? "is-invalid" : ""}`}
              aria-live="polite"
            >
              {tooShort
                ? `${MIN_LENGTH - trimmed.length} more characters needed`
                : tooLong
                  ? `${trimmed.length - MAX_LENGTH} characters over the limit`
                  : `${trimmed.length.toLocaleString()} characters`}
            </span>
          </div>
        </section>

        {/* ---------------- result ---------------- */}
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
      </div>

      {result && (
        <div className="columns">
          <section className="panel">
            <h2>Why this score</h2>

            {sortedHits.length === 0 ? (
              <p className="empty">No deterministic rules were triggered.</p>
            ) : (
              sortedHits.map((hit) => (
                <RuleCard key={hit.rule_id} hit={hit} locale={result.locale} />
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
            <HighlightedText
              text={result.analysed_text}
              ruleHits={result.rule_hits}
              sentences={result.sentence_evidence}
            />

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
        </div>
      )}

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
