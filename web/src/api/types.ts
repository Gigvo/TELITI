/**
 * Mirrors api/schemas.py — MVP_PLAN.md step 1.1.
 *
 * This file is the frontend half of the frozen contract. When `api/schemas.py`
 * changes, change this to match; the API serves an OpenAPI document at
 * /openapi.json if you want to regenerate rather than hand-edit.
 */

export type RiskLabel = "Rendah" | "Sedang" | "Tinggi";

export type SourceChannel =
  | "whatsapp"
  | "telegram"
  | "instagram"
  | "job_board"
  | "web"
  | "other";

export type Polarity = "risk" | "safe";

export type RuleCategory =
  | "contact"
  | "company"
  | "compensation"
  | "qualification"
  | "language";

export interface Span {
  start: number;
  end: number;
}

export interface SentenceEvidence {
  text: string;
  /** Change in p(scam) when this sentence is removed. Positive = carried risk. */
  delta: number;
  polarity: Polarity;
  span: Span | null;
}

export interface RuleHit {
  rule_id: string;
  category: RuleCategory;
  label_id: string;
  label_en: string;
  severity: number;
  /** Score points removed. Positive pushes the integrity score DOWN. */
  contribution: number;
  /**
   * Human-readable explanation. NOT guaranteed to be a literal substring of the
   * input — use `span` for highlighting.
   */
  evidence: string;
  span: Span | null;
}

export interface ExtractedFields {
  title: string | null;
  company: string | null;
  location: string | null;
  salary_raw: string | null;
  salary_idr_monthly: number | null;
  emails: string[];
  phones: string[];
  urls: string[];
}

/**
 * Exactly one of `text` or `url`. The server rejects both-or-neither rather than
 * silently preferring one — a user who supplied a link AND text should never get a
 * score for the text while believing the link was checked.
 */
export interface AnalyzeRequest {
  text?: string;
  url?: string;
  source_channel?: SourceChannel;
  profile?: "text_only" | "structured";
  locale?: string;
}

export interface AnalyzeResponse {
  integrity_score: number;
  risk_label: RiskLabel;
  model_probability: number;
  fused_probability: number;
  summary: string;
  sentence_evidence: SentenceEvidence[];
  rule_hits: RuleHit[];
  extracted_fields: ExtractedFields;
  /**
   * The exact text that was scored, after server-side sanitisation. All spans
   * index THIS string — render it rather than the raw submission, or highlights
   * drift from what was actually analysed.
   */
  analysed_text: string;
  /**
   * False when the rule layer is ADVISORY — its findings are real but did not move
   * the score, and every `contribution` is 0.0. Disabled on measured evidence: the
   * rules cost 0.064 PR-AUC and five times the false positives on the Indonesian
   * holdout. The UI must not present these as if they changed the number.
   */
  rule_layer_enabled: boolean;
  /**
   * True while `sentence_evidence` comes from keyword matching rather than model
   * occlusion. The UI must not imply the model pointed at those sentences.
   */
  sentence_evidence_approximate: boolean;
  /** Set when the ad was fetched from a link. This is the URL after redirects. */
  source_url: string | null;
  locale: string;
  locale_detected: string;
  /**
   * Rules that could NOT be evaluated. These are NOT "clean" — the signal was
   * never checked. The UI must surface them so a partial analysis is never
   * mistaken for a complete one.
   */
  unassessed_rules: string[];
  disclaimer: string;
  privacy_note: string;
  request_id: string;
  model_version: string;
  latency_ms: number;
}

export interface HealthResponse {
  status: string;
  model_version: string;
  /** False while the text model is stubbed. The demo must not run with this false. */
  model_loaded: boolean;
  thresholds_loaded: boolean;
  locales_available: string[];
  locale_resources: Record<string, string[]>;
}

/**
 * Appeal / label correction — concept paper §3.6.
 *
 * ⚠️ Filing this STORES the advertisement text. Analysis stores nothing, so the UI
 * must say so before the user submits.
 */
export type CorrectionType =
  | "false_positive"
  | "false_negative"
  | "wrong_evidence"
  | "other";

export interface ReportRequest {
  correction: CorrectionType;
  text: string;
  reported_score?: number;
  reported_label?: RiskLabel;
  request_id?: string;
  comment?: string;
  contact?: string;
}

export interface ReportResponse {
  report_id: string;
  received_at: string;
  message: string;
  stored_text: boolean;
  /** Always false — reports are quarantined for human review, never retrained on. */
  used_for_training: boolean;
}
