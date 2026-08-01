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

export interface AnalyzeRequest {
  text: string;
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
