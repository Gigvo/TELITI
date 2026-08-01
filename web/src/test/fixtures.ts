/** Shared fixtures, shaped exactly like real API responses. */

import type { AnalyzeResponse, HealthResponse } from "../api/types";

export const SCAM_TEXT = `URGENT HIRING - Data Entry Clerk (Work From Home)
Earn $5,000 per month, no experience necessary, start immediately!
A one-time administration fee of $50 is required to process your application.
Interview conducted via Telegram.
Send your CV to hiring.dept2024@gmail.com`;

/** Offsets below are computed from SCAM_TEXT so the spans are genuinely valid. */
const feeStart = SCAM_TEXT.indexOf("administration fee");
const telegramStart = SCAM_TEXT.indexOf("Interview conducted via Telegram.");

export const SCAM_RESPONSE: AnalyzeResponse = {
  integrity_score: 21,
  risk_label: "Tinggi",
  model_probability: 0.5,
  fused_probability: 0.79,
  summary: "[STUB] Integrity score 21/100 (High risk). Findings: requests payment.",
  sentence_evidence: [
    {
      text: "Interview conducted via Telegram.",
      delta: 0.12,
      polarity: "risk",
      span: { start: telegramStart, end: telegramStart + 33 },
    },
  ],
  rule_hits: [
    {
      rule_id: "payment_request_id",
      category: "language",
      label_id: "Meminta pembayaran atau transfer dari pelamar",
      label_en: "Requests payment or a transfer from the applicant",
      severity: 0.95,
      contribution: 9.5,
      evidence: 'Found "administration fee" in the job posting.',
      span: { start: feeStart, end: feeStart + 18 },
    },
    {
      rule_id: "email_free_provider",
      category: "company",
      label_id: "Kontak memakai email gratis",
      label_en: "Contact uses a free email provider",
      severity: 0.55,
      contribution: 2.2,
      evidence: "hiring.dept2024@gmail.com",
      span: null,
    },
  ],
  extracted_fields: {
    title: "URGENT HIRING - Data Entry Clerk (Work From Home)",
    company: null,
    location: null,
    salary_raw: null,
    salary_idr_monthly: null,
    emails: ["hiring.dept2024@gmail.com"],
    phones: [],
    urls: [],
  },
  locale: "en",
  locale_detected: "en",
  unassessed_rules: [],
  disclaimer: "This score is a risk indicator, not a verdict.",
  privacy_note: "The text you analyse is not stored by this system.",
  request_id: "test-request",
  model_version: "stub-0.0.0",
  latency_ms: 12,
};

export const STUBBED_HEALTH: HealthResponse = {
  status: "ok",
  model_version: "stub-0.0.0",
  model_loaded: false,
  thresholds_loaded: false,
  locales_available: ["en", "id"],
  locale_resources: { en: [], id: [] },
};

export const LOADED_HEALTH: HealthResponse = { ...STUBBED_HEALTH, model_loaded: true };
