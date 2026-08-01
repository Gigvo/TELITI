/**
 * One fired rule, rendered as evidence the user can weigh.
 *
 * Shows BOTH the human explanation and the numeric contribution, because concept
 * paper section 3.1 requires the score be explainable per item — "18/100" with no
 * breakdown is exactly the opaque verdict the paper argues against.
 */

import type { RuleHit } from "../api/types";

const CATEGORY_LABEL: Record<string, string> = {
  contact: "Contact",
  company: "Company",
  compensation: "Compensation",
  qualification: "Qualification",
  language: "Language",
};

function tier(severity: number): string {
  if (severity >= 0.7) return "rule--high";
  if (severity >= 0.4) return "rule--med";
  return "rule--low";
}

export function RuleCard({ hit, locale }: { hit: RuleHit; locale: string }) {
  // Show the label in the language of the ad, matching the rest of the analysis.
  const title = locale === "id" ? hit.label_id : hit.label_en;

  return (
    <article className={`rule ${tier(hit.severity)}`}>
      <div className="rule-head">
        <span className="rule-title">{title}</span>
        <span className="rule-points">−{hit.contribution.toFixed(1)} pts</span>
      </div>
      <p className="rule-evidence">{hit.evidence}</p>
      <div className="rule-meta">
        {CATEGORY_LABEL[hit.category] ?? hit.category} · severity{" "}
        {hit.severity.toFixed(2)} · <code>{hit.rule_id}</code>
      </div>
    </article>
  );
}
