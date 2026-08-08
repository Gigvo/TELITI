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

interface Props {
  hit: RuleHit;
  locale: string;
  /**
   * When false the rule layer is advisory: findings are real, but they did not move
   * the score. Rendering "−9.5 pts" beside a score those points never touched would
   * be the interface lying, so the badge is replaced with a neutral marker.
   */
  affectsScore: boolean;
}

export function RuleCard({ hit, locale, affectsScore }: Props) {
  // Show the label in the language of the ad, matching the rest of the analysis.
  const title = locale === "id" ? hit.label_id : hit.label_en;

  return (
    <article className={`rule ${tier(hit.severity)}`}>
      <div className="rule-head">
        <span className="rule-title">{title}</span>
        {affectsScore ? (
          <span className="rule-points">−{hit.contribution.toFixed(1)} pts</span>
        ) : (
          <span className="rule-note" title="Shown as context; did not change the score">
            note only
          </span>
        )}
      </div>
      <p className="rule-evidence">{hit.evidence}</p>
      <div className="rule-meta">
        {CATEGORY_LABEL[hit.category] ?? hit.category} · severity{" "}
        {hit.severity.toFixed(2)} · <code>{hit.rule_id}</code>
      </div>
    </article>
  );
}
