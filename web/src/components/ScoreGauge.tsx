/**
 * Integrity Score gauge — concept paper section 3.1.
 *
 * The score is ALWAYS shown as a number alongside the ring. The ring is a visual
 * aid, never the sole carrier of the value: colour and arc length both fail for
 * colour-blind users and in greyscale, and the risk label must remain readable.
 */

import type { RiskLabel } from "../api/types";

const RADIUS = 46;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

const COLOURS: Record<RiskLabel, string> = {
  Tinggi: "var(--risk-high)",
  Sedang: "var(--risk-med)",
  Rendah: "var(--risk-low)",
};

interface Props {
  score: number;
  label: RiskLabel;
}

export function ScoreGauge({ score, label }: Props) {
  const clamped = Math.max(0, Math.min(100, score));
  const filled = (clamped / 100) * CIRCUMFERENCE;

  return (
    <div
      className="gauge"
      role="img"
      aria-label={`Integrity score ${clamped} out of 100, risk level ${label}`}
    >
      <svg width="108" height="108" viewBox="0 0 108 108" aria-hidden="true">
        <circle
          cx="54"
          cy="54"
          r={RADIUS}
          fill="none"
          stroke="var(--border)"
          strokeWidth="9"
        />
        <circle
          cx="54"
          cy="54"
          r={RADIUS}
          fill="none"
          stroke={COLOURS[label]}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${CIRCUMFERENCE - filled}`}
        />
      </svg>
      <div className="gauge-value">
        <strong>{clamped}</strong>
        <span>/ 100</span>
      </div>
    </div>
  );
}
