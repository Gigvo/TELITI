/**
 * Highlighting must never corrupt the user's text.
 *
 * The API returns two independent, overlapping sets of spans. Naive rendering
 * duplicates or drops characters, which looks to the user like the tool
 * misquoting their own advertisement — worse than showing no highlights at all.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { RuleHit, SentenceEvidence } from "../api/types";
import { SCAM_RESPONSE, SCAM_TEXT } from "../test/fixtures";
import { HighlightedText } from "./HighlightedText";

function renderText(
  text: string,
  ruleHits: RuleHit[] = [],
  sentences: SentenceEvidence[] = [],
) {
  const { container } = render(
    <HighlightedText text={text} ruleHits={ruleHits} sentences={sentences} />,
  );
  return container.querySelector(".highlighted") as HTMLElement;
}

describe("HighlightedText", () => {
  it("preserves the original text exactly", () => {
    const el = renderText(SCAM_TEXT, SCAM_RESPONSE.rule_hits, SCAM_RESPONSE.sentence_evidence);
    expect(el.textContent).toBe(SCAM_TEXT);
  });

  it("preserves the text when no spans are supplied", () => {
    expect(renderText(SCAM_TEXT).textContent).toBe(SCAM_TEXT);
  });

  it("marks rule evidence", () => {
    const el = renderText(SCAM_TEXT, SCAM_RESPONSE.rule_hits);
    const marks = el.querySelectorAll("mark.hl-rule");
    expect(marks.length).toBe(1);
    expect(marks[0].textContent).toBe("administration fee");
  });

  it("marks risky sentences", () => {
    const el = renderText(SCAM_TEXT, [], SCAM_RESPONSE.sentence_evidence);
    expect(el.querySelector("mark.hl-sentence")?.textContent).toBe(
      "Interview conducted via Telegram.",
    );
  });

  it("does not corrupt text when spans overlap", () => {
    // A rule span sitting inside a sentence span — the common real case.
    const text = "Please pay an administration fee of $50 today.";
    const el = renderText(
      text,
      [
        {
          ...SCAM_RESPONSE.rule_hits[0],
          span: { start: text.indexOf("administration fee"), end: text.indexOf("administration fee") + 18 },
        },
      ],
      [{ text, delta: 0.2, polarity: "risk", span: { start: 0, end: text.length } }],
    );
    expect(el.textContent).toBe(text);
    // The more specific rule span wins the overlap.
    expect(el.querySelectorAll("mark.hl-rule").length).toBe(1);
  });

  it("ignores safe sentences", () => {
    const el = renderText(SCAM_TEXT, [], [
      { text: "Earn", delta: -0.01, polarity: "safe", span: { start: 0, end: 4 } },
    ]);
    expect(el.querySelectorAll("mark").length).toBe(0);
  });

  it("survives a span running past the end of the text", () => {
    // Defensive: a malformed span must not throw inside render.
    const el = renderText("Short text here for testing.", [
      { ...SCAM_RESPONSE.rule_hits[0], span: { start: 5, end: 9999 } },
    ]);
    expect(el.textContent).toBe("Short text here for testing.");
  });

  it("handles emoji without corrupting the string", () => {
    const text = "🔥 URGENT HIRING 🔥 pay a fee now 💰";
    const el = renderText(text, [
      { ...SCAM_RESPONSE.rule_hits[0], span: { start: text.indexOf("pay a fee"), end: text.indexOf("pay a fee") + 9 } },
    ]);
    expect(el.textContent).toBe(text);
  });

  it("renders a legend so the colours are decodable", () => {
    renderText(SCAM_TEXT, SCAM_RESPONSE.rule_hits);
    expect(screen.getByText("Rule evidence")).toBeInTheDocument();
    expect(screen.getByText("Suspicious sentence")).toBeInTheDocument();
  });
});
