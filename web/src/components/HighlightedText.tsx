/**
 * Renders the submitted text with rule and sentence evidence highlighted in place.
 *
 * ## Why this is not a one-liner
 *
 * The API returns two independent sets of spans — rule hits and sentence evidence —
 * and they overlap constantly. A rule span for "administration fee" sits inside the
 * sentence span that contains it. Rendering them by naive substring replacement
 * would duplicate or drop text.
 *
 * So spans are flattened into a non-overlapping sequence first: sort by start,
 * drop any span that begins before the previous one ended, and keep rule spans in
 * preference to sentence spans because they are more specific.
 *
 * ## Offsets index the RAW text
 *
 * `api/schemas.py` guarantees spans address exactly the string the user submitted,
 * which is why this can slice `text` directly. Any drift there shows up here as
 * highlights landing on the wrong words — that invariant is asserted server-side in
 * tests/test_api.py.
 */

import type { RuleHit, SentenceEvidence } from "../api/types";

type Kind = "rule" | "sentence";

interface Marked {
  start: number;
  end: number;
  kind: Kind;
  title: string;
}

interface Props {
  text: string;
  ruleHits: RuleHit[];
  sentences: SentenceEvidence[];
}

function collectSpans(ruleHits: RuleHit[], sentences: SentenceEvidence[]): Marked[] {
  const spans: Marked[] = [];

  for (const hit of ruleHits) {
    if (hit.span) {
      spans.push({
        start: hit.span.start,
        end: hit.span.end,
        kind: "rule",
        title: `${hit.label_en} (-${hit.contribution.toFixed(1)} pts)`,
      });
    }
  }

  // Only risk-bearing sentences are worth highlighting; a "safe" sentence carries
  // no signal the user needs to see.
  for (const sentence of sentences) {
    if (sentence.span && sentence.polarity === "risk") {
      spans.push({
        start: sentence.span.start,
        end: sentence.span.end,
        kind: "sentence",
        title: `Suspicious sentence (Δp ${sentence.delta.toFixed(3)})`,
      });
    }
  }

  // Rule spans are the more specific evidence and must win any overlap, even when
  // a sentence span starts earlier and encloses them. Sorting by start alone would
  // let an enclosing sentence claim the region and drop the rule highlight
  // entirely — silently losing the most precise thing we have to show.
  //
  // So rules are placed first, then sentences fill only the gaps left over.
  const byPosition = (a: Marked, b: Marked) => a.start - b.start || b.end - a.end;
  const ruleSpans = spans.filter((s) => s.kind === "rule").sort(byPosition);
  const sentenceSpans = spans.filter((s) => s.kind === "sentence").sort(byPosition);

  const placed: Marked[] = [];
  const overlapsPlaced = (candidate: Marked) =>
    placed.some((p) => candidate.start < p.end && p.start < candidate.end);

  for (const span of [...ruleSpans, ...sentenceSpans]) {
    if (span.end > span.start && !overlapsPlaced(span)) {
      placed.push(span);
    }
  }

  return placed.sort((a, b) => a.start - b.start);
}

export function HighlightedText({ text, ruleHits, sentences }: Props) {
  const spans = collectSpans(ruleHits, sentences);

  const nodes: React.ReactNode[] = [];
  let position = 0;

  spans.forEach((span, index) => {
    // Guard against a malformed span running past the end of the text rather than
    // throwing inside render.
    const start = Math.max(position, Math.min(span.start, text.length));
    const end = Math.max(start, Math.min(span.end, text.length));
    if (start > position) {
      nodes.push(text.slice(position, start));
    }
    if (end > start) {
      nodes.push(
        <mark
          key={`${span.kind}-${index}`}
          className={span.kind === "rule" ? "hl-rule" : "hl-sentence"}
          title={span.title}
        >
          {text.slice(start, end)}
        </mark>,
      );
    }
    position = end;
  });

  if (position < text.length) {
    nodes.push(text.slice(position));
  }

  return (
    <>
      <div className="highlighted">{nodes}</div>
      <div className="legend">
        <span className="k-rule">Rule evidence</span>
        <span className="k-sentence">Suspicious sentence</span>
      </div>
    </>
  );
}
