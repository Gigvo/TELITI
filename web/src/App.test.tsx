/**
 * App behaviour — MVP_PLAN.md step 1.6.
 *
 * The two invariants worth guarding are honesty invariants, not cosmetic ones:
 * the stub warning must be visible while the model is fake, and unassessed rules
 * must be surfaced rather than silently omitted.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { LOADED_HEALTH, SCAM_RESPONSE, SCAM_TEXT, STUBBED_HEALTH } from "./test/fixtures";

function mockFetch(health = STUBBED_HEALTH, analyze: unknown = SCAM_RESPONSE, ok = true) {
  return vi.fn(async (url: string | URL) => {
    const href = url.toString();
    if (href.includes("/health")) {
      return new Response(JSON.stringify(health), { status: 200 });
    }
    return new Response(JSON.stringify(analyze), { status: ok ? 200 : 422 });
  });
}

beforeEach(() => {
  vi.stubGlobal("fetch", mockFetch());
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("App", () => {
  it("warns while the text model is stubbed", async () => {
    render(<App />);
    expect(await screen.findByText(/Text model is stubbed/i)).toBeInTheDocument();
  });

  it("hides the stub warning once a real model is loaded", async () => {
    vi.stubGlobal("fetch", mockFetch(LOADED_HEALTH));
    render(<App />);
    await waitFor(() => expect(screen.queryByText(/Text model is stubbed/i)).toBeNull());
  });

  it("warns when the API is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("network"); }));
    render(<App />);
    expect(await screen.findByText(/API unreachable/i)).toBeInTheDocument();
  });

  it("disables Analyse until the text is long enough", async () => {
    const user = userEvent.setup();
    render(<App />);
    const button = screen.getByRole("button", { name: /^Analyse$/i });
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText(/Paste the job advertisement/i), "too short");
    expect(button).toBeDisabled();
    expect(screen.getByText(/more characters needed/i)).toBeInTheDocument();
  });

  it("shows the score, risk label and rule evidence after analysing", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Scam \(English\)/i }));
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(await screen.findByText("21")).toBeInTheDocument();
    expect(screen.getByText("Tinggi")).toBeInTheDocument();
    expect(
      screen.getByText(/Requests payment or a transfer from the applicant/i),
    ).toBeInTheDocument();
    expect(screen.getByText("−9.5 pts")).toBeInTheDocument();
  });

  it("surfaces unassessed rules as NOT clean", async () => {
    // The honesty invariant: a check that could not run must never look like a
    // check that found nothing.
    vi.stubGlobal(
      "fetch",
      mockFetch(STUBBED_HEALTH, {
        ...SCAM_RESPONSE,
        unassessed_rules: ["salary_implausible_vs_umk"],
      }),
    );
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Scam \(English\)/i }));
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(await screen.findByText(/Not assessed:/i)).toBeInTheDocument();
    expect(screen.getByText(/not.*clean results/i)).toBeInTheDocument();
  });

  it("renders a friendly message when the API rejects the text", async () => {
    vi.stubGlobal("fetch", mockFetch(STUBBED_HEALTH, { detail: "too short" }, false));
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Scam \(English\)/i }));
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(await screen.findByText(/could not be analysed/i)).toBeInTheDocument();
  });

  it("shows the ethics disclaimer from the response", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: /Scam \(English\)/i }));
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(
      await screen.findByText(/risk indicator, not a verdict/i),
    ).toBeInTheDocument();
  });

  it("loads example text into the textarea", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: /Scam \(English\)/i }));
    const textarea = screen.getByLabelText(/Paste the job advertisement/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain("URGENT HIRING");
  });

  it("highlights against the analysed text, not later edits", async () => {
    // If the user edits the box after analysing, offsets from the previous
    // response no longer describe what is on screen. The rendered evidence must
    // keep showing the text that was actually analysed.
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Scam \(English\)/i }));
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));
    await screen.findByText("21");

    const textarea = screen.getByLabelText(/Paste the job advertisement/i);
    await user.clear(textarea);
    await user.type(textarea, "completely different text that is long enough to pass");

    const highlighted = document.querySelector(".highlighted") as HTMLElement;
    expect(highlighted.textContent).toBe(SCAM_TEXT);
  });
});
