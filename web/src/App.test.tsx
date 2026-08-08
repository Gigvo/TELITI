/**
 * App behaviour — MVP_PLAN.md step 1.6.
 *
 * The two invariants worth guarding are honesty invariants, not cosmetic ones:
 * the stub warning must be visible while the model is fake, and unassessed rules
 * must be surfaced rather than silently omitted.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { LOADED_HEALTH, SCAM_RESPONSE, SCAM_TEXT, STUBBED_HEALTH } from "./test/fixtures";

/** Put an advertisement in the textarea without simulating every keystroke. */
function enterAdText(value: string = SCAM_TEXT) {
  fireEvent.change(screen.getByLabelText(/Paste the job advertisement/i), {
    target: { value },
  });
}

const DEFAULT_REPORT = {
  report_id: "rep-test",
  received_at: "2026-08-08T00:00:00Z",
  message: "ok",
  stored_text: true,
  used_for_training: false,
};

/**
 * Routes by path. /analyze and /report must not share a response — returning the
 * report payload for an analyse call silently breaks every assertion downstream.
 */
function mockFetch(
  health = STUBBED_HEALTH,
  analyze: unknown = SCAM_RESPONSE,
  ok = true,
  reportResponse: unknown = DEFAULT_REPORT,
) {
  // `init` is declared so tests can assert on the request BODY, not just the URL.
  return vi.fn(async (url: string | URL, init?: RequestInit) => {
    void init;
    const href = url.toString();
    if (href.includes("/health")) {
      return new Response(JSON.stringify(health), { status: 200 });
    }
    if (href.includes("/report")) {
      return new Response(JSON.stringify(reportResponse), { status: 200 });
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

    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(await screen.findByText("21")).toBeInTheDocument();
    expect(screen.getByText("Tinggi")).toBeInTheDocument();
    expect(
      screen.getByText(/Requests payment or a transfer from the applicant/i),
    ).toBeInTheDocument();
    // No point badge: the fixture has rule_layer_enabled false, so these findings
    // did not move the score. One badge per hit. See the advisory tests below.
    expect(screen.getAllByText("note only")).toHaveLength(SCAM_RESPONSE.rule_hits.length);
  });


  it("does not claim advisory rules changed the score", async () => {
    /* The honesty invariant. With the rule layer disabled, a card reading
       "−9.5 pts" beside a score those points never touched would be the interface
       lying about its own reasoning. */
    const user = userEvent.setup();
    render(<App />);
    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    await screen.findByText("21");
    expect(screen.queryByText(/pts/)).toBeNull();
    expect(screen.getByText(/Additional observations/i)).toBeInTheDocument();
    expect(screen.getByText(/context only/i)).toBeInTheDocument();
  });


  it("shows point contributions when the rule layer is enabled", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch(STUBBED_HEALTH, { ...SCAM_RESPONSE, rule_layer_enabled: true }),
    );
    const user = userEvent.setup();
    render(<App />);
    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(await screen.findByText("−9.5 pts")).toBeInTheDocument();
    expect(screen.getByText(/Why this score/i)).toBeInTheDocument();
  });


  it("does not add the keyword caveat once evidence is model-derived", async () => {
    /* Inverted at step 3.4. The caveat was correct while sentences came from a
       keyword list; leaving it up now would understate what the system knows. */
    const user = userEvent.setup();
    render(<App />);
    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    await screen.findByText("21");
    expect(screen.queryByText(/matched by keyword/i)).toBeNull();
  });


  it("keeps the caveat when the API reports approximate evidence", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch(STUBBED_HEALTH, { ...SCAM_RESPONSE, sentence_evidence_approximate: true }),
    );
    const user = userEvent.setup();
    render(<App />);
    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(await screen.findByText(/matched by keyword/i)).toBeInTheDocument();
  });


  // ---------------------------------------------------------------- URL input


  it("offers a link input alongside pasted text", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: /From a link/i }));
    expect(screen.getByLabelText(/Link to the job posting/i)).toBeInTheDocument();
  });


  it("sets expectations about which links can be read", async () => {
    /* Measured: roughly three in ten real dataset URLs could be parsed. Telling the
       user up front beats letting them hit a refusal and assume it is broken. */
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: /From a link/i }));
    expect(screen.getByText(/Expired links, forum threads/i)).toBeInTheDocument();
    expect(screen.getByText(/have no link/i)).toBeInTheDocument();
  });


  it("requires a complete http address before submitting a link", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: /From a link/i }));

    const button = screen.getByRole("button", { name: /^Analyse$/i });
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText(/Link to the job posting/i), "not-a-url");
    expect(button).toBeDisabled();
    expect(screen.getByText(/Needs a full http/i)).toBeInTheDocument();

    await user.clear(screen.getByLabelText(/Link to the job posting/i));
    await user.type(
      screen.getByLabelText(/Link to the job posting/i),
      "https://example.com/lowongan/1",
    );
    expect(button).toBeEnabled();
  });


  it("sends url instead of text when the link tab is active", async () => {
    const fetchMock = mockFetch(STUBBED_HEALTH, {
      ...SCAM_RESPONSE,
      source_url: "https://example.com/lowongan/1",
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: /From a link/i }));
    await user.type(
      screen.getByLabelText(/Link to the job posting/i),
      "https://example.com/lowongan/1",
    );
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));
    await screen.findByText("21");

    const analyzeCall = fetchMock.mock.calls.find(([u]) => String(u).includes("/analyze"));
    expect(analyzeCall).toBeDefined();
    const body = JSON.parse(String(analyzeCall![1]!.body));
    // Exactly one field — the server rejects both-or-neither.
    expect(body).toEqual({ url: "https://example.com/lowongan/1" });
  });


  it("shows where a fetched advertisement came from", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch(STUBBED_HEALTH, {
        ...SCAM_RESPONSE,
        source_url: "https://example.com/lowongan/1?ref=x",
      }),
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: /From a link/i }));
    await user.type(
      screen.getByLabelText(/Link to the job posting/i),
      "https://example.com/lowongan/1",
    );
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(
      await screen.findByText(/example\.com\/lowongan\/1\?ref=x/),
    ).toBeInTheDocument();
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

    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(await screen.findByText(/Not assessed:/i)).toBeInTheDocument();
    expect(screen.getByText(/not.*clean results/i)).toBeInTheDocument();
  });

  it("renders a friendly message when the API rejects the text", async () => {
    vi.stubGlobal("fetch", mockFetch(STUBBED_HEALTH, { detail: "too short" }, false));
    const user = userEvent.setup();
    render(<App />);

    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(await screen.findByText(/could not be analysed/i)).toBeInTheDocument();
  });

  it("shows the ethics disclaimer from the response", async () => {
    const user = userEvent.setup();
    render(<App />);
    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));

    expect(
      await screen.findByText(/risk indicator, not a verdict/i),
    ).toBeInTheDocument();
  });

  it("keeps the pasted advertisement in the textarea", async () => {
    /* Previously asserted that an example button filled the box. The example
       buttons were removed from the UI, so this now checks the plain paste path. */
    render(<App />);
    enterAdText();
    const textarea = screen.getByLabelText(/Paste the job advertisement/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain("URGENT HIRING");
  });


  it("offers no preset example buttons", async () => {
    render(<App />);
    expect(screen.queryByRole("button", { name: /Scam \(/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /Legitimate \(/i })).toBeNull();
  });

  it("highlights against the analysed text, not later edits", async () => {
    // If the user edits the box after analysing, offsets from the previous
    // response no longer describe what is on screen. The rendered evidence must
    // keep showing the text that was actually analysed.
    const user = userEvent.setup();
    render(<App />);

    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));
    await screen.findByText("21");

    const textarea = screen.getByLabelText(/Paste the job advertisement/i);
    await user.clear(textarea);
    await user.type(textarea, "completely different text that is long enough to pass");

    const highlighted = document.querySelector(".highlighted") as HTMLElement;
    expect(highlighted.textContent).toBe(SCAM_TEXT);
  });
});


// ------------------------------------------------- appeal (concept paper 3.6)


describe("appeal mechanism", () => {
  it("offers a way to dispute a result", async () => {
    const user = userEvent.setup();
    render(<App />);
    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));
    await screen.findByText("21");

    expect(screen.getByRole("button", { name: /Report it/i })).toBeInTheDocument();
  });

  it("says the advertisement will be stored BEFORE it is sent", async () => {
    /* Analysing stores nothing, so a user has every reason to assume reporting
       does not either. The difference has to be stated up front, not after. */
    const user = userEvent.setup();
    render(<App />);
    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));
    await screen.findByText("21");
    await user.click(screen.getByRole("button", { name: /Report it/i }));

    expect(screen.getByText(/stores the advertisement text/i)).toBeInTheDocument();
    expect(screen.getByText(/never used to retrain/i)).toBeInTheDocument();
  });

  it("sends the disputed score and request id so the report can be traced", async () => {
    const fetchMock = mockFetch();
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);
    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));
    await screen.findByText("21");
    await user.click(screen.getByRole("button", { name: /Report it/i }));
    await user.click(screen.getByRole("button", { name: /Send report/i }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes("/report"));
      expect(call).toBeDefined();
      const body = JSON.parse(String(call![1]!.body));
      expect(body.correction).toBe("false_positive");
      expect(body.reported_score).toBe(SCAM_RESPONSE.integrity_score);
      expect(body.request_id).toBe(SCAM_RESPONSE.request_id);
      expect(body.text).toBe(SCAM_RESPONSE.analysed_text);
    });
  });

  it("confirms the report was filed and not retrained on", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch(STUBBED_HEALTH, SCAM_RESPONSE, true, {
        ...DEFAULT_REPORT,
        report_id: "rep-abc123",
      }),
    );
    const user = userEvent.setup();
    render(<App />);
    enterAdText();
    await user.click(screen.getByRole("button", { name: /^Analyse$/i }));
    await screen.findByText("Tinggi");
    await user.click(screen.getByRole("button", { name: /Report it/i }));
    await user.click(screen.getByRole("button", { name: /Send report/i }));

    expect(await screen.findByText(/rep-abc123/)).toBeInTheDocument();
    expect(screen.getByText(/not used to retrain/i)).toBeInTheDocument();
  });
});
