/**
 * The base-URL join. This is deployment configuration, and it fails in a way
 * that does not look like configuration: a trailing slash on VITE_API_BASE
 * produces "//api/v1/analyze", FastAPI answers 405 Method Not Allowed, and the
 * error points at the HTTP method rather than at the pasted variable.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

async function urlFor(base: string | undefined): Promise<string> {
  vi.resetModules();
  vi.stubEnv("VITE_API_BASE", base ?? "");

  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({}),
  });
  vi.stubGlobal("fetch", fetchMock);

  const { health } = await import("./client");
  await health();

  return fetchMock.mock.calls[0][0] as string;
}

describe("API base URL", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("uses a same-origin path when unset", async () => {
    // The Docker image serves this bundle itself, so same-origin is correct.
    expect(await urlFor("")).toBe("/health");
  });

  it("prefixes an absolute base for a split deploy", async () => {
    expect(await urlFor("https://teliti.hf.space")).toBe(
      "https://teliti.hf.space/health",
    );
  });

  it("tolerates a trailing slash", async () => {
    expect(await urlFor("https://teliti.hf.space/")).toBe(
      "https://teliti.hf.space/health",
    );
  });

  it("tolerates several trailing slashes", async () => {
    expect(await urlFor("https://teliti.hf.space///")).toBe(
      "https://teliti.hf.space/health",
    );
  });
});
