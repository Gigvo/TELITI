/**
 * API client. Talks to FastAPI through the Vite dev proxy (see vite.config.ts),
 * so there is no CORS preflight in development and no hardcoded host.
 */

import type { AnalyzeRequest, AnalyzeResponse, HealthResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  // Declared explicitly rather than as constructor parameter properties: the Vite
  // template enables `erasableSyntaxOnly`, which forbids that shorthand because it
  // emits runtime code from a type-position annotation.
  readonly status: number;
  readonly detail?: string;

  constructor(message: string, status: number, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // fetch only rejects on network failure, so this is almost always "API down".
    throw new ApiError(
      "Cannot reach the TELITI API. Is it running on port 8000?",
      0,
    );
  }

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body = await response.json();
      // FastAPI validation errors are a list of objects; flatten to something readable.
      detail = Array.isArray(body?.detail)
        ? body.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join("; ")
        : body?.detail;
    } catch {
      /* body was not JSON; fall through to the generic message */
    }
    throw new ApiError(
      response.status === 422
        ? "That text could not be analysed."
        : `Request failed (${response.status}).`,
      response.status,
      detail,
    );
  }

  return response.json() as Promise<T>;
}

export function analyze(payload: AnalyzeRequest): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/api/v1/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function health(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}
