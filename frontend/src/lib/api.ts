import type { AnalystDecision, DecisionLogEntry, EvalResults, ScoredOrder, SimulateResponse } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${path} failed (${res.status}): ${body}`);
  }
  return res.json() as Promise<T>;
}

export function getMetrics(): Promise<EvalResults> {
  return request<EvalResults>("/metrics");
}

export function simulateOrders(n: number, riskShift: number): Promise<SimulateResponse> {
  return request<SimulateResponse>("/simulate", {
    method: "POST",
    body: JSON.stringify({ n, risk_shift: riskShift }),
  });
}

export interface ScoreRequestBody {
  order_value: number;
  product_category: string;
  payment_mode: string;
  delivery_pincode_tier: string;
  discount_applied?: number;
  customer_id?: string;
}

export function scoreOrder(body: ScoreRequestBody): Promise<ScoredOrder> {
  return request<ScoredOrder>("/score", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function postDecision(orderId: string, decision: AnalystDecision) {
  return request<{ order_id: string; decision: AnalystDecision; decided_at: string }>("/decide", {
    method: "POST",
    body: JSON.stringify({ order_id: orderId, decision }),
  });
}

export function getDecisions(limit = 100): Promise<{ decisions: DecisionLogEntry[] }> {
  return request(`/decisions?limit=${limit}`);
}
