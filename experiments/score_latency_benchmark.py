"""
RiskGuard -- /score latency benchmark (one-off, not run in the UI).

SHAP computation can be slow; this measures actual end-to-end latency of
POST /score against a running API so p50/p95 numbers are ready to state if
asked, rather than guessed. Varies order fields and customer_id across
calls so it's not just measuring one cached code path.

Usage:
    python experiments/score_latency_benchmark.py [n_calls] [base_url]
"""

import sys
import time
import random

import requests

CATEGORIES = ["footwear", "apparel", "electronics_accessories", "groceries", "home_goods", "beauty"]
PAYMENT_MODES = ["COD", "prepaid_card", "UPI", "wallet"]
TIERS = ["metro", "tier2", "tier3"]


def main(n_calls: int, base_url: str):
    rng = random.Random(42)
    latencies_ms = []

    # Warm up (first call pays for any lazy imports / JIT-ish overhead)
    requests.post(f"{base_url}/score", json={
        "order_value": 1000, "product_category": "beauty", "payment_mode": "UPI",
        "delivery_pincode_tier": "metro",
    })

    for i in range(n_calls):
        body = {
            "order_value": round(rng.uniform(200, 15000), 2),
            "product_category": rng.choice(CATEGORIES),
            "payment_mode": rng.choice(PAYMENT_MODES),
            "delivery_pincode_tier": rng.choice(TIERS),
            "discount_applied": round(rng.uniform(0, 30), 1),
            "customer_id": f"BENCH{i % 50:03d}",  # 50 distinct customers, some repeat (real history) some not
        }
        start = time.perf_counter()
        resp = requests.post(f"{base_url}/score", json=body)
        elapsed_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    n = len(latencies_ms)
    p50 = latencies_ms[int(n * 0.50)]
    p95 = latencies_ms[min(int(n * 0.95), n - 1)]
    p99 = latencies_ms[min(int(n * 0.99), n - 1)]

    print(f"POST /score latency over {n} calls (varying order + customer_id):")
    print(f"  min: {latencies_ms[0]:.1f} ms")
    print(f"  p50: {p50:.1f} ms")
    print(f"  p95: {p95:.1f} ms")
    print(f"  p99: {p99:.1f} ms")
    print(f"  max: {latencies_ms[-1]:.1f} ms")
    print(f"  mean: {sum(latencies_ms) / n:.1f} ms")


if __name__ == "__main__":
    n_calls = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
    main(n_calls, base_url)
