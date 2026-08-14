const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health() {
    return request("/api/health");
  },
  recentTransactions(limit = 50) {
    return request(`/api/transactions/recent?limit=${limit}`);
  },
  stats() {
    return request("/api/transactions/stats");
  },
  scoreTransaction(payload) {
    return request("/api/transactions/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
  modelMetrics() {
    return request("/api/models/metrics");
  },
  rocCurves() {
    return request("/api/models/roc-curves");
  },
  prCurves() {
    return request("/api/models/pr-curves");
  },
  featureImportance() {
    return request("/api/models/feature-importance");
  },
};
