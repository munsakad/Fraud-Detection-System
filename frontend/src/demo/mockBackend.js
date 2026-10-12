// Stand-in for the FastAPI backend when the app is built as a static
// GitHub Pages demo (VITE_DEMO=true). It intercepts the same REST and
// WebSocket endpoints the real backend serves, so none of the React
// components change. Scores come from the real trained models, exported by
// ml/export_web.py and evaluated in the browser by ./inference.js.
import { scoreRow, toFeatureRow } from "./inference.js";

const BASE = import.meta.env.BASE_URL;
const REPO_URL = "https://github.com/munsakad/Fraud-Detection-System";
const INTERVAL_MS = 1500; // matches SIMULATOR_INTERVAL_SECONDS
const BUFFER_SIZE = 500; // matches LIVE_BUFFER_SIZE

const realFetch = window.fetch.bind(window);
const cache = {};
const loadJSON = (path) =>
  (cache[path] ??= realFetch(BASE + "demo/" + path).then((r) => {
    if (!r.ok) throw new Error("Could not load demo/" + path + " (" + r.status + ")");
    return r.json();
  }));

// ---- live simulator (port of backend/app/simulator.py) ----------------------
const buffer = [];
const stats = { total: 0, flagged: { low: 0, medium: 0, high: 0, critical: 0 } };
const listeners = new Set();
let simulator = null;

function emit(pool, timestamp) {
  const src = pool[Math.floor(Math.random() * pool.length)];
  const event = { event_id: crypto.randomUUID(), timestamp, ...src };
  buffer.push(event);
  if (buffer.length > BUFFER_SIZE) buffer.shift();
  stats.total += 1;
  stats.flagged[event.scores.risk_level] += 1;
  listeners.forEach((fn) => fn(event));
}

function startSimulator() {
  simulator ??= loadJSON("events.json").then((pool) => {
    // backfill a short history so the feed isn't empty on first load
    const now = Date.now();
    for (let i = 20; i > 0; i--) emit(pool, new Date(now - i * INTERVAL_MS).toISOString());
    setInterval(() => emit(pool, new Date().toISOString()), INTERVAL_MS);
  });
  return simulator;
}

class DemoLiveSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.listener = (event) => this.deliver(event);
    startSimulator().then(() => {
      if (this.readyState === 3) return;
      this.readyState = 1;
      this.onopen?.({ type: "open" });
      // like the backend: send a snapshot of recent events on connect
      buffer.slice(-30).forEach((event) => this.deliver(event));
      listeners.add(this.listener);
    });
  }
  deliver(event) {
    if (this.readyState === 1) this.onmessage?.({ data: JSON.stringify(event) });
  }
  send() {}
  close() {
    if (this.readyState === 3) return;
    this.readyState = 3;
    listeners.delete(this.listener);
    this.onclose?.({ type: "close" });
  }
}

// ---- REST routes -------------------------------------------------------------
const json = (data, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });

const REPORTS = {
  metrics: "metrics",
  "roc-curves": "roc_curves",
  "pr-curves": "pr_curves",
  "feature-importance": "feature_importance",
};

async function route(url, init = {}) {
  const u = new URL(url, window.location.href);
  const path = u.pathname.slice(u.pathname.indexOf("/api/"));

  if (path === "/api/health") return json({ status: "ok", models_ready: true });

  if (path === "/api/transactions/recent") {
    await startSimulator();
    const limit = Number(u.searchParams.get("limit") || 50);
    return json(buffer.slice(-limit).reverse());
  }

  if (path === "/api/transactions/stats") {
    const { medium, high, critical } = stats.flagged;
    const flaggedTotal = medium + high + critical;
    return json({
      total_scored: stats.total,
      flagged_counts: { ...stats.flagged },
      flagged_rate: stats.total ? flaggedTotal / stats.total : 0,
    });
  }

  if (path === "/api/transactions/score" && (init.method || "GET").toUpperCase() === "POST") {
    const body = JSON.parse(init.body || "{}");
    if (!(Number(body.amount) > 0)) return json({ detail: "amount must be greater than 0" }, 422);
    if (!body.category || !body.channel) return json({ detail: "category and channel are required" }, 422);
    const model = await loadJSON("model.json");
    return json({ input: body, scores: scoreRow(model, toFeatureRow(body)) });
  }

  const report = path.startsWith("/api/models/") && REPORTS[path.split("/").pop()];
  if (report) return json(await loadJSON("reports/" + report + ".json"));

  return json({ detail: "Not Found" }, 404);
}

function addBanner() {
  const el = document.createElement("aside");
  el.setAttribute("aria-label", "About this demo");
  el.style.cssText =
    "position:fixed;right:14px;bottom:14px;z-index:9999;max-width:360px;padding:12px 34px 12px 14px;" +
    "border-radius:6px;background:#161d26;border:1px solid #2b3642;color:#b8c4d0;" +
    "font:12px/1.5 system-ui,sans-serif;box-shadow:0 6px 20px rgba(0,0,0,.4)";
  el.innerHTML =
    '<strong style="color:#e6edf3">Static demo build.</strong> The FastAPI backend is swapped for an ' +
    "in-browser port. GitHub Actions generates the data, trains all three models and exports them, " +
    "so every score here comes from the real models. " +
    '<a href="' + REPO_URL + '" style="color:#5b8def">View source</a>' +
    '<button aria-label="Dismiss" style="position:absolute;top:6px;right:8px;background:none;border:0;' +
    'color:#8a99a8;font-size:16px;cursor:pointer">&times;</button>';
  el.querySelector("button").onclick = () => el.remove();
  document.body.appendChild(el);
}

export function install() {
  window.fetch = (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    if (url.includes("/api/")) {
      return route(url, init).catch((e) => json({ detail: e.message }, 500));
    }
    return realFetch(input, init);
  };
  const RealWebSocket = window.WebSocket;
  window.WebSocket = function (url, protocols) {
    return String(url).includes("/ws/live") ? new DemoLiveSocket(url) : new RealWebSocket(url, protocols);
  };
  addBanner();
}
