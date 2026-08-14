import { useEffect, useState } from "react";
import { api } from "./api";
import { useLiveFeed } from "./useLiveFeed.js";
import StatsBar from "./components/StatsBar.jsx";
import LiveFeed from "./components/LiveFeed.jsx";
import ModelComparison from "./components/ModelComparison.jsx";
import ScoreSimulator from "./components/ScoreSimulator.jsx";
import "./App.css";

const TABS = [
  { id: "feed", label: "Live Feed", glyph: "\u25CF" },
  { id: "compare", label: "Model Comparison", glyph: "\u2317" },
  { id: "score", label: "Score a Transaction", glyph: ">_" },
];

export default function App() {
  const [tab, setTab] = useState("feed");
  const { events, connected } = useLiveFeed();
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "down", models_ready: false }));
  }, []);

  useEffect(() => {
    if (!connected) return;
    const interval = setInterval(() => {
      api.stats().then(setStats).catch(() => {});
    }, 2000);
    return () => clearInterval(interval);
  }, [connected]);

  const modelsReady = health?.models_ready;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark mono">&#9673;</span>
          <span className="brand-name">Sentry</span>
        </div>
        <div className="brand-tagline">Real-time fraud detection</div>

        <nav className="tab-nav">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab-nav-item ${tab === t.id ? "is-active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              <span className="tab-nav-glyph mono">{t.glyph}</span>
              {t.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-models">
          <div className="sidebar-models-title mono">Models</div>
          {["Random Forest", "XGBoost", "Deep Learning"].map((m) => (
            <div className="sidebar-model-row" key={m}>
              <span className={`model-dot ${modelsReady ? "is-ready" : ""}`} />
              {m}
            </div>
          ))}
          {health && !modelsReady && (
            <div className="sidebar-warning">
              No trained models found. Run <code>train_models.py</code> in{" "}
              <code>ml/</code>, then restart the backend.
            </div>
          )}
        </div>

        <div className="sidebar-foot mono">RF &middot; XGBoost &middot; Deep Learning</div>
      </aside>

      <main className="main">
        <StatsBar connected={connected} stats={stats} />
        {tab === "feed" && <LiveFeed events={events} />}
        {tab === "compare" && <ModelComparison />}
        {tab === "score" && <ScoreSimulator />}
      </main>
    </div>
  );
}
