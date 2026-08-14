import "./StatsBar.css";

export default function StatsBar({ connected, stats }) {
  const flagged = stats?.flagged_counts || {};
  return (
    <div className="stats-bar">
      <div className="stats-conn">
        <span className={`conn-dot ${connected ? "is-live" : ""}`} />
        <span className="mono">{connected ? "LIVE" : "RECONNECTING\u2026"}</span>
      </div>
      <div className="stats-cell">
        <span className="stats-label">Scored</span>
        <span className="stats-value mono">{(stats?.total_scored ?? 0).toLocaleString()}</span>
      </div>
      <div className="stats-cell">
        <span className="stats-label">Flagged rate</span>
        <span className="stats-value mono">
          {((stats?.flagged_rate ?? 0) * 100).toFixed(1)}%
        </span>
      </div>
      <div className="stats-cell">
        <span className="stats-label">Critical</span>
        <span className="stats-value mono stats-critical">{flagged.critical ?? 0}</span>
      </div>
      <div className="stats-cell">
        <span className="stats-label">High</span>
        <span className="stats-value mono stats-high">{flagged.high ?? 0}</span>
      </div>
      <div className="stats-cell">
        <span className="stats-label">Medium</span>
        <span className="stats-value mono stats-medium">{flagged.medium ?? 0}</span>
      </div>
    </div>
  );
}
