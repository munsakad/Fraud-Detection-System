import { useState } from "react";
import RiskBadge from "./RiskBadge.jsx";
import "./LiveFeed.css";

function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour12: false });
}

export default function LiveFeed({ events }) {
  const [selected, setSelected] = useState(null);

  return (
    <div className="panel-tab">
      <div className="panel-tab-head">
        <div>
          <h2>Live Feed</h2>
          <p className="panel-tab-sub">
            A replayed stream of held-out test transactions, scored by all three models the
            instant they arrive. Click a row for the full score breakdown.
          </p>
        </div>
      </div>
      <div className="live-feed">
      <div className="live-feed-table-wrap">
        <table className="live-feed-table">
          <thead>
            <tr>
              <th className="mono">Time</th>
              <th className="mono">Account</th>
              <th className="mono">Amount</th>
              <th className="mono">Category</th>
              <th className="mono">Channel</th>
              <th className="mono">Ensemble</th>
              <th className="mono">Risk</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e) => (
              <tr
                key={e.event_id}
                className={`feed-row risk-row-${e.scores.risk_level} ${
                  selected?.event_id === e.event_id ? "is-selected" : ""
                }`}
                onClick={() => setSelected(e)}
              >
                <td className="mono">{formatTime(e.timestamp)}</td>
                <td className="mono">{e.account_id}</td>
                <td className="mono">${e.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                <td>{e.category.replace(/_/g, " ")}</td>
                <td>{e.channel.replace(/_/g, " ")}</td>
                <td className="mono">{(e.scores.ensemble * 100).toFixed(1)}%</td>
                <td>
                  <RiskBadge level={e.scores.risk_level} pulse={e.scores.risk_level === "critical"} />
                </td>
              </tr>
            ))}
            {events.length === 0 && (
              <tr>
                <td colSpan={7} className="feed-empty">
                  Waiting for the live feed to connect&hellip;
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="feed-detail">
          <div className="feed-detail-head">
            <div>
              <div className="feed-detail-title mono">{selected.txn_id}</div>
              <div className="feed-detail-sub">{selected.account_id}</div>
            </div>
            <button className="feed-detail-close" onClick={() => setSelected(null)}>
              &times;
            </button>
          </div>

          <div className="feed-detail-amount mono">
            ${selected.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
          <div className="feed-detail-meta">
            {selected.category.replace(/_/g, " ")} &middot; {selected.channel.replace(/_/g, " ")}
            {selected.is_foreign && <span className="feed-detail-flag"> &middot; foreign</span>}
          </div>

          <div className="feed-detail-scores">
            {["random_forest", "xgboost", "deep_learning"].map((m) => (
              <div className="score-row" key={m}>
                <span className="score-row-label mono">{m.replace(/_/g, " ")}</span>
                <div className="score-row-bar">
                  <div
                    className="score-row-fill"
                    style={{ width: `${selected.scores[m] * 100}%` }}
                  />
                </div>
                <span className="score-row-value mono">
                  {(selected.scores[m] * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>

          <div className="feed-detail-ensemble">
            <span>Ensemble score</span>
            <RiskBadge level={selected.scores.risk_level} />
            <span className="mono">{(selected.scores.ensemble * 100).toFixed(1)}%</span>
          </div>

          <div className="feed-detail-truth">
            Ground truth (held out from the model):{" "}
            <strong className={selected.ground_truth_is_fraud ? "truth-fraud" : "truth-legit"}>
              {selected.ground_truth_is_fraud ? "Fraudulent" : "Legitimate"}
            </strong>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
