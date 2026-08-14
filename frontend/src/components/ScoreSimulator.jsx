import { useState } from "react";
import { api } from "../api";
import RiskBadge from "./RiskBadge.jsx";
import "./ScoreSimulator.css";

const CATEGORIES = [
  "groceries", "restaurant", "gas_station", "utilities", "online_retail",
  "electronics", "travel", "entertainment", "atm_withdrawal", "jewelry",
  "healthcare", "wire_transfer",
];
const CHANNELS = ["card_present", "online", "mobile", "card_not_present"];

const PRESETS = {
  typical: {
    amount: 45,
    category: "groceries",
    channel: "card_present",
    account_age_days: 900,
    avg_amount_30d: 50,
    txn_count_1h: 1,
    txn_count_24h: 3,
    amount_sum_24h: 120,
    is_foreign: false,
    is_new_category: false,
    hour: 14,
    day_of_week: 2,
    is_weekend: false,
    is_night: false,
  },
  suspicious: {
    amount: 2400,
    category: "electronics",
    channel: "online",
    account_age_days: 900,
    avg_amount_30d: 55,
    txn_count_1h: 4,
    txn_count_24h: 6,
    amount_sum_24h: 5200,
    is_foreign: true,
    is_new_category: true,
    hour: 3,
    day_of_week: 5,
    is_weekend: true,
    is_night: true,
  },
};

export default function ScoreSimulator() {
  const [form, setForm] = useState(PRESETS.typical);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const update = (key, value) => setForm((f) => ({ ...f, [key]: value }));

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = {
        ...form,
        amount: Number(form.amount),
        account_age_days: Number(form.account_age_days),
        avg_amount_30d: Number(form.avg_amount_30d),
        txn_count_1h: Number(form.txn_count_1h),
        txn_count_24h: Number(form.txn_count_24h),
        amount_sum_24h: Number(form.amount_sum_24h),
        hour: Number(form.hour),
        day_of_week: Number(form.day_of_week),
      };
      const res = await api.scoreTransaction(payload);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel-tab">
      <div className="panel-tab-head">
        <div>
          <h2>Score a Transaction</h2>
          <p className="panel-tab-sub">
            Build a hypothetical transaction and send it through all three models to see how
            each one scores it.
          </p>
        </div>
        <div className="preset-buttons">
          <button type="button" className="btn btn-ghost" onClick={() => setForm(PRESETS.typical)}>
            Typical example
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => setForm(PRESETS.suspicious)}>
            Suspicious example
          </button>
        </div>
      </div>

      <div className="score-sim-grid">
        <form className="score-sim-form" onSubmit={submit}>
          <div className="form-row">
            <label>
              <span>Amount ($)</span>
              <input type="number" min="0.01" step="0.01" value={form.amount}
                onChange={(e) => update("amount", e.target.value)} required />
            </label>
            <label>
              <span>30-day avg amount ($)</span>
              <input type="number" min="0" step="0.01" value={form.avg_amount_30d}
                onChange={(e) => update("avg_amount_30d", e.target.value)} />
            </label>
          </div>

          <div className="form-row">
            <label>
              <span>Category</span>
              <select value={form.category} onChange={(e) => update("category", e.target.value)}>
                {CATEGORIES.map((c) => <option key={c} value={c}>{c.replace(/_/g, " ")}</option>)}
              </select>
            </label>
            <label>
              <span>Channel</span>
              <select value={form.channel} onChange={(e) => update("channel", e.target.value)}>
                {CHANNELS.map((c) => <option key={c} value={c}>{c.replace(/_/g, " ")}</option>)}
              </select>
            </label>
          </div>

          <div className="form-row">
            <label>
              <span>Transactions in last 1h</span>
              <input type="number" min="0" value={form.txn_count_1h}
                onChange={(e) => update("txn_count_1h", e.target.value)} />
            </label>
            <label>
              <span>Transactions in last 24h</span>
              <input type="number" min="0" value={form.txn_count_24h}
                onChange={(e) => update("txn_count_24h", e.target.value)} />
            </label>
          </div>

          <div className="form-row">
            <label>
              <span>Amount sum last 24h ($)</span>
              <input type="number" min="0" step="0.01" value={form.amount_sum_24h}
                onChange={(e) => update("amount_sum_24h", e.target.value)} />
            </label>
            <label>
              <span>Account age (days)</span>
              <input type="number" min="0" value={form.account_age_days}
                onChange={(e) => update("account_age_days", e.target.value)} />
            </label>
          </div>

          <div className="form-row">
            <label>
              <span>Hour of day (0-23)</span>
              <input type="number" min="0" max="23" value={form.hour}
                onChange={(e) => update("hour", e.target.value)} />
            </label>
            <label>
              <span>Day of week (0=Mon)</span>
              <input type="number" min="0" max="6" value={form.day_of_week}
                onChange={(e) => update("day_of_week", e.target.value)} />
            </label>
          </div>

          <div className="form-row form-row-checks">
            <label className="check-label">
              <input type="checkbox" checked={form.is_foreign}
                onChange={(e) => update("is_foreign", e.target.checked)} />
              <span>Foreign transaction</span>
            </label>
            <label className="check-label">
              <input type="checkbox" checked={form.is_new_category}
                onChange={(e) => update("is_new_category", e.target.checked)} />
              <span>New merchant category for this account</span>
            </label>
            <label className="check-label">
              <input type="checkbox" checked={form.is_night}
                onChange={(e) => update("is_night", e.target.checked)} />
              <span>Late night (12&ndash;5am)</span>
            </label>
            <label className="check-label">
              <input type="checkbox" checked={form.is_weekend}
                onChange={(e) => update("is_weekend", e.target.checked)} />
              <span>Weekend</span>
            </label>
          </div>

          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? "Scoring\u2026" : "Score transaction"}
          </button>
        </form>

        <div className="score-sim-result">
          {error && <div className="banner banner-error">{error}</div>}
          {!result && !error && (
            <div className="empty-state">Fill out a transaction and score it to see results.</div>
          )}
          {result && (
            <>
              <div className="result-ensemble">
                <span className="result-ensemble-label mono">Ensemble score</span>
                <span className="result-ensemble-value mono">
                  {(result.scores.ensemble * 100).toFixed(1)}%
                </span>
                <RiskBadge level={result.scores.risk_level} pulse={result.scores.risk_level === "critical"} />
              </div>
              <div className="result-breakdown">
                {["random_forest", "xgboost", "deep_learning"].map((m) => (
                  <div className="score-row" key={m}>
                    <span className="score-row-label mono">{m.replace(/_/g, " ")}</span>
                    <div className="score-row-bar">
                      <div className="score-row-fill" style={{ width: `${result.scores[m] * 100}%` }} />
                    </div>
                    <span className="score-row-value mono">{(result.scores[m] * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
