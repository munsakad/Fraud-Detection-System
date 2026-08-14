import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import "./ModelComparison.css";

const MODEL_COLORS = {
  random_forest: "#5b8def",
  xgboost: "#e8a33d",
  deep_learning: "#a685e2",
};
const MODEL_LABELS = {
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  deep_learning: "Deep Learning",
};

const tooltipStyle = {
  background: "#161d26",
  border: "1px solid #232c37",
  borderRadius: 4,
  fontSize: 12,
};

function metricsToBarData(metrics) {
  const fields = ["precision", "recall", "f1", "roc_auc", "pr_auc"];
  return fields.map((field) => {
    const row = { metric: field.replace("_", " ").toUpperCase() };
    for (const model of Object.keys(metrics)) {
      row[model] = Number((metrics[model][field] * 100).toFixed(1));
    }
    return row;
  });
}

function mergeCurves(curves, xKey, yKey) {
  // recharts wants one array with a shared x-domain; since each model has its
  // own threshold sweep, merge by index position (all curves are resampled to
  // the same point count server-side) rather than trying to align x exactly.
  const models = Object.keys(curves);
  const len = Math.max(...models.map((m) => curves[m].length));
  const merged = [];
  for (let i = 0; i < len; i++) {
    const point = { i };
    for (const m of models) {
      const p = curves[m][Math.min(i, curves[m].length - 1)];
      point[`${m}_${xKey}`] = p[xKey];
      point[`${m}_${yKey}`] = p[yKey];
    }
    merged.push(point);
  }
  return merged;
}

function ConfusionMatrix({ cm }) {
  return (
    <table className="cm-table">
      <tbody>
        <tr>
          <td className="cm-label" />
          <td className="cm-label mono">pred legit</td>
          <td className="cm-label mono">pred fraud</td>
        </tr>
        <tr>
          <td className="cm-label mono">actual legit</td>
          <td className="cm-cell cm-good mono">{cm.true_negative}</td>
          <td className="cm-cell cm-bad mono">{cm.false_positive}</td>
        </tr>
        <tr>
          <td className="cm-label mono">actual fraud</td>
          <td className="cm-cell cm-bad mono">{cm.false_negative}</td>
          <td className="cm-cell cm-good mono">{cm.true_positive}</td>
        </tr>
      </tbody>
    </table>
  );
}

export default function ModelComparison() {
  const [metrics, setMetrics] = useState(null);
  const [roc, setRoc] = useState(null);
  const [pr, setPr] = useState(null);
  const [importance, setImportance] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      api.modelMetrics(),
      api.rocCurves(),
      api.prCurves(),
      api.featureImportance(),
    ])
      .then(([m, r, p, fi]) => {
        setMetrics(m);
        setRoc(r);
        setPr(p);
        setImportance(fi);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="panel-tab">
        <div className="banner banner-error">{error}</div>
      </div>
    );
  }
  if (!metrics) {
    return (
      <div className="panel-tab">
        <div className="empty-state">Loading model comparison&hellip;</div>
      </div>
    );
  }

  const barData = metricsToBarData(metrics.models);
  const rocData = mergeCurves(roc, "fpr", "tpr");
  const prData = mergeCurves(pr, "recall", "precision");
  const models = Object.keys(metrics.models);

  return (
    <div className="panel-tab">
      <div className="panel-tab-head">
        <div>
          <h2>Model Comparison</h2>
          <p className="panel-tab-sub">
            Evaluated on a held-out, time-based test split ({metrics.n_test.toLocaleString()}{" "}
            transactions, {(metrics.fraud_rate_test * 100).toFixed(2)}% fraud) never seen during
            training.
          </p>
        </div>
      </div>

      <div className="mc-grid">
        <div className="mc-card mc-card-wide">
          <div className="mc-card-title mono">Precision / Recall / F1 / ROC-AUC / PR-AUC</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke="#1b232c" vertical={false} />
              <XAxis dataKey="metric" tick={{ fill: "#8a99a8", fontSize: 10 }} axisLine={{ stroke: "#232c37" }} tickLine={false} />
              <YAxis tick={{ fill: "#8a99a8", fontSize: 10 }} tickLine={false} axisLine={false} domain={[0, 100]} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#8a99a8" }} />
              <Legend
                formatter={(v) => <span style={{ color: "#8a99a8", fontSize: 11 }}>{MODEL_LABELS[v]}</span>}
              />
              {models.map((m) => (
                <Bar key={m} dataKey={m} fill={MODEL_COLORS[m]} radius={[3, 3, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mc-card">
          <div className="mc-card-title mono">ROC Curve</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={rocData} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke="#1b232c" vertical={false} />
              <XAxis
                dataKey="random_forest_fpr"
                type="number"
                domain={[0, 1]}
                tick={{ fill: "#8a99a8", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: "#232c37" }}
                label={{ value: "FPR", position: "insideBottom", offset: -2, fill: "#56636f", fontSize: 10 }}
              />
              <YAxis domain={[0, 1]} tick={{ fill: "#8a99a8", fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#8a99a8" }} />
              {models.map((m) => (
                <Line
                  key={m}
                  type="monotone"
                  dataKey={`${m}_tpr`}
                  stroke={MODEL_COLORS[m]}
                  strokeWidth={2}
                  dot={false}
                  name={MODEL_LABELS[m]}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="mc-card">
          <div className="mc-card-title mono">Precision-Recall Curve</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={prData} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke="#1b232c" vertical={false} />
              <XAxis
                dataKey="random_forest_recall"
                type="number"
                domain={[0, 1]}
                tick={{ fill: "#8a99a8", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: "#232c37" }}
                label={{ value: "Recall", position: "insideBottom", offset: -2, fill: "#56636f", fontSize: 10 }}
              />
              <YAxis domain={[0, 1]} tick={{ fill: "#8a99a8", fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#8a99a8" }} />
              {models.map((m) => (
                <Line
                  key={m}
                  type="monotone"
                  dataKey={`${m}_precision`}
                  stroke={MODEL_COLORS[m]}
                  strokeWidth={2}
                  dot={false}
                  name={MODEL_LABELS[m]}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {models.map((m) => (
          <div className="mc-card" key={m}>
            <div className="mc-card-title mono">{MODEL_LABELS[m]} &mdash; confusion matrix</div>
            <ConfusionMatrix cm={metrics.models[m].confusion_matrix} />
          </div>
        ))}

        {importance &&
          Object.keys(importance).map((m) => (
            <div className="mc-card mc-card-wide" key={`fi-${m}`}>
              <div className="mc-card-title mono">{MODEL_LABELS[m]} &mdash; top features</div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart
                  data={importance[m]}
                  layout="vertical"
                  margin={{ top: 4, right: 16, left: 8, bottom: 0 }}
                >
                  <CartesianGrid stroke="#1b232c" horizontal={false} />
                  <XAxis type="number" tick={{ fill: "#8a99a8", fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis
                    dataKey="feature"
                    type="category"
                    width={140}
                    tick={{ fill: "#8a99a8", fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#8a99a8" }} />
                  <Bar dataKey="importance" fill={MODEL_COLORS[m]} radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
      </div>
    </div>
  );
}
