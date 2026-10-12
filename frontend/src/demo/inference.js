// Pure-JS port of backend/app/inference.py, driven by the weights that
// ml/export_web.py exports. No Vite or browser APIs in here, so CI can run it
// under plain Node (scripts/check-parity.mjs) and compare against Python.

export const RISK_THRESHOLDS = { medium: 0.2, high: 0.5, critical: 0.8 };

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const round4 = (v) => Math.round(v * 1e4) / 1e4;

// StandardScaler -> boolean passthrough -> OneHotEncoder(handle_unknown="ignore"),
// in the same column order as the training ColumnTransformer
export function transform(pre, row) {
  const x = [];
  pre.numeric.forEach((k, i) => x.push((Number(row[k]) - pre.mean[i]) / pre.scale[i]));
  pre.boolean.forEach((k) => x.push(row[k] ? 1 : 0));
  pre.categorical.forEach((k, i) => {
    for (const c of pre.categories[i]) x.push(String(row[k]) === c ? 1 : 0);
  });
  return x;
}

// MLP with BatchNorm already folded into each Linear layer (eval mode)
function mlp(layers, x) {
  let h = x;
  layers.forEach((L, li) => {
    const out = L.W.map((w, j) => {
      let s = L.b[j];
      for (let k = 0; k < w.length; k++) s += w[k] * h[k];
      return s;
    });
    h = li < layers.length - 1 ? out.map((v) => (v > 0 ? v : 0)) : out;
  });
  return sigmoid(h[0]);
}

// sklearn trees compare float32 features with "<=" against float64 thresholds
function randomForest(trees, xf) {
  let sum = 0;
  for (const t of trees) {
    let i = 0;
    while (t.f[i] !== -1) i = xf[t.f[i]] <= t.t[i] ? t.l[i] : t.r[i];
    sum += t.p[i];
  }
  return sum / trees.length;
}

// xgboost compares float32 features with "<" against float32 split conditions
function xgboost(model, xf) {
  let margin = model.base_margin;
  for (const t of model.trees) {
    let i = 0;
    while (t.f[i] !== -1) i = xf[t.f[i]] < Math.fround(t.t[i]) ? t.y[i] : t.n[i];
    margin += t.l[i];
  }
  return sigmoid(margin);
}

export function riskLevel(p) {
  if (p >= RISK_THRESHOLDS.critical) return "critical";
  if (p >= RISK_THRESHOLDS.high) return "high";
  if (p >= RISK_THRESHOLDS.medium) return "medium";
  return "low";
}

export function scoreRow(model, row) {
  const x = transform(model.preprocessor, row);
  const xf = x.map(Math.fround);
  const rf = randomForest(model.random_forest, xf);
  const xgb = xgboost(model.xgboost, xf);
  const dl = mlp(model.mlp, x);
  const ensemble = (rf + xgb + dl) / 3;
  return {
    random_forest: round4(rf),
    xgboost: round4(xgb),
    deep_learning: round4(dl),
    ensemble: round4(ensemble),
    risk_level: riskLevel(ensemble),
  };
}

// mirrors TransactionScoreRequest.to_feature_row in backend/app/schemas.py
export function toFeatureRow(req) {
  const amount = Number(req.amount);
  const avg = req.avg_amount_30d ?? amount;
  return {
    amount,
    category: req.category,
    channel: req.channel,
    account_age_days: req.account_age_days ?? 365,
    avg_amount_30d: avg,
    amount_ratio_to_avg: req.amount_ratio_to_avg ?? amount / Math.max(avg, 1),
    txn_count_1h: req.txn_count_1h ?? 1,
    txn_count_24h: req.txn_count_24h ?? 1,
    amount_sum_24h: req.amount_sum_24h ?? amount,
    is_foreign: !!req.is_foreign,
    is_new_category: !!req.is_new_category,
    hour: req.hour ?? 12,
    day_of_week: req.day_of_week ?? 2,
    is_weekend: !!req.is_weekend,
    is_night: !!req.is_night,
  };
}
