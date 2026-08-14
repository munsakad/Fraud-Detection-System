# Sentry &mdash; Real-Time Fraud Detection System

Detects suspicious banking transactions using three models trained side-by-side
&mdash; Random Forest, XGBoost, and a PyTorch deep learning model &mdash; with a
live dashboard that streams transactions in and shows how each model scores them.

## Architecture

```
fraud-detection/
  ml/          data generation + model training (Python)
  backend/     FastAPI inference API + live WebSocket feed
  frontend/    React dashboard (Vite + Recharts)
```

```
generate_data.py  -->  transactions.csv  -->  train_models.py  -->  models/*.joblib, *.pt
                                                                          |
                                                                          v
                                                              FastAPI backend loads models
                                                                          |
                                                                          v
                                                    WebSocket live feed  +  REST scoring API
                                                                          |
                                                                          v
                                                                 React dashboard
```

## Why the data is synthetic (and how it's still realistic)

There's no real banking dataset here &mdash; I don't have access to real transaction
data, and there was no network access in the environment I built this in to pull
something like Kaggle's credit card fraud dataset. Instead, `ml/generate_data.py`
generates one from domain rules rather than pure randomness:

- Every account has its **own** spending baseline (amount distribution, preferred
  merchant categories, active hours). Fraud is injected as a deviation from that
  account's own history (unusually large amount vs. their average, an unfamiliar
  merchant category, card-not-present, foreign, odd hour), plus short bursts of
  rapid-fire transactions ("card testing") &mdash; not just globally rare rows.
  That's what gives the behavioral/velocity features real signal to learn from.
- All rolling features (30-day average spend, 1h/24h transaction velocity) are
  computed **causally** &mdash; only from transactions at or before the current
  one. A model trained on features that peek into the future looks great in
  a notebook and is useless in production.
- The resulting fraud rate (~0.5%) is in the same range as published real-world
  card fraud rates, on purpose.

If you have a real transaction dataset (e.g. Kaggle's credit card fraud dataset),
you can drop your own CSV at `ml/data/transactions.csv` with the same columns
`train_models.py` expects (see `NUMERIC_FEATURES` / `BOOLEAN_FEATURES` /
`CATEGORICAL_FEATURES` at the top of that file) and skip `generate_data.py`.

## Why three models, and how they're compared honestly

- **Random Forest** &mdash; strong baseline, handles nonlinear feature
  interactions well, `class_weight="balanced"` for the imbalance.
- **XGBoost** &mdash; usually the strongest tabular performer; imbalance handled
  via `scale_pos_weight` rather than resampling.
- **Deep Learning (PyTorch MLP)** &mdash; a small feedforward network
  (64&rarr;32&rarr;1) with batch norm and dropout, trained with a class-weighted
  `BCEWithLogitsLoss`.

All three share **one preprocessing pipeline** (fit on train only) so the
comparison is apples-to-apples, and are evaluated on a **time-based** test split
(last 20% of transactions by timestamp) rather than a random split &mdash; a
random split would leak future account behavior into training and quietly
inflate every metric.

With <1% fraud, accuracy is nearly meaningless (predicting "not fraud" every
time already scores >99%). The dashboard leads with **precision, recall, F1,
ROC-AUC, and PR-AUC** instead, and PR-AUC specifically is the number to trust
most under this much class imbalance.

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Generate data and train the models

```bash
cd ml
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py         # -> data/transactions.csv (~110k rows, ~30s)
python train_models.py          # -> models/*.joblib, models/deep_learning.pt
                                 #    reports/metrics.json, roc_curves.json, ...
                                 #    (a few minutes; XGBoost + the MLP are the slow parts)
```

You should see per-model precision/recall/F1/ROC-AUC/PR-AUC printed at the end.

### 2. Run the backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # defaults already point at ../ml
uvicorn app.main:app --reload --port 8000
```

If `models/` from step 1 isn't found, the API still starts but returns clear
400s telling you to run `train_models.py` first; the live simulator just won't
start.

### 3. Run the frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open `http://localhost:5173`. You'll see live-scored transactions streaming in
immediately (replayed from the held-out test set), a model comparison tab with
ROC/PR curves and feature importance, and a form to score your own hypothetical
transaction.

## API reference

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Whether models are loaded and ready |
| `WS /ws/live` | Live feed of scored transactions (replayed from the test set) |
| `GET /api/transactions/recent?limit=50` | Recent scored transactions (REST fallback) |
| `GET /api/transactions/stats` | Running totals: scored count, flagged rate by risk level |
| `POST /api/transactions/score` | Score a custom transaction against all 3 models |
| `GET /api/models/metrics` | Precision/recall/F1/ROC-AUC/PR-AUC/confusion matrix per model |
| `GET /api/models/roc-curves` | ROC curve points per model |
| `GET /api/models/pr-curves` | Precision-recall curve points per model |
| `GET /api/models/feature-importance` | Top 15 features per model (RF, XGBoost) |

## How "live" transactions and risk levels work

The WebSocket feed replays transactions from the **held-out test split**
(transactions the models never trained on) at a configurable interval
(`SIMULATOR_INTERVAL_SECONDS` in `backend/.env`, default 1.5s), scoring each one
in real time as it "arrives." Each transaction's true label (fraud or not) is
kept and shown in the detail panel so you can see how the models did, but it's
never given to the models as an input feature.

Risk levels are thresholds on the **ensemble score** (mean of the three model
probabilities), configurable in `backend/app/config.py`:

| Level | Ensemble score |
|---|---|
| Low | < 0.20 |
| Medium | 0.20 &ndash; 0.50 |
| High | 0.50 &ndash; 0.80 |
| Critical | &ge; 0.80 |

## Limitations & what's simplified for a demo

- **No persistence.** Scored-transaction history lives in memory (a ring
  buffer) and resets on backend restart. Fine for a demo; add a database if you
  want history to survive restarts.
- **No auth.** The scoring API and dashboard are open. Add auth before
  deploying this anywhere shared.
- **Single-process simulator.** The live feed is one asyncio loop, not a real
  message queue (Kafka/Kinesis, which real fraud pipelines use). The inference
  layer (`backend/app/inference.py`) is decoupled from it, though, so swapping
  the simulator for a real stream consumer is a matter of calling
  `ModelBundle.score()` from wherever the real events arrive.
- **Threshold is fixed at 0.5** for the reported precision/recall in
  `metrics.json`. Real fraud systems tune the threshold to a target recall or
  a cost-weighted operating point; `reports/pr_curves.json` has what you need
  to pick a different one.
- **Synthetic data**, as covered above &mdash; directionally realistic, not a
  substitute for validating against real transaction data before trusting this
  for anything beyond a demo/portfolio project.

## Extending it

- Swap in a real dataset by matching the column schema in `ml/train_models.py`.
- Add SHAP values to `train_models.py` for per-transaction explainability
  instead of just global feature importance.
- Add a `/api/transactions/{id}/feedback` endpoint so an analyst can label a
  transaction as confirmed fraud/not-fraud, and retrain periodically on that
  feedback (this is how real fraud systems stay current as fraud patterns
  shift).
- Persist scored transactions to Postgres instead of the in-memory ring buffer.
