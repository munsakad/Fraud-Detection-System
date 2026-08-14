"""
Trains and compares three fraud-detection models on data/transactions.csv:
Random Forest, XGBoost, and a small PyTorch MLP (the "Deep Learning" model).

Design choices that matter for a real fraud pipeline (not just a toy demo):

- Time-based split, not a random split. Transactions after a cutoff date are the
  test set. A random split would leak future behavioral patterns into training
  and quietly inflate every metric.
- Class imbalance is handled explicitly per model (class_weight for RF,
  scale_pos_weight for XGBoost, a weighted loss for the MLP) rather than
  resampling the data, so the reported metrics reflect the real class balance.
- Accuracy is reported but not treated as the headline number - with <1% fraud,
  a model that predicts "not fraud" every time scores >99% accuracy and is
  useless. ROC-AUC, PR-AUC, precision, recall, and F1 are the numbers that
  actually matter here, and PR-AUC in particular for this class imbalance.
- One shared preprocessing pipeline (fit on train only) feeds all three models,
  so a) the comparison is apples-to-apples and b) the backend only needs to
  load one preprocessor to score all three.

Run: python train_models.py
Outputs: models/*.joblib, models/deep_learning.pt, reports/metrics.json,
         reports/roc_curves.json, reports/pr_curves.json,
         reports/feature_importance.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/transactions.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "amount", "account_age_days", "avg_amount_30d", "amount_ratio_to_avg",
    "txn_count_1h", "txn_count_24h", "amount_sum_24h", "hour", "day_of_week",
]
BOOLEAN_FEATURES = ["is_foreign", "is_new_category", "is_weekend", "is_night"]
CATEGORICAL_FEATURES = ["category", "channel"]
TARGET = "is_fraud"

RANDOM_STATE = 42


def load_and_split():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(bool).astype(int)

    cutoff_idx = int(len(df) * 0.8)
    train_df = df.iloc[:cutoff_idx].reset_index(drop=True)
    test_df = df.iloc[cutoff_idx:].reset_index(drop=True)
    print(
        f"Train: {len(train_df):,} rows ({train_df[TARGET].sum()} fraud) | "
        f"Test: {len(test_df):,} rows ({test_df[TARGET].sum()} fraud) "
        f"| split at {train_df['timestamp'].max()}"
    )
    return train_df, test_df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def feature_names(preprocessor):
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + BOOLEAN_FEATURES + cat_names


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


class FraudMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # returns raw logits


def train_deep_learning(X_train, y_train, input_dim, epochs=25, batch_size=256):
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # carve out a validation slice (last 10% of train, still time-respecting)
    val_cut = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_cut], X_train[val_cut:]
    y_tr, y_val = y_train[:val_cut], y_train[val_cut:]

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    model = FraudMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -1.0
    best_state = None
    n = len(X_tr_t)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = (
            roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"  [DL] epoch {epoch+1}/{epochs} "
                f"loss={total_loss/n:.4f} val_auc={val_auc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, device


def predict_proba_dl(model, device, X):
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "threshold": threshold,
    }
    print(
        f"{name:>15} | acc={metrics['accuracy']:.4f} "
        f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f} "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )
    return metrics


def curve_points(y_true, y_prob, kind, max_points=60):
    if kind == "roc":
        x, y, _ = roc_curve(y_true, y_prob)
        keys = ("fpr", "tpr")
    else:
        y, x, _ = precision_recall_curve(y_true, y_prob)  # returns precision, recall
        keys = ("recall", "precision")
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x, y = x[idx], y[idx]
    return [{keys[0]: float(a), keys[1]: float(b)} for a, b in zip(x, y)]


def main():
    train_df, test_df = load_and_split()

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(
        train_df[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES]
    )
    X_test = preprocessor.transform(
        test_df[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES]
    )
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    y_train, y_test = train_df[TARGET].values, test_df[TARGET].values
    names = feature_names(preprocessor)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(names, f)

    results = {}
    roc_curves = {}
    pr_curves = {}
    feature_importance = {}

    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = evaluate("Random Forest", y_test, rf_prob)
    roc_curves["random_forest"] = curve_points(y_test, rf_prob, "roc")
    pr_curves["random_forest"] = curve_points(y_test, rf_prob, "pr")
    feature_importance["random_forest"] = sorted(
        [{"feature": n, "importance": float(v)} for n, v in zip(names, rf.feature_importances_)],
        key=lambda d: -d["importance"],
    )[:15]
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    results["xgboost"] = evaluate("XGBoost", y_test, xgb_prob)
    roc_curves["xgboost"] = curve_points(y_test, xgb_prob, "roc")
    pr_curves["xgboost"] = curve_points(y_test, xgb_prob, "pr")
    feature_importance["xgboost"] = sorted(
        [{"feature": n, "importance": float(v)} for n, v in zip(names, xgb_model.feature_importances_)],
        key=lambda d: -d["importance"],
    )[:15]
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")

    print("\nTraining Deep Learning (MLP)...")
    dl_model, device = train_deep_learning(X_train, y_train, input_dim=X_train.shape[1])
    dl_prob = predict_proba_dl(dl_model, device, X_test)
    results["deep_learning"] = evaluate("Deep Learning", y_test, dl_prob)
    roc_curves["deep_learning"] = curve_points(y_test, dl_prob, "roc")
    pr_curves["deep_learning"] = curve_points(y_test, dl_prob, "pr")
    torch.save(dl_model.state_dict(), MODELS_DIR / "deep_learning.pt")
    with open(MODELS_DIR / "deep_learning_config.json", "w") as f:
        json.dump({"input_dim": X_train.shape[1]}, f)

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "trained_at": pd.Timestamp.utcnow().isoformat(),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "fraud_rate_train": float(y_train.mean()),
                "fraud_rate_test": float(y_test.mean()),
                "models": results,
            },
            f,
            indent=2,
        )
    with open(REPORTS_DIR / "roc_curves.json", "w") as f:
        json.dump(roc_curves, f)
    with open(REPORTS_DIR / "pr_curves.json", "w") as f:
        json.dump(pr_curves, f)
    with open(REPORTS_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f)

    print("\nSaved models to models/ and reports to reports/")


if __name__ == "__main__":
    main()
