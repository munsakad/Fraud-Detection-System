import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config

NUMERIC_FEATURES = [
    "amount", "account_age_days", "avg_amount_30d", "amount_ratio_to_avg",
    "txn_count_1h", "txn_count_24h", "amount_sum_24h", "hour", "day_of_week",
]
BOOLEAN_FEATURES = ["is_foreign", "is_new_category", "is_weekend", "is_night"]
CATEGORICAL_FEATURES = ["category", "channel"]
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


class ArtifactsNotFound(Exception):
    pass


class FraudMLP(nn.Module):
    """Must match the architecture in ml/train_models.py exactly, since we're
    loading a state_dict trained with that definition."""

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
        return self.net(x).squeeze(-1)


class ModelBundle:
    def __init__(self):
        required = [
            config.MODELS_DIR / "preprocessor.joblib",
            config.MODELS_DIR / "random_forest.joblib",
            config.MODELS_DIR / "xgboost.joblib",
            config.MODELS_DIR / "deep_learning.pt",
            config.MODELS_DIR / "deep_learning_config.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise ArtifactsNotFound(
                "Missing trained model artifacts: "
                + ", ".join(missing)
                + ". Run `python train_models.py` inside ml/ first."
            )

        self.preprocessor = joblib.load(config.MODELS_DIR / "preprocessor.joblib")
        self.random_forest = joblib.load(config.MODELS_DIR / "random_forest.joblib")
        self.xgboost = joblib.load(config.MODELS_DIR / "xgboost.joblib")

        with open(config.MODELS_DIR / "deep_learning_config.json") as f:
            dl_config = json.load(f)
        self.deep_learning = FraudMLP(dl_config["input_dim"])
        self.deep_learning.load_state_dict(
            torch.load(config.MODELS_DIR / "deep_learning.pt", map_location="cpu")
        )
        self.deep_learning.eval()

    def _transform(self, row: dict) -> np.ndarray:
        df = pd.DataFrame([{k: row.get(k) for k in ALL_FEATURES}])
        return np.asarray(self.preprocessor.transform(df))

    def score(self, row: dict) -> dict:
        """row: a dict with all ALL_FEATURES keys. Returns per-model fraud
        probabilities plus an ensemble score and risk level."""
        X = self._transform(row)

        rf_prob = float(self.random_forest.predict_proba(X)[:, 1][0])
        xgb_prob = float(self.xgboost.predict_proba(X)[:, 1][0])
        with torch.no_grad():
            dl_logit = self.deep_learning(torch.tensor(X, dtype=torch.float32))
            dl_prob = float(torch.sigmoid(dl_logit).item())

        ensemble = (rf_prob + xgb_prob + dl_prob) / 3.0
        return {
            "random_forest": round(rf_prob, 4),
            "xgboost": round(xgb_prob, 4),
            "deep_learning": round(dl_prob, 4),
            "ensemble": round(ensemble, 4),
            "risk_level": risk_level(ensemble),
        }


def risk_level(prob: float) -> str:
    if prob >= config.RISK_THRESHOLDS["critical"]:
        return "critical"
    if prob >= config.RISK_THRESHOLDS["high"]:
        return "high"
    if prob >= config.RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return ModelBundle()
