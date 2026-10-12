"""
Exports the trained models and evaluation reports into a static bundle so the
GitHub Pages demo can run entirely in the browser, with no FastAPI backend.

Run after generate_data.py and train_models.py (from the repo root):

    python ml/export_web.py

Writes frontend/public/demo/:
  reports/*.json  the evaluation reports produced by train_models.py
  model.json      preprocessor params plus all three models in a compact form:
                  BatchNorm folded into the MLP's linear layers, XGBoost and
                  Random Forest trees flattened into arrays
  events.json     held-out transactions scored by the backend's ModelBundle,
                  replayed by the demo's live feed
  parity.json     feature rows with their Python scores; frontend/scripts/
                  check-parity.mjs uses them to verify the JS port in CI
"""

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "ml"
OUT = ROOT / "frontend" / "public" / "demo"

# score with the exact same code path the API uses
sys.path.insert(0, str(ROOT / "backend"))
from app.inference import (  # noqa: E402
    ALL_FEATURES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    get_bundle,
)

N_EVENTS = 800
N_PARITY = 400
random.seed(7)


def export_preprocessor(ct):
    num = ct.named_transformers_["num"]
    cat = ct.named_transformers_["cat"]
    names = [name for name, _, _ in ct.transformers_ if name != "remainder"]
    assert names == ["num", "bool", "cat"], names
    return {
        "numeric": NUMERIC_FEATURES,
        "mean": num.mean_.tolist(),
        "scale": num.scale_.tolist(),
        "boolean": BOOLEAN_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "categories": [[str(c) for c in cats] for cats in cat.categories_],
    }


def export_mlp(model):
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    layers = []
    for lin, bn in (("net.0", "net.1"), ("net.4", "net.5"), ("net.8", None)):
        W, b = sd[f"{lin}.weight"], sd[f"{lin}.bias"]
        if bn:
            # eval-mode BatchNorm is an affine map, so fold it into the Linear layer
            s = sd[f"{bn}.weight"] / np.sqrt(sd[f"{bn}.running_var"] + 1e-5)
            W = W * s[:, None]
            b = (b - sd[f"{bn}.running_mean"]) * s + sd[f"{bn}.bias"]
        layers.append({"W": np.round(W, 7).tolist(), "b": np.round(b, 7).tolist()})
    return layers


def _xgb_leaf(tree, x):
    i = 0
    while tree["f"][i] != -1:
        v = np.float32(x[tree["f"][i]])
        i = tree["y"][i] if v < np.float32(tree["t"][i]) else tree["n"][i]
    return tree["l"][i]


def export_xgb(model, X_sample):
    booster = model.get_booster()
    trees = []
    for dump in booster.get_dump(dump_format="json"):
        nodes = {}

        def walk(node):
            nodes[node["nodeid"]] = node
            for child in node.get("children", []):
                walk(child)

        walk(json.loads(dump))
        size = max(nodes) + 1
        t = {"f": [-1] * size, "t": [0.0] * size, "y": [0] * size, "n": [0] * size, "l": [0.0] * size}
        for i, node in nodes.items():
            if "leaf" in node:
                t["l"][i] = node["leaf"]
            else:
                t["f"][i] = int(str(node["split"]).lstrip("f"))
                t["t"][i] = node["split_condition"]
                t["y"][i] = node["yes"]
                t["n"][i] = node["no"]
        trees.append(t)
    # derive the base margin empirically instead of depending on xgboost's
    # internal config format: margin = base + sum(leaf values)
    margins = booster.predict(xgb.DMatrix(X_sample), output_margin=True)
    leaf_sums = np.array([sum(_xgb_leaf(t, x) for t in trees) for x in X_sample])
    base = float(np.median(margins - leaf_sums))
    spread = float(np.max(np.abs(margins - leaf_sums - base)))
    print(f"  xgboost: {len(trees)} trees, base_margin={base:.6f}, max residual={spread:.2e}")
    return {"trees": trees, "base_margin": base}


def export_rf(rf):
    assert list(rf.classes_) == [0, 1], rf.classes_
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        p = v[:, 1] / np.maximum(v.sum(axis=1), 1e-12)
        leaf = t.children_left == -1
        trees.append({
            "f": np.where(leaf, -1, t.feature).astype(int).tolist(),
            "t": [0 if lf else float(th) for lf, th in zip(leaf, t.threshold)],
            "l": t.children_left.astype(int).tolist(),
            "r": t.children_right.astype(int).tolist(),
            "p": [round(float(pp), 6) if lf else 0 for lf, pp in zip(leaf, p)],
        })
    print(f"  random forest: {len(trees)} trees, {sum(len(t['f']) for t in trees):,} nodes")
    return trees


def json_row(row):
    out = {}
    for k in NUMERIC_FEATURES:
        out[k] = float(row[k])
    for k in BOOLEAN_FEATURES:
        out[k] = bool(row[k])
    for k in CATEGORICAL_FEATURES:
        out[k] = str(row[k])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = get_bundle()

    # same held-out pool the backend simulator replays (last 20%, time-based)
    df = pd.read_csv(ML_DIR / "data" / "transactions.csv", parse_dates=["timestamp"]).sort_values("timestamp")
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(bool)
    pool = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
    sample = pool.sample(n=min(N_EVENTS, len(pool)), random_state=7).reset_index(drop=True)

    events, parity = [], []
    for i, row in sample.iterrows():
        features = json_row(row)
        scores = bundle.score({k: row[k] for k in ALL_FEATURES})
        events.append({
            "txn_id": row["txn_id"],
            "account_id": row["account_id"],
            "amount": float(row["amount"]),
            "category": row["category"],
            "channel": row["channel"],
            "is_foreign": bool(row["is_foreign"]),
            "ground_truth_is_fraud": bool(row["is_fraud"]),
            "scores": scores,
        })
        if i < N_PARITY:
            parity.append({"x": features, "s": scores})

    X_sample = bundle._transform(json_row(sample.iloc[0]))
    for j in range(1, 50):
        X_sample = np.vstack([X_sample, bundle._transform(json_row(sample.iloc[j]))])

    print("Exporting models...")
    model = {
        "preprocessor": export_preprocessor(bundle.preprocessor),
        "mlp": export_mlp(bundle.deep_learning),
        "xgboost": export_xgb(bundle.xgboost, X_sample),
        "random_forest": export_rf(bundle.random_forest),
    }

    (OUT / "reports").mkdir(exist_ok=True)
    for name in ["metrics", "roc_curves", "pr_curves", "feature_importance"]:
        shutil.copy(ML_DIR / "reports" / f"{name}.json", OUT / "reports" / f"{name}.json")

    for name, obj in [("model", model), ("events", events), ("parity", parity)]:
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(obj, separators=(",", ":")))
        print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.2f} MB)")

    flagged = sum(e["scores"]["risk_level"] != "low" for e in events)
    print(f"{len(events)} replay events ({flagged} flagged), {len(parity)} parity rows")


if __name__ == "__main__":
    main()
