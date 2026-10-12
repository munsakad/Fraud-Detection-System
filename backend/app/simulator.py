import asyncio
import random
import uuid
from collections import deque
from datetime import datetime, timezone

import pandas as pd
from fastapi import WebSocket

from . import config
from .inference import ALL_FEATURES, get_bundle

_buffer: deque = deque(maxlen=config.LIVE_BUFFER_SIZE)
_connections: set[WebSocket] = set()
_pool_df: pd.DataFrame | None = None
_stats = {"total": 0, "flagged": {"low": 0, "medium": 0, "high": 0, "critical": 0}}


def _load_pool() -> pd.DataFrame:
    """The held-out (last 20%, time-based) slice of transactions.csv - the same
    rows the models were evaluated on - replayed here as a stand-in for a live
    transaction feed. Ground-truth labels are kept for the dashboard's "was this
    actually fraud?" comparison but are never given to the models as input."""
    global _pool_df
    if _pool_df is None:
        path = config.DATA_DIR / "transactions.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python generate_data.py` inside ml/ first."
            )
        df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
        for col in ["is_foreign", "is_new_category", "is_weekend", "is_night"]:
            df[col] = df[col].astype(bool)
        cutoff = int(len(df) * 0.8)
        _pool_df = df.iloc[cutoff:].reset_index(drop=True)
    return _pool_df


def _score_row(row: pd.Series) -> dict:
    bundle = get_bundle()
    feature_row = {k: row[k] for k in ALL_FEATURES}
    scores = bundle.score(feature_row)
    return {
        "event_id": str(uuid.uuid4()),
        "txn_id": row["txn_id"],
        "account_id": row["account_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "amount": float(row["amount"]),
        "category": row["category"],
        "channel": row["channel"],
        "is_foreign": bool(row["is_foreign"]),
        "ground_truth_is_fraud": bool(row["is_fraud"]),
        "scores": scores,
    }


async def broadcast(event: dict):
    dead = []
    for ws in _connections:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connections.discard(ws)


def register(ws: WebSocket):
    _connections.add(ws)


def unregister(ws: WebSocket):
    _connections.discard(ws)


def recent_events(limit: int = 50) -> list[dict]:
    return list(_buffer)[-limit:][::-1]


def get_stats() -> dict:
    total = _stats["total"]
    # "low" is tracked for the breakdown but is not a flag
    flagged_total = sum(v for k, v in _stats["flagged"].items() if k != "low")
    return {
        "total_scored": total,
        "flagged_counts": dict(_stats["flagged"]),
        "flagged_rate": (flagged_total / total) if total else 0.0,
    }


async def run_simulator():
    pool = _load_pool()
    n = len(pool)
    while True:
        idx = random.randrange(n)
        row = pool.iloc[idx]
        try:
            event = _score_row(row)
        except Exception as e:
            # keep the simulator alive even if a single row fails to score
            await asyncio.sleep(config.SIMULATOR_INTERVAL_SECONDS)
            continue
        _buffer.append(event)
        _stats["total"] += 1
        _stats["flagged"][event["scores"]["risk_level"]] += 1
        await broadcast(event)
        await asyncio.sleep(config.SIMULATOR_INTERVAL_SECONDS)
