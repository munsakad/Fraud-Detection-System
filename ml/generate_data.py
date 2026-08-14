"""
Generates a synthetic-but-behaviorally-realistic banking transaction dataset for
fraud detection. There's no network access to pull a real dataset (e.g. Kaggle's
credit card fraud set) in this environment, so this builds one from scratch using
domain-realistic generative rules instead of pure noise:

- Each account has its own spending baseline (amount distribution, preferred
  merchant categories, typical hours of activity).
- Fraud is injected as deviations from an account's OWN baseline (unusually large
  amount relative to their history, unfamiliar merchant category, foreign/
  card-not-present, odd hour) plus short bursts of rapid-fire transactions
  ("card testing"), not just globally-rare rows. That's what makes the behavioral
  features (rolling averages, velocity counts) carry real signal instead of being
  decorative.
- All behavioral/velocity features are computed causally (only using transactions
  at or before the current one), which matters: a model trained on features that
  peek into the future looks great on paper and is useless in production.

Run: python generate_data.py
Output: data/transactions.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_ACCOUNTS = 1500
DAYS = 90
START = pd.Timestamp("2026-01-01")

CATEGORIES = {
    # name: (lognormal mean(log), lognormal sigma(log), typical_channel, fraud_affinity)
    "groceries":      (3.6, 0.45, "card_present", 0.2),
    "restaurant":     (3.3, 0.55, "card_present", 0.2),
    "gas_station":    (3.5, 0.35, "card_present", 0.2),
    "utilities":      (4.3, 0.25, "online", 0.1),
    "online_retail":  (4.0, 0.75, "online", 0.7),
    "electronics":    (5.1, 0.85, "online", 1.4),
    "travel":         (5.6, 0.9, "online", 1.3),
    "entertainment":  (3.2, 0.5, "card_present", 0.3),
    "atm_withdrawal": (4.6, 0.5, "card_present", 1.1),
    "jewelry":        (6.0, 0.9, "card_present", 1.6),
    "healthcare":     (4.9, 0.55, "card_present", 0.2),
    "wire_transfer":  (5.2, 1.0, "online", 1.5),
}
CATEGORY_NAMES = list(CATEGORIES.keys())

HOUR_WEIGHTS = np.array(
    [0.5, 0.3, 0.2, 0.15, 0.15, 0.3, 0.8, 1.5, 2.2, 2.6, 2.8, 3.0,
     3.2, 3.0, 2.8, 2.6, 2.6, 2.8, 3.0, 2.6, 2.0, 1.6, 1.1, 0.7]
)
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

FRAUD_HOUR_WEIGHTS = np.array(
    [3.0, 3.2, 3.4, 3.4, 3.0, 2.2, 1.2, 0.8, 0.6, 0.6, 0.6, 0.6,
     0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6]
)
FRAUD_HOUR_WEIGHTS = FRAUD_HOUR_WEIGHTS / FRAUD_HOUR_WEIGHTS.sum()


def make_accounts(n):
    segment = RNG.choice(["light", "medium", "heavy"], size=n, p=[0.4, 0.4, 0.2])
    monthly_txns = np.select(
        [segment == "light", segment == "medium", segment == "heavy"],
        [RNG.integers(6, 15, n), RNG.integers(15, 35, n), RNG.integers(35, 70, n)],
    )
    spend_multiplier = RNG.lognormal(mean=0.0, sigma=0.4, size=n)
    account_age_days = RNG.integers(30, 4000, n)
    n_pref = RNG.integers(2, 5, n)
    preferred = [
        RNG.choice(CATEGORY_NAMES, size=k, replace=False).tolist() for k in n_pref
    ]
    return pd.DataFrame(
        {
            "account_id": [f"ACC{i:06d}" for i in range(n)],
            "monthly_txns": monthly_txns,
            "spend_multiplier": spend_multiplier,
            "account_age_days": account_age_days,
            "preferred_categories": preferred,
        }
    )


def sample_hour(weights):
    return int(RNG.choice(24, p=weights))


def gen_legit_transactions(accounts):
    rows = []
    for _, acc in accounts.iterrows():
        n_txns = RNG.poisson(acc["monthly_txns"] * DAYS / 30.0)
        n_txns = max(n_txns, 1)
        days_offset = RNG.uniform(0, DAYS, n_txns)
        for d in days_offset:
            if RNG.random() < 0.85 and acc["preferred_categories"]:
                category = RNG.choice(acc["preferred_categories"])
            else:
                category = RNG.choice(CATEGORY_NAMES)
            mu, sigma, channel, _ = CATEGORIES[category]
            amount = float(RNG.lognormal(mu, sigma) * acc["spend_multiplier"])
            amount = round(min(amount, 8000.0), 2)
            hour = sample_hour(HOUR_WEIGHTS)
            minute = int(RNG.integers(0, 60))
            second = int(RNG.integers(0, 60))
            ts = START + pd.Timedelta(days=d) + pd.Timedelta(
                hours=hour, minutes=minute, seconds=second
            )
            is_foreign = RNG.random() < 0.02
            channel_noise = channel if RNG.random() < 0.8 else RNG.choice(
                ["card_present", "online", "mobile"]
            )
            rows.append(
                {
                    "account_id": acc["account_id"],
                    "timestamp": ts,
                    "amount": amount,
                    "category": category,
                    "channel": channel_noise,
                    "is_foreign": is_foreign,
                    "account_age_days": acc["account_age_days"],
                    "is_fraud": 0,
                }
            )
    return rows


def gen_fraud_transactions(accounts, legit_df):
    rows = []
    fraud_account_ids = RNG.choice(
        accounts["account_id"], size=max(int(N_ACCOUNTS * 0.11), 20), replace=False
    )
    acc_lookup = accounts.set_index("account_id")

    for acc_id in fraud_account_ids:
        acc = acc_lookup.loc[acc_id]
        hist = legit_df[legit_df["account_id"] == acc_id]
        baseline_amount = hist["amount"].mean() if len(hist) else 50.0
        if np.isnan(baseline_amount):
            baseline_amount = 50.0

        n_episodes = int(RNG.choice([1, 2], p=[0.7, 0.3]))
        episode_days = sorted(RNG.uniform(10, DAYS - 1, n_episodes))

        for episode_day in episode_days:
            episode_start = START + pd.Timedelta(days=episode_day)
            pattern = RNG.choice(["burst", "single_high_value"], p=[0.55, 0.45])

            if pattern == "burst":
                n_fraud = int(RNG.integers(2, 6))
                for k in range(n_fraud):
                    category = RNG.choice(
                        CATEGORY_NAMES,
                        p=_fraud_affinity_probs(),
                    )
                    multiplier = RNG.uniform(2.5, 9.0)
                    amount = round(min(baseline_amount * multiplier, 9500.0), 2)
                    ts = episode_start + pd.Timedelta(minutes=float(RNG.uniform(0, 25 * (k + 1))))
                    rows.append(
                        _fraud_row(acc_id, ts, amount, category, acc["account_age_days"])
                    )
            else:
                category = RNG.choice(CATEGORY_NAMES, p=_fraud_affinity_probs())
                multiplier = RNG.uniform(4.0, 12.0)
                amount = round(min(baseline_amount * multiplier, 9800.0), 2)
                hour = sample_hour(FRAUD_HOUR_WEIGHTS)
                ts = episode_start.normalize() + pd.Timedelta(
                    hours=hour, minutes=int(RNG.integers(0, 60))
                )
                rows.append(
                    _fraud_row(acc_id, ts, amount, category, acc["account_age_days"])
                )
    return rows


def _fraud_affinity_probs():
    affinities = np.array([CATEGORIES[c][3] for c in CATEGORY_NAMES])
    return affinities / affinities.sum()


def _fraud_row(acc_id, ts, amount, category, account_age_days):
    return {
        "account_id": acc_id,
        "timestamp": ts,
        "amount": amount,
        "category": category,
        "channel": RNG.choice(["online", "card_not_present", "mobile"], p=[0.55, 0.3, 0.15]),
        "is_foreign": bool(RNG.random() < 0.4),
        "account_age_days": account_age_days,
        "is_fraud": 1,
    }


def add_behavioral_features(df):
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    df["txn_id"] = [f"TX{100000 + i}" for i in range(len(df))]

    df = df.set_index("timestamp")
    grouped = df.groupby("account_id")

    # causal, trailing-window features: 30D behavioral baseline excludes the
    # current row (closed="left") so a fraud transaction can't inflate its own
    # baseline; velocity counts include the current row (closed="right"),
    # which mirrors what a real-time scoring system would know at auth time.
    df["avg_amount_30d"] = grouped["amount"].apply(
        lambda s: s.rolling("30D", closed="left").mean()
    ).reset_index(level=0, drop=True)
    df["txn_count_1h"] = grouped["amount"].apply(
        lambda s: s.rolling("1h", closed="right").count()
    ).reset_index(level=0, drop=True)
    df["txn_count_24h"] = grouped["amount"].apply(
        lambda s: s.rolling("24h", closed="right").count()
    ).reset_index(level=0, drop=True)
    df["amount_sum_24h"] = grouped["amount"].apply(
        lambda s: s.rolling("24h", closed="right").sum()
    ).reset_index(level=0, drop=True)

    df = df.reset_index()
    df["avg_amount_30d"] = df["avg_amount_30d"].fillna(df["amount"])
    df["amount_ratio_to_avg"] = df["amount"] / df["avg_amount_30d"].clip(lower=1.0)

    # "seen this category before" - causal, computed per account in time order
    seen = {}
    is_new_category = np.zeros(len(df), dtype=bool)
    for i, row in enumerate(df.itertuples()):
        key = row.account_id
        cats = seen.setdefault(key, set())
        is_new_category[i] = row.category not in cats
        cats.add(row.category)
    df["is_new_category"] = is_new_category

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5])

    return df


def main():
    accounts = make_accounts(N_ACCOUNTS)
    legit_rows = gen_legit_transactions(accounts)
    legit_df = pd.DataFrame(legit_rows)
    fraud_rows = gen_fraud_transactions(accounts, legit_df)
    fraud_df = pd.DataFrame(fraud_rows)

    full = pd.concat([legit_df, fraud_df], ignore_index=True)
    full = add_behavioral_features(full)

    cols = [
        "txn_id", "account_id", "timestamp", "amount", "category", "channel",
        "is_foreign", "account_age_days", "avg_amount_30d", "amount_ratio_to_avg",
        "txn_count_1h", "txn_count_24h", "amount_sum_24h", "is_new_category",
        "hour", "day_of_week", "is_weekend", "is_night", "is_fraud",
    ]
    full = full[cols].sort_values("timestamp").reset_index(drop=True)

    full.to_csv("data/transactions.csv", index=False)

    n = len(full)
    n_fraud = int(full["is_fraud"].sum())
    print(f"Generated {n:,} transactions, {n_fraud:,} fraudulent ({n_fraud/n:.3%})")
    print(full.groupby("is_fraud")[["amount", "txn_count_1h", "amount_ratio_to_avg"]].mean())


if __name__ == "__main__":
    main()
