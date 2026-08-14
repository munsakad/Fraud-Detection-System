from typing import Optional

from pydantic import BaseModel, Field


class TransactionScoreRequest(BaseModel):
    amount: float = Field(..., gt=0)
    category: str
    channel: str
    account_age_days: int = Field(365, ge=0)
    avg_amount_30d: Optional[float] = None
    amount_ratio_to_avg: Optional[float] = None
    txn_count_1h: int = 1
    txn_count_24h: int = 1
    amount_sum_24h: Optional[float] = None
    is_foreign: bool = False
    is_new_category: bool = False
    hour: int = Field(12, ge=0, le=23)
    day_of_week: int = Field(2, ge=0, le=6)
    is_weekend: bool = False
    is_night: bool = False

    def to_feature_row(self) -> dict:
        avg = self.avg_amount_30d if self.avg_amount_30d is not None else self.amount
        return {
            "amount": self.amount,
            "category": self.category,
            "channel": self.channel,
            "account_age_days": self.account_age_days,
            "avg_amount_30d": avg,
            "amount_ratio_to_avg": self.amount_ratio_to_avg
            if self.amount_ratio_to_avg is not None
            else self.amount / max(avg, 1.0),
            "txn_count_1h": self.txn_count_1h,
            "txn_count_24h": self.txn_count_24h,
            "amount_sum_24h": self.amount_sum_24h
            if self.amount_sum_24h is not None
            else self.amount,
            "is_foreign": self.is_foreign,
            "is_new_category": self.is_new_category,
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "is_night": self.is_night,
        }
