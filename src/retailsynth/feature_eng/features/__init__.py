from enum import Enum

from .features_avg import AveragePurchaseFrequency, AveragePurchaseQuantity
from .features_base import BaseFeature, IdentityFeature
from .features_cumulative import (
    CumulativeMoneySpent,
    CumulativePurchaseCount,
    CumulativePurchaseQuantity,
)
from .features_lag import LagDiscountPortion, LagMoneySpent, LagPurchaseQuantity
from .features_last import LastMoneySpent, LastPurchaseQuantity
from .features_time import TimeSinceLastPurchase

# mypy: ignore-errors

__all__ = [
    BaseFeature,
    IdentityFeature,
    LastPurchaseQuantity,
    LastMoneySpent,
    CumulativePurchaseCount,
    CumulativePurchaseQuantity,
    CumulativeMoneySpent,
    AveragePurchaseQuantity,
    AveragePurchaseFrequency,
    LagPurchaseQuantity,
    LagMoneySpent,
    LagDiscountPortion,
    TimeSinceLastPurchase,
]


class FeatureCatalog(Enum):
    """record a list of all supported features."""

    item_qty = IdentityFeature
    unit_price = IdentityFeature
    discount_portion = IdentityFeature
    sales_amt = IdentityFeature
    last_purchase_quantity = LastPurchaseQuantity
    last_money_spent = LastMoneySpent
    cumulative_purchase_quantity = CumulativePurchaseQuantity
    cumulative_money_spent = CumulativeMoneySpent
    cumulative_purchase_count = CumulativePurchaseCount
    time_since_last_purchase = TimeSinceLastPurchase
    lag_money_spent = LagMoneySpent
    lag_purchase_quantity = LagPurchaseQuantity
    lag_discount_portion = LagDiscountPortion
    avg_purchase_frequency = AveragePurchaseFrequency
    avg_purchase_quantity = AveragePurchaseQuantity
