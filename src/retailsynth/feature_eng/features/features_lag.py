from dataclasses import dataclass, field

import numpy as np

from .features_base import BaseFeature

# mypy: ignore-errors


@dataclass
class LagPurchaseQuantity(BaseFeature):
    """Collect the purchase quantity of each customer from the last time step."""

    name: str = field(default="lag_purchase_quantity")
    target_column: str = field(default="item_qty")
    lag_time: int = field(default=1)  # number of time steps to retrospect

    def _initialize_online_features(self):
        self.online_feature_array = self._historical_data[-self.lag_time :]

    def get_historical_feature(
        self, customer_index: np.array = None, item_index: np.array = None
    ) -> np.ndarray:
        """Return the cumulative purchase count of each customer since the first purchase event."""
        current_trx_array = self._get_subset_historical_feature(
            customer_index, item_index
        )
        # roll the data by lag time
        feature_array = np.roll(current_trx_array, self.lag_time, axis=0)
        feature_array[: self.lag_time] = 0
        return feature_array

    def update_online_feature(self):
        """Update the online feature array by taking one if purchased."""
        # put the latest the online data at the first place
        # roll the feature array backward by 1 time step
        # so that the feature to output next is always tracked in the first position of the online feature array
        self.online_feature_array[0] = self.online_data
        feature_array = np.roll(self.online_feature_array, -1, axis=0)
        self.online_feature_array = feature_array

    def get_online_feature(self) -> np.ndarray:
        """Return the online lag feature.

        Returns
        -------
            np.ndarray: online lag feature array in a shape of (n_customer, n_item)
        """
        assert isinstance(self.online_feature_array[0], np.ndarray)
        return self.online_feature_array[0]


@dataclass
class LagMoneySpent(LagPurchaseQuantity):
    """Collect the money spent of each customer from the last time step."""

    name: str = field(default="lag_money_spent")
    target_column: str = field(default="sales_amt")


@dataclass
class LagDiscountPortion(LagPurchaseQuantity):
    """Collect the discount portion of each customer from the last time step."""

    name: str = field(default="lag_discount_portion")
    target_column: str = field(default="discount_portion")
