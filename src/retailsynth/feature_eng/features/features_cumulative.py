from dataclasses import dataclass, field

import numpy as np

from .features_base import BaseFeature

# mypy: ignore-errors


@dataclass
class CumulativePurchaseQuantity(BaseFeature):
    """Collect the cumulative purchase quantity since a customer's first purchase event."""

    name: str = field(default="cumulative_purchase_quantity")
    target_column: str = field(default="item_qty")

    def _initialize_online_features(self):
        self.online_feature_array = np.nansum(self._historical_data, axis=0)

    def get_historical_feature(
        self, customer_index: np.array = None, item_index: np.array = None
    ) -> np.ndarray:
        """Return the cumulative purchase quantity of each customer for each item.

        Parameters
        ----------
            customer_index (np.array): a subset of customers
            item_index (np.array): a subset of items (category, subcategory, product_nbr)

        Returns
        -------
            np.ndarray: historical feature array in a shape of (n_week, n_customer, n_item)
        """
        current_trx_array = self._get_subset_historical_feature(
            customer_index, item_index
        )
        # compute cumsum
        feature_array = np.nancumsum(current_trx_array, axis=0)
        # roll the feature array along the time axis
        feature_array = np.roll(feature_array, 1, axis=0)
        # fill the first time step with 0
        feature_array[0] = 0
        return feature_array

    def update_online_feature(self):
        """Update the online feature array by adding the online data if purchased."""
        # get flag of purchase event
        purchase_event = self.online_data > 0

        self.online_feature_array[self.online_feature_array == None] = 0  # noqa: E711
        # take value from online data if purchased, otherwise keep the original value
        online_feature = np.where(
            purchase_event,
            self.online_data + self.online_feature_array,
            self.online_feature_array,
        )
        # assign the online feature array the reference to online feature
        self.online_feature_array = online_feature


@dataclass
class CumulativeMoneySpent(CumulativePurchaseQuantity):
    """Collect the cumulative money spent since a customer's first purchase event."""

    name: str = field(default="cumulative_money_spent")
    target_column: str = field(default="sales_amt")


@dataclass
class CumulativePurchaseCount(BaseFeature):
    """Collect the cumulative purchase count since a customer's first purchase event."""

    name: str = field(default="cumulative_purchase_count")
    target_column: str = field(default="item_qty")

    def _initialize_online_features(self):
        self.online_feature_array = np.nansum(self._historical_data > 0, axis=0)

    def get_historical_feature(
        self, customer_index: np.array = None, item_index: np.array = None
    ) -> np.ndarray:
        """Return the cumulative purchase count of each customer since the first purchase event.

        Parameters
        ----------
            customer_index (np.array): a subset of customers
            item_index (np.array): a subset of items (category, subcategory, product_nbr)

        Returns
        -------
            np.ndarray: historical feature array in a shape of (n_week, n_customer, n_item)
        """
        current_trx_array = self._get_subset_historical_feature(
            customer_index, item_index
        )
        # compute cumsum
        feature_array = np.nancumsum(current_trx_array > 0, axis=0)
        # roll the feature array along the time axis
        feature_array = np.roll(feature_array, 1, axis=0)
        # fill the first time step with 0
        feature_array[0] = 0
        return feature_array

    def update_online_feature(self):
        """Update the online feature array by taking one if purchased."""
        # get flag of purchase event
        purchase_event = self.online_data > 0
        # take value from online data if purchased, otherwise keep the original value
        online_feature = np.where(
            purchase_event,
            self.online_feature_array + 1,
            self.online_feature_array,
        )
        # assign the online feature array the reference to online feature
        self.online_feature_array = online_feature
