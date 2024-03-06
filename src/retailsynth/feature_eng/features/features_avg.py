from dataclasses import dataclass, field

import numpy as np

from .features_base import BaseFeature


@dataclass
class AveragePurchaseQuantity(BaseFeature):
    """Collect the average purchase quantity of a customer's purchase events."""

    name: str = field(default="avg_purchase_quantity")
    target_column: str = field(default="item_qty")

    def get_historical_feature(
        self, customer_index: np.array = None, item_index: np.array = None
    ) -> np.ndarray:
        """Return the cumulative average quantity of each customer for each item, filling missing values with 1s.

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
        purchase_mask = current_trx_array > 0

        # Calculate average purchase quantity for each customer and item
        cumsum_items = np.nancumsum(current_trx_array, axis=0)
        cumsum_events = np.nancumsum(purchase_mask, axis=0)
        avg_purchase_quantity = cumsum_items / (1e-9 + cumsum_events)

        # roll the feature array along the time axis
        avg_purchase_quantity = np.roll(avg_purchase_quantity, 1, axis=0)
        # fill the first time step with 0
        avg_purchase_quantity[0] = 0
        return avg_purchase_quantity

    def _initialize_online_features(self):
        self.purchase_counts = np.nansum(self._historical_data > 0, axis=0)
        self.online_feature_array = (
            np.nansum(self._historical_data, axis=0) / self.purchase_counts
        )

    def update_online_feature(self):
        """Update the online feature array by adding the online data if purchased."""
        # get flag of purchase event
        purchase_event = self.online_data > 0
        self.purchase_counts[purchase_event] = self.purchase_counts[purchase_event] + 1

        self.online_feature_array[purchase_event] = (
            self.purchase_counts[purchase_event] - 1
        ) * self.online_feature_array[purchase_event] + self.online_data[purchase_event]
        self.online_feature_array[purchase_event] = (
            self.online_feature_array[purchase_event]
            / self.purchase_counts[purchase_event]
        )


@dataclass
class AveragePurchaseFrequency(BaseFeature):
    """Collect the average purchase frequency since a customer's first purchase event."""

    name: str = field(default="avg_purchase_frequency")
    target_column: str = field(default="item_qty")

    def get_historical_feature(
        self, customer_index: np.array = None, item_index: np.array = None
    ) -> np.ndarray:
        """Return the avg. purchase freq. of each customer since the first purchase event.

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

        cumulative_time_period = np.arange(1, feature_array.shape[0] + 1).reshape(
            -1, 1, 1
        )
        feature_array = feature_array / cumulative_time_period
        # roll the feature array along the time axis
        feature_array = np.roll(feature_array, 1, axis=0)
        # fill the first time step with 0
        feature_array[0] = 0
        return feature_array

    def _initialize_online_features(self):
        self.online_feature_array = (
            np.nansum(self._historical_data > 0, axis=0) / self.n_historical_weeks
        )

    def update_online_feature(self):
        """Update the online feature array by taking one if purchased."""
        # get flag of purchase event
        purchase_event = self.online_data > 0

        cumulative_purchase_counts = (
            self.n_historical_weeks - 1
        ) * self.online_feature_array + purchase_event

        # assign the online feature array the reference to online feature
        self.online_feature_array = cumulative_purchase_counts / self.n_historical_weeks
