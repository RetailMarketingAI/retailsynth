from dataclasses import dataclass, field

import numpy as np

from .features_base import BaseFeature

# mypy: ignore-errors


@dataclass
class TimeSinceLastPurchase(BaseFeature):
    """Collect the time since last purchase event."""

    name: str = field(default="time_since_last_purchase")
    target_column: str = field(default="item_qty")

    def _initialize_online_features(self):
        last_valid_time_index = (
            (self._historical_data > 0).cumsum(axis=0).argmax(axis=0)
        )
        self.online_feature_array = (
            self._historical_data.shape[0] - last_valid_time_index
        )

    def get_historical_feature(
        self, customer_index: np.array = None, item_index: np.array = None
    ) -> np.ndarray:
        """
        return the time since last purchase of each customer for each item.

        Parameters
        ----------
            customer_index (np.array): a subset of customers
            item_index (np.array): a subset of items (category, subcategory, product_nbr)

        Returns
        -------
            np.ndarray: historical feature array in a shape of (n_week, n_customer, n_item)

        """
        # create the flag if customer purchases the product at the current time step
        current_trx_array = self._get_subset_historical_feature(
            customer_index, item_index
        )
        if_purchase = current_trx_array > 0
        # create a series with weeks rank from 0 to the end
        order_array = np.ones(if_purchase.shape) * np.arange(
            if_purchase.shape[0]
        ).reshape(-1, 1, 1)
        # find the purchase event and restart the week series from 0 again
        idx = np.where(if_purchase, order_array, 0)
        idx = np.maximum.accumulate(idx, axis=0).astype(int)
        feature = np.roll(order_array - idx + 1, shift=1, axis=0)
        zero = np.argmax(if_purchase, axis=0)
        index = np.indices(zero.shape).transpose(1, 2, 0)
        index = np.concatenate((np.expand_dims(zero, axis=-1), index), axis=-1)
        mask = np.broadcast_to(
            np.arange(feature.shape[0])[:, None, None] <= index[:, :, 0], feature.shape
        )
        feature[mask] = np.nan
        return np.nan_to_num(feature, copy=False, nan=0)

    def update_online_feature(self):
        """Update the online feature array by computing the time difference since last purchase event."""
        # get flag of purchase event
        purchase_event = self.online_data > 0
        # change feature to 1 if purchased, otherwise adding 1 time step from the previous feature value
        online_feature = np.where(
            purchase_event,
            1,
            self.online_feature_array + 1,
        )
        # assign the online feature array the reference to online feature
        self.online_feature_array = online_feature

    def get_online_feature(self) -> np.ndarray:
        """Overwrite the getter function to fillna value in the online features with 0."""
        online_feature = np.nan_to_num(self.online_feature_array)
        assert isinstance(online_feature, np.ndarray)
        return online_feature
