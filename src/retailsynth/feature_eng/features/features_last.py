from dataclasses import dataclass, field

import numpy as np

from .features_base import BaseFeature

# mypy: ignore-errors


@dataclass
class LastPurchaseQuantity(BaseFeature):
    """Compute the quantity in the last purchase event.

    Attributes
    ----------
        name (str): name of the feature. Default to be "last_purchase_quantity"
    """

    name: str = field(default="last_purchase_quantity")
    target_column: str = field(default="item_qty")

    def _initialize_online_features(self):
        # extract the time index of last valid transaction record
        last_valid_time_index = (
            (self._historical_data > 0).cumsum(axis=0).argmax(axis=0)
        )
        time_index_flatten = last_valid_time_index.flatten()
        # prepare the index tuple to locate the last valid quantity
        index = np.zeros((len(time_index_flatten), 3), dtype=int)
        index[:, 0] = time_index_flatten  # time index
        index[:, 1:] = np.array(
            list(np.ndindex(last_valid_time_index.shape))
        )  # customer_index, item_index
        # extract the last valid quantity and reshape it to the right format
        self.online_feature_array = self._historical_data[
            index[:, 0], index[:, 1], index[:, 2]
        ].reshape(last_valid_time_index.shape)

    def get_historical_feature(
        self, customer_index: np.array = None, item_index: np.array = None
    ) -> np.ndarray:
        """Return the last purchase quantity of each customer for each item.

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
        # initialize a nan container for historical feature with indicated shape
        historical_features = np.zeros(current_trx_array.shape)
        historical_features[:] = np.nan
        # take value from historical data if purchased, otherwise keep nan
        # roll back 1 unit along the time axis
        historical_features = np.where(
            np.roll(current_trx_array > 0, 1, axis=0),  # checking if purchased
            np.roll(current_trx_array, 1, axis=0),
            historical_features,
        )
        # set the first time step of historical feature to be nan
        historical_features[0] = np.nan
        # fill the nan with the last valid value
        idx = np.isnan(historical_features)
        order_array = np.ones(historical_features.shape) * np.arange(
            historical_features.shape[0]
        ).reshape(-1, 1, 1)
        idx = np.where(~idx, order_array, 0)
        idx = np.maximum.accumulate(idx, axis=0).astype(int)
        historical_features = np.take_along_axis(historical_features, idx, axis=0)
        historical_features = np.nan_to_num(historical_features, copy=False, nan=0)
        return historical_features

    def update_online_feature(self):
        """Update the online feature array by taking the online data if purchased."""
        # get flag of purchase event
        purchase_event = self.online_data > 0
        # take value from online data if purchased, otherwise keep the original value
        online_feature = np.where(
            purchase_event,
            self.online_data,
            self.online_feature_array,
        )
        # assign the online feature array the reference to online feature
        self.online_feature_array = online_feature
        self.online_feature_array[self.online_feature_array == None] = 0  # noqa: E711


@dataclass
class LastMoneySpent(LastPurchaseQuantity):
    """Collect the money spent in last purchase event."""

    # Name the feature according to intended usage
    # assuming that the user passes in purchase amount as
    # the transaction array

    name: str = field(default="last_money_spent")
    target_column: str = field(default="sales_amt")
