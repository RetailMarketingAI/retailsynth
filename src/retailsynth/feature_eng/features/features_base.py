"""Features defined for RetMar."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# mypy: ignore-errors


@dataclass
class BaseFeature(ABC):
    """Abstract class of feature object.

        Expected usage includes:
        1. Write historical data and initialize the online feature array from the historical datasets
            >>> feature.set_historical_data(trx_array)
        2. Get historical data.
            >>> feature.get_historical_feature(customer_index, item_index)
        2. Update the online feature array by adding the online data
            >>> feature.write_online_data(online_trx_array)
            >>> feature.update_online_feature()
        3. Get online features
            >>> feature.get_online_feature()


    Attributes
    ----------
        aggregation_level (str): product hierarchy level of the feature
        time_freq (str): time frequency of the feature, either week or day
    """

    aggregation_level: str = field(default="product_nbr")
    time_freq: str = field(default="week")

    def __post_init__(self):
        """Set default values for attributes that helps generate online features."""
        self.online_feature_array = None
        self.n_historical_weeks = 0

    @abstractmethod
    def _initialize_online_features(self):
        """Abstract method to initialize the online features from the historical datasets.

        The online feature should be fully processed. No call to update_online_feature()
        is to be needed after initialization.
        """
        raise NotImplementedError

    def set_historical_data(self, transaction_array: np.ndarray):
        """Expect the input trx_array in a shape of (n_week, n_customer, n_item) and store it as an attribute in the feature object.

        Parameters
        ----------
            transaction_array (np.ndarray): multi-dimensional array of transaction data
                in a shape of (n_week, n_customer, n_item)
        """
        assert len(transaction_array.shape) == 3
        self._historical_data = transaction_array

        # Initialize the online features using historical data
        self.online_data = transaction_array[-1]
        self.n_historical_weeks = transaction_array.shape[0]
        self._initialize_online_features()

    @abstractmethod
    def get_historical_feature(
        self, customer_index: np.array = None, item_index: np.array = None
    ) -> np.ndarray:
        """Return historical data attribute as it is.

        Parameters
        ----------
            customer_index (np.array): a subset of customers
            item_index (np.array): a subset of items (category, subcategory, product_nbr)

        Returns
        -------
            np.ndarray: historical data attribute in a shape of (n_week, n_customer, n_item)

        """
        pass

    def _get_subset_historical_feature(
        self, customer_index: np.array, item_index: np.array
    ):
        """Get subset of historical features for specific customer and item pair.

        Parameters
        ----------
            customer_index (np.array): a subset of customers
            item_index (np.array): a subset of items (category, subcategory, product_nbr)

        Returns
        -------
            np.ndarray: subset of historical features
        """
        current_trx_array = (
            self._historical_data[:, customer_index]
            if customer_index is not None
            else self._historical_data
        )
        current_trx_array = (
            current_trx_array[:, :, item_index]
            if item_index is not None
            else current_trx_array
        )

        return current_trx_array

    def write_online_data(self, transaction_array: np.ndarray):
        """Expect the input trx_array in a shape of (n_customer, n_item).

        Parameters
        ----------
            transaction_array (np.ndarray): update online data attribute by changing the reference
        """
        self.online_data = transaction_array
        self.n_historical_weeks += 1

    @abstractmethod
    def update_online_feature(self):
        """Update online feature array by updating online data attribute."""
        pass

    def get_online_feature(self) -> np.ndarray:
        """Directly return attribute of  online feature array.

        Returns
        -------
            np.ndarray: online feature array in a shape of (n_customer, n_item)
        """
        if isinstance(self.online_feature_array, np.matrix):
            features = self.online_feature_array.A
        else:
            features = self.online_feature_array
            assert isinstance(self.online_feature_array, np.ndarray)
        features[features == None] = 0  # noqa: E711
        return features


@dataclass
class IdentityFeature(BaseFeature):
    """Helper class to return the original transaction data as feature.

    Attributes
    ----------
        name (str): name of the feature. Default to be "identity"
    """

    name: str = field(default="identity")

    def __post_init__(self):
        """Assign target_column identical to feature name."""
        self.target_column = self.name

    def _initialize_online_features(self):
        """Initialize online feature array from historical data."""
        self.online_feature_array = self._historical_data[-1]

    def get_historical_feature(
        self, customer_index: np.array = None, item_index: np.array = None
    ) -> np.ndarray:
        """Return historical data attribute as it is.

        Parameters
        ----------
            customer_index (np.array): a subset of customers
            item_index (np.array): a subset of items (category, subcategory, product_nbr)

        Returns
        -------
            np.ndarray: historical data attribute in a shape of (n_week, n_customer, n_item)
        """
        current_trx_array = self._get_subset_historical_feature(
            customer_index, item_index
        )
        return current_trx_array

    def write_online_data(self, transaction_array: np.ndarray):
        """Expect the input trx_array in a shape of (n_customer, n_item).

        Parameters
        ----------
            transaction_array (np.ndarray): update online data attribute by changing the reference
        """
        self.online_data = transaction_array
        self.online_feature_array = transaction_array

    def update_online_feature(self):
        pass
