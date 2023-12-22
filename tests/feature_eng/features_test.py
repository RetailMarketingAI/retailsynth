import numpy as np
from numpy.testing import assert_equal

from retailsynth.feature_eng.features import (
    CumulativeMoneySpent,
    CumulativePurchaseCount,
    CumulativePurchaseQuantity,
    IdentityFeature,
    LagMoneySpent,
    LagPurchaseQuantity,
    LastMoneySpent,
    LastPurchaseQuantity,
    TimeSinceLastPurchase,
)

SAMPLE_TRX_ARRAY = np.array(
    [
        # first day of transaction
        [
            [1, 0, 1],  # customer 1, purchase quantity of 3 products
            [0, 1, 0],
        ],  # customer 2, purchase quantity of 3 products
        # second day of transaction
        [
            [0, 0, 2],  # customer 1, purchase quantity of 3 products
            [1, 1, 1],
        ],  # customer 2, purchase quantity of 3 products
        # third day of transaction
        [
            [2, 1, 0],  # customer 1, purchase quantity of 3 products
            [0, 0, 0],
        ],  # customer 2, purchase quantity of 3 products
        # fourth day of transaction
        [
            [0, 0, 1],  # customer 1, purchase quantity of 3 products
            [0, 0, 0],
        ],  # customer 2, purchase quantity of 3 products
    ]
)


class TestIdentityFeature:
    feature_object = IdentityFeature(aggregation_level="product_nbr", time_freq="week")

    @classmethod
    def setup_class(cls):
        cls.feature_object.set_historical_data(SAMPLE_TRX_ARRAY)
        # check if the historical_data attribute is updated
        assert cls.feature_object._historical_data.shape == SAMPLE_TRX_ARRAY.shape

    def test_get_historical_features(self):
        historical_features = self.feature_object.get_historical_feature()
        assert historical_features.shape == self.feature_object._historical_data.shape
        assert_equal(
            historical_features[1],  # test if return the original input at week 2
            np.array([[0, 0, 2], [1, 1, 1]]),
        )

    def test_online_features(self, sample_online_array):
        self.feature_object.write_online_data(sample_online_array)
        self.feature_object.update_online_feature()
        online_features = self.feature_object.get_online_feature()
        assert online_features.shape == self.feature_object.online_feature_array.shape
        # expect the feature object return the original input of online data
        assert_equal(online_features, sample_online_array)


class TestLastPurchaseQuantity:
    feature_object = LastPurchaseQuantity(
        aggregation_level="product_nbr", time_freq="week"
    )

    @classmethod
    def setup_class(cls):
        cls.feature_object.set_historical_data(SAMPLE_TRX_ARRAY)
        # check if the historical_data attribute is updated
        assert cls.feature_object._historical_data.shape == SAMPLE_TRX_ARRAY.shape

    def test_get_historical_features(self):
        historical_features = self.feature_object.get_historical_feature()
        assert historical_features.shape == self.feature_object._historical_data.shape
        assert_equal(
            historical_features[1],  # test the last purchase quantity feature at week 2
            np.array([[1, 0, 1], [0, 1, 0]]),
        )
        assert_equal(
            historical_features[2],  # test the last purchase quantity feature at week 3
            np.array([[1, 0, 2], [1, 1, 1]]),
        )

    def test_write_online_data(self, sample_online_array):
        self.feature_object.write_online_data(sample_online_array)
        # check if the online_data attribute is updated
        assert self.feature_object.online_data.shape == sample_online_array.shape

    def test_update_online_features(self):
        self.feature_object.update_online_feature()
        assert (
            self.feature_object.online_feature_array.shape
            == self.feature_object.online_data.shape
        )
        # check if the online_feature is updated with the new input data
        assert_equal(
            self.feature_object.online_feature_array, np.array([[2, 1, 1], [1, 1, 1]])
        )

    def test_get_online_features(self):
        online_features = self.feature_object.get_online_feature()
        assert online_features.shape == self.feature_object.online_feature_array.shape
        assert_equal(online_features, np.array([[2, 1, 1], [1, 1, 1]]))


class TestLastMoneySpent:
    feature_object = LastMoneySpent(aggregation_level="product_nbr", time_freq="week")
    unit_price = 0.9

    @classmethod
    def setup_class(cls):
        SAMPLE_TRX_ARRAY_revenue = SAMPLE_TRX_ARRAY * cls.unit_price
        cls.feature_object.set_historical_data(SAMPLE_TRX_ARRAY_revenue)
        # check if the historical_data attribute is updated
        assert cls.feature_object._historical_data.shape == SAMPLE_TRX_ARRAY.shape

    def test_get_historical_features(self):
        historical_features = self.feature_object.get_historical_feature()
        assert historical_features.shape == self.feature_object._historical_data.shape
        assert_equal(
            historical_features[1],  # test the last purchase quantity feature at week 2
            np.array([[1, 0, 1], [0, 1, 0]]) * self.unit_price,
        )
        assert_equal(
            historical_features[2],  # test the last purchase quantity feature at week 3
            np.array([[1, 0, 2], [1, 1, 1]]) * self.unit_price,
        )

    def test_write_online_data(self, sample_online_array):
        sample_online_array_revenue = sample_online_array * self.unit_price
        self.feature_object.write_online_data(sample_online_array_revenue)
        # check if the online_data attribute is updated
        assert (
            self.feature_object.online_data.shape == sample_online_array_revenue.shape
        )

    def test_update_online_features(self):
        self.feature_object.update_online_feature()
        assert (
            self.feature_object.online_feature_array.shape
            == self.feature_object.online_data.shape
        )
        # check if the online_feature is updated with the new input data
        assert_equal(
            self.feature_object.online_feature_array,
            np.array([[2, 1, 1], [1, 1, 1]]) * self.unit_price,
        )

    def test_get_online_features(self):
        online_features = self.feature_object.get_online_feature()
        assert online_features.shape == self.feature_object.online_feature_array.shape
        assert_equal(
            online_features, np.array([[2, 1, 1], [1, 1, 1]]) * self.unit_price
        )


class TestCumulativePurchaseQuantity:
    feature_object = CumulativePurchaseQuantity(
        aggregation_level="product_nbr", time_freq="week"
    )

    @classmethod
    def setup_class(cls):
        cls.feature_object.set_historical_data(SAMPLE_TRX_ARRAY)
        # check if the historical_data attribute is updated
        assert cls.feature_object._historical_data.shape == SAMPLE_TRX_ARRAY.shape

    def test_get_historical_features(self):
        historical_features = self.feature_object.get_historical_feature()
        assert historical_features.shape == self.feature_object._historical_data.shape
        assert_equal(
            historical_features[1],  # test the last purchase quantity feature at week 2
            np.array([[1, 0, 1], [0, 1, 0]]),
        )
        assert_equal(
            historical_features[2],  # test the last purchase quantity feature at week 3
            np.array([[1, 0, 3], [1, 2, 1]]),
        )

    def test_write_online_data(self, sample_online_array):
        self.feature_object.write_online_data(sample_online_array)
        # check if the online_data attribute is updated
        assert self.feature_object.online_data.shape == sample_online_array.shape

    def test_update_online_features(self):
        self.feature_object.update_online_feature()
        assert (
            self.feature_object.online_feature_array.shape
            == self.feature_object.online_data.shape
        )
        # check if the online_feature is updated with the new input data
        assert_equal(
            self.feature_object.online_feature_array, np.array([[3, 2, 4], [1, 2, 2]])
        )

    def test_get_online_features(self):
        online_features = self.feature_object.get_online_feature()
        assert online_features.shape == self.feature_object.online_feature_array.shape
        assert_equal(online_features, np.array([[3, 2, 4], [1, 2, 2]]))


class TestCumulativeMoneySpent:
    feature_object = CumulativeMoneySpent(
        aggregation_level="product_nbr", time_freq="week"
    )
    unit_price = 0.9

    @classmethod
    def setup_class(cls):
        SAMPLE_TRX_ARRAY_revenue = SAMPLE_TRX_ARRAY * cls.unit_price
        cls.feature_object.set_historical_data(SAMPLE_TRX_ARRAY_revenue)
        # check if the historical_data attribute is updated
        assert cls.feature_object._historical_data.shape == SAMPLE_TRX_ARRAY.shape

    def test_get_historical_features(self):
        historical_features = self.feature_object.get_historical_feature()
        assert historical_features.shape == self.feature_object._historical_data.shape
        assert_equal(
            historical_features[1],  # test the last purchase quantity feature at week 2
            np.array([[1, 0, 1], [0, 1, 0]]) * self.unit_price,
        )
        assert_equal(
            historical_features[2],  # test the last purchase quantity feature at week 3
            np.array([[1, 0, 3], [1, 2, 1]]) * self.unit_price,
        )

    def test_write_online_data(self, sample_online_array):
        sample_online_array_revenue = sample_online_array * self.unit_price
        self.feature_object.write_online_data(sample_online_array_revenue)
        # check if the online_data attribute is updated
        assert (
            self.feature_object.online_data.shape == sample_online_array_revenue.shape
        )

    def test_update_online_features(self):
        self.feature_object.update_online_feature()
        assert (
            self.feature_object.online_feature_array.shape
            == self.feature_object.online_data.shape
        )
        # check if the online_feature is updated with the new input data
        assert_equal(
            self.feature_object.online_feature_array,
            np.array([[3, 2, 4], [1, 2, 2]]) * self.unit_price,
        )

    def test_get_online_features(self):
        online_features = self.feature_object.get_online_feature()
        assert online_features.shape == self.feature_object.online_feature_array.shape
        assert_equal(
            online_features, np.array([[3, 2, 4], [1, 2, 2]]) * self.unit_price
        )


class TestCumulativePurchaseCount:
    feature_object = CumulativePurchaseCount(
        aggregation_level="product_nbr", time_freq="week"
    )

    @classmethod
    def setup_class(cls):
        cls.feature_object.set_historical_data(SAMPLE_TRX_ARRAY)
        # check if the historical_data attribute is updated
        assert cls.feature_object._historical_data.shape == SAMPLE_TRX_ARRAY.shape

    def test_get_historical_features(self):
        historical_features = self.feature_object.get_historical_feature()
        assert historical_features.shape == self.feature_object._historical_data.shape
        assert_equal(
            historical_features[1],  # test the last purchase quantity feature at week 2
            np.array([[1, 0, 1], [0, 1, 0]]),
        )
        assert_equal(
            historical_features[2],  # test the last purchase quantity feature at week 3
            np.array([[1, 0, 2], [1, 2, 1]]),
        )

    def test_write_online_data(self, sample_online_array):
        self.feature_object.write_online_data(sample_online_array)
        # check if the online_data attribute is updated
        assert self.feature_object.online_data.shape == sample_online_array.shape

    def test_update_online_features(self):
        self.feature_object.update_online_feature()
        assert (
            self.feature_object.online_feature_array.shape
            == self.feature_object.online_data.shape
        )
        # check if the online_feature is updated with the new input data
        assert_equal(
            self.feature_object.online_feature_array, np.array([[2, 2, 3], [1, 2, 2]])
        )

    def test_get_online_features(self):
        online_features = self.feature_object.get_online_feature()
        assert online_features.shape == self.feature_object.online_feature_array.shape
        assert_equal(online_features, np.array([[2, 2, 3], [1, 2, 2]]))


class TestTimeSinceLastPurchase:
    feature_object = TimeSinceLastPurchase(
        aggregation_level="product_nbr", time_freq="week"
    )

    @classmethod
    def setup_class(cls):
        cls.feature_object.set_historical_data(SAMPLE_TRX_ARRAY)
        # check if the historical_data attribute is updated
        assert cls.feature_object._historical_data.shape == SAMPLE_TRX_ARRAY.shape

    def test_get_historical_features(self):
        historical_features = self.feature_object.get_historical_feature()
        assert historical_features.shape == self.feature_object._historical_data.shape
        assert_equal(
            historical_features[1],  # test the last purchase quantity feature at week 2
            np.array([[1, 0, 1], [0, 1, 0]]),
        )
        assert_equal(
            historical_features[2],  # test the last purchase quantity feature at week 3
            np.array([[2, 0, 1], [1, 1, 1]]),
        )

    def test_write_online_data(self, sample_online_array):
        self.feature_object.write_online_data(sample_online_array)
        # check if the online_data attribute is updated
        assert self.feature_object.online_data.shape == sample_online_array.shape

    def test_update_online_features(self):
        self.feature_object.update_online_feature()
        assert (
            self.feature_object.online_feature_array.shape
            == self.feature_object.online_data.shape
        )
        # check if the online_feature is updated with the new input data
        assert_equal(
            self.feature_object.online_feature_array,
            np.array([[3, 1, 2], [4, 4, 1]]),
        )

    def test_get_online_features(self):
        online_features = self.feature_object.get_online_feature()
        assert online_features.shape == self.feature_object.online_feature_array.shape
        assert_equal(online_features, np.array([[3, 1, 2], [4, 4, 1]]))


class TestLagPurchaseQuantity:
    lag_time = 2
    feature_object = LagPurchaseQuantity(
        aggregation_level="product_nbr", time_freq="week", lag_time=lag_time
    )

    @classmethod
    def setup_class(cls):
        cls.feature_object.set_historical_data(SAMPLE_TRX_ARRAY)
        # check if the historical_data attribute is updated
        assert cls.feature_object._historical_data.shape == SAMPLE_TRX_ARRAY.shape

    def test_get_historical_features(self):
        historical_features = self.feature_object.get_historical_feature()
        assert historical_features.shape == self.feature_object._historical_data.shape
        assert_equal(
            historical_features[1],  # test the last purchase quantity feature at week 2
            np.array([[0, 0, 0], [0, 0, 0]]),
        )
        assert_equal(
            historical_features[2],  # test the last purchase quantity feature at week 3
            np.array([[1, 0, 1], [0, 1, 0]]),
        )
        assert_equal(
            historical_features[3],  # test the last purchase quantity feature at week 3
            np.array([[0, 0, 2], [1, 1, 1]]),
        )

    def test_write_online_data(self, sample_online_array):
        self.feature_object.write_online_data(sample_online_array)
        # check if the online_data attribute is updated
        assert self.feature_object.online_data.shape == sample_online_array.shape

    def test_update_online_features(self):
        self.feature_object.update_online_feature()
        assert (
            self.feature_object.online_feature_array.shape
            == (self.lag_time,) + self.feature_object.online_data.shape
        )
        # check if the online_feature is updated with the new input data
        assert_equal(
            self.feature_object.online_feature_array[0],
            np.array([[0, 0, 1], [0, 0, 0]], dtype=np.float32),
        )

    def test_get_online_features(self):
        online_features = self.feature_object.get_online_feature()
        assert (
            online_features.shape == self.feature_object.online_feature_array[0].shape
        )
        assert_equal(
            online_features, np.array([[0, 0, 1], [0, 0, 0]], dtype=np.float32)
        )

    def test_update_online_features_again(self, sample_online_array):
        self.feature_object.write_online_data(sample_online_array)
        self.feature_object.update_online_feature()
        online_features = self.feature_object.get_online_feature()
        assert_equal(
            online_features, np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32)
        )


class TestLagMoneySpent:
    lag_time = 1
    feature_object = LagMoneySpent(
        aggregation_level="product_nbr", time_freq="week", lag_time=lag_time
    )

    def test_default_name(self):
        assert self.feature_object.name == "lag_money_spent"
