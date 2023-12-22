import numpy as np

from retailsynth.feature_eng.features import (
    AveragePurchaseFrequency,
    AveragePurchaseQuantity,
)
from tests.feature_eng.features_test import SAMPLE_TRX_ARRAY


class TestAveragePurchaseQuantity:
    feature_object = AveragePurchaseQuantity(
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
        np.testing.assert_allclose(
            historical_features[1],  # test feature at week 2
            np.array([[1, 0, 1], [0, 1, 0]]),
        )
        np.testing.assert_allclose(
            historical_features[2],  # test at week 3
            np.array([[1, 0, 1.5], [1, 1, 1]]),
        )

    def test_write_online_data(self, sample_online_array):
        self.feature_object.write_online_data(sample_online_array)
        # check if the online_data attribute is updated
        assert self.feature_object.online_data.shape == sample_online_array.shape

    def test_update_online_features(self):
        np.testing.assert_allclose(
            self.feature_object.online_data, [[0, 1, 0], [0, 0, 1]]
        )
        self.feature_object.update_online_feature()

        # check if the online_feature is updated with the new input data
        np.testing.assert_allclose(
            self.feature_object.online_feature_array,
            np.array([[3 / 2, 1, 4 / 3], [1, 1, 1]]),
        )

    def test_get_online_features(self):
        online_features = self.feature_object.get_online_feature()
        assert online_features.shape == self.feature_object.online_feature_array.shape
        np.testing.assert_allclose(
            online_features,
            np.array([[3 / 2, 1, 4 / 3], [1, 1, 1]]),
        )


class TestAvgPurchaseFrequency:
    feature_object = AveragePurchaseFrequency(
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
        np.testing.assert_allclose(
            historical_features[1],  # test the last purchase quantity feature at week 2
            np.array([[1, 0, 1], [0, 1, 0]]),
        )
        np.testing.assert_allclose(
            historical_features[2],  # test the last purchase quantity feature at week 3
            np.array([[1, 0, 2], [1, 2, 1]]) / 2,
        )

    def test_write_online_data(self, sample_online_array):
        self.feature_object.write_online_data(sample_online_array)
        # check if the online_data attribute is updated
        assert self.feature_object.online_data.shape == sample_online_array.shape

    def test_update_online_features(self):
        np.testing.assert_allclose(
            self.feature_object.online_data, [[0, 1, 0], [0, 0, 1]]
        )
        self.feature_object.update_online_feature()
        assert (
            self.feature_object.online_feature_array.shape
            == self.feature_object.online_data.shape
        )
        # check if the online_feature is updated with the new input data
        np.testing.assert_allclose(
            self.feature_object.online_feature_array,
            np.array([[2, 2, 3], [1, 2, 2]]) / 5,
        )

    def test_get_online_features(self):
        online_features = self.feature_object.get_online_feature()
        assert online_features.shape == self.feature_object.online_feature_array.shape
        np.testing.assert_allclose(
            online_features, np.array([[2, 2, 3], [1, 2, 2]]) / 5
        )
