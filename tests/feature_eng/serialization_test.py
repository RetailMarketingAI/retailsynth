import os

import pytest

from retailsynth.utils.storage import clear_feature_directory


def test_serialization(sample_config, serialized_trx_arrays):
    data_path = sample_config.paths.product_feature_path
    assert os.path.exists(data_path + "sales_amt.zarr")
    assert os.path.exists(data_path + "unit_price.zarr")
    assert os.path.exists(data_path + "item_qty.zarr")


@pytest.mark.last
def test_clear_directory(sample_config):
    data_path = sample_config.paths.processed_data
    clear_feature_directory(data_path)
    assert ~os.path.isdir(data_path)
