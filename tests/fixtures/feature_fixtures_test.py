import numpy as np
import pandas as pd
import pytest

from retailsynth.feature_eng.feature_generator import serialize_historical_transactions
from retailsynth.feature_eng.feature_pipeline import run_feature_generation
from retailsynth.feature_eng.indices import IndexMap
from retailsynth.feature_eng.response_tracker import ResponseTracker


@pytest.fixture(scope="module")
def sample_trx_array():
    historical_data = np.array(
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
    return historical_data


@pytest.fixture(scope="module")
def sample_online_array():
    online_data = np.array(
        [
            [0, 1, 0],  # customer 1, purchase quantity of 3 products
            [0, 0, 1],  # customer 2, purchase quantity of 3 products
        ]
    )
    return online_data


@pytest.fixture(scope="module")
def sample_transaction_df(raw_sample_tables):
    _, _, transactions = raw_sample_tables
    return transactions


@pytest.fixture(scope="module")
def tracker():
    time_column = "week"
    customer_list = [1001, 1002, 1003, 1004]
    time_list = [1, 2, 3, 4, 5, 6, 7, 8]

    products = pd.DataFrame(
        {
            "product_nbr": np.arange(5),
            "category_nbr": [0, 1, 0, 1, 0],
        }
    )
    return ResponseTracker(time_column, customer_list, time_list, products)


@pytest.fixture(scope="module")
def sample_indices(raw_sample_tables):
    _, _, txns = raw_sample_tables
    txns = txns.sort_index()
    time_list = txns.index.levels[0].tolist()
    customer_list = txns.index.levels[1].tolist()
    item_list = txns.index.levels[2].tolist()

    return {
        "time_index": IndexMap(time_list, "week"),
        "customer_index": IndexMap(customer_list, "customer_key"),
        "item_index": IndexMap(item_list, "product_nbr"),
    }


@pytest.fixture(scope="module")
def serialized_trx_arrays(sample_config, raw_sample_tables):
    data_path = sample_config.paths.product_feature_path
    customer_chunk_size = sample_config.feature_setup.customer_chunk_size
    serialize_historical_transactions(
        *raw_sample_tables, data_path, customer_chunk_size
    )


@pytest.fixture(scope="module")
def sample_derived_features(sample_config, serialized_trx_arrays, raw_sample_tables):
    data_path = sample_config.paths.product_feature_path
    store_feature_path = sample_config.paths.store_feature_path
    category_feature_path = sample_config.paths.category_feature_path
    product_feature_path = sample_config.paths.product_feature_path
    run_feature_generation(
        *raw_sample_tables,
        sample_config.feature_setup,
        txns_array_path=data_path,
        store_feature_path=store_feature_path,
        category_feature_path=category_feature_path,
        product_feature_path=product_feature_path,
    )
