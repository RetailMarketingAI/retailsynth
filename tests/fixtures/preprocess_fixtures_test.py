import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from retailsynth.base_config import load_config_store
from retailsynth.datasets.complete_journey.preprocess_pipeline import PreprocessPipeline

load_config_store()


@pytest.fixture(scope="module", autouse=True)
def sample_config():
    with initialize(version_base=None, config_path="../cfg"):
        cfg = compose(config_name="indiv_env_small_dataset")
        cfg = OmegaConf.to_object(cfg)
    return cfg


@pytest.fixture(scope="module")
def raw_sample_tables(sample_config):
    complete_journey_preprocess_pipeline = PreprocessPipeline(
        raw_data_config=sample_config.raw_data,  # type: ignore
    )
    complete_journey_preprocess_pipeline.apply_all_transformations()

    dataset = complete_journey_preprocess_pipeline.dataset
    return dataset.customers, dataset.products, dataset.transactions


@pytest.fixture(scope="session")
def customers():
    return pd.DataFrame(
        {
            "AGE_DESC": ["18", "25", "35", "45"],
            "MARITAL_STATUS_CODE": ["M", "S", "M", "M"],
            "INCOME_DESC": ["Under $15K", "$15-24K", "$50-74K", "$100-150K"],
            "HOMEOWNER_DESC": ["Unknown", "Unknown", "Unknown", "Unknown"],
            "HH_COMP_DESC": [
                "Single Female",
                "2 Adults No Kids",
                "2 Adults Kids",
                "Single Female",
            ],
            "HOUSEHOLD_SIZE_DESC": ["1", "2", "3", "4+"],
            "KID_CATEGORY_DESC": ["None/Unknown", "1", "None/Unknown", "2"],
            "household_key": [1, 2, 3, -3],  # Invalid value should be greater than 0
        }
    )


@pytest.fixture(scope="session")
def products():
    return pd.DataFrame(
        {
            "PRODUCT_ID": [101, 102, 103, 0],  # invalid row
            "MANUFACTURER": [1, 2, 3, 4],
            "DEPARTMENT": ["Grocery", "Produce", "Dairy", "Snacks"],
            "BRAND": ["Brand1", "Brand2", "Brand3", "Brand4"],
            "COMMODITY_DESC": ["Commodity1", "Commodity2", "Commodity3", "Commodity4"],
            "SUB_COMMODITY_DESC": [
                "SubCommodity1",
                "SubCommodity2",
                "SubCommodity3",
                "SubCommodity4",
            ],
            "CURR_SIZE_OF_PRODUCT": ["12 oz", "16 oz", "32 oz", "8 oz"],
        }
    )


@pytest.fixture(scope="session")
def transactions():
    return pd.DataFrame(
        {
            "household_key": [1, 2, 3],
            "BASKET_ID": [10001, 10002, 10003],
            "DAY": [1, 1, 2],
            "PRODUCT_ID": [101, 102, 103],
            "QUANTITY": [1, 2, 1],
            "SALES_VALUE": [2.0, 4.5, 3.0],
            "STORE_ID": [101, 102, 103],
            "RETAIL_DISC": [0.5, 0.0, 0.25],
            "TRANS_TIME": [1000, 1100, 1200],
            "WEEK_NO": [1, 1, 2],
            "COUPON_DISC": [-0.2, 0.0, 0.0],
            "COUPON_MATCH_DISC": [0.0, 0.0, 0.1],  # Invalid value should be less than 0
        }
    )
