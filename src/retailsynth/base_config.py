"""base config for the project."""
from dataclasses import asdict, field
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass

from retailsynth.datasets.complete_journey.config import (
    Colnames,
    Preprocess,
    ProductAnnotation,
    TimeSlices,
)


@dataclass
class Paths:
    """Define the path to store any intermediate tables if there's any and trained models."""

    # use the following to override the default
    processed_data: str = "data/processed/complete_journey_data"
    txns_array_path: str = "data/processed/complete_journey_data/txns_array/"
    logs_path: str = "logs/"
    store_feature_path: str = "data/processed/complete_journey_data/store_features/"
    category_feature_path: str = (
        "data/processed/complete_journey_data/category_features/"
    )
    product_feature_path: str = "data/processed/complete_journey_data/product_features/"


@dataclass
class RawData:
    """A complete list of parameters needed to define for a source of input data."""

    colnames: Colnames = Colnames()
    preprocess: Preprocess = Preprocess()
    product_annotation: ProductAnnotation = ProductAnnotation()
    if_local_data: bool = (
        False  # whether to use local datasets or download the Complete Journey datasets
    )
    # if use local data, the following parameters are needed to load the local datasets
    raw_data_path: str = ""  # name of the folder that contains the following files
    customer_table_name: str = ""  # customer demographics table name
    product_table_name: str = ""  # product description table name
    transaction_table_name: str = ""

    # time slices to use for testing and training
    time_slices: TimeSlices = TimeSlices()


@dataclass
class SyntheticData:
    """A complete list of parameters needed to define for a source of input data."""

    synthetic_data_setup: Dict
    sample_time_steps: int = 30


@dataclass
class FeatureSetup:
    """Summary of the trx feature view setup, specify each argument required to initialize a trx feature view object.

    Attributes
    ----------
        aggregation_column (str): the column from trx table to aggregate on and compute features, for example: product_nbr, category_nbr, etc.
        features (List[str]): list of original columns from the trx table and derived features to track in the feature store, for example: item_qty, sales_amt, unit_price, discount_portion, etc.
        chunks (int): number of chunks to use for parallel processing
    """

    features: Optional[List[str]] = field(
        default_factory=lambda: ["item_qty", "discount_portion", "unit_price"]
    )
    aggregation_column: str = "product_nbr"
    chunks: Optional[int] = 1

    def to_dict(self):
        """Transform the object to a dictionary.

        Returns
        -------
        dict
            dict of object attributes
        """
        return asdict(self)


@dataclass
class AllFeatureSetup:
    """Summary of the feature store setup, specify each argument required to initialize a feature store object.

    Attributes
    ----------
        time_column (str): time column to use for feature store, for example: week or day
        customer_feature_view (FeatureSetup): feature view setup for customer level features
        customer_product_feature_view (FeatureSetup): feature view setup for customer-product level features
        customer_category_feature_view (FeatureSetup): feature view setup for customer-category level features
    """

    time_column: str = "week"
    customer_feature: FeatureSetup = FeatureSetup(
        aggregation_column="all",
        features=[
            "item_qty",
            "unit_price",
            "discount_portion",
            "time_since_last_purchase",
        ],
    )  # type: ignore
    customer_product_feature: FeatureSetup = FeatureSetup(aggregation_column="product_nbr", chunks=10)  # type: ignore
    customer_category_feature: FeatureSetup = FeatureSetup(aggregation_column="category_nbr")  # type: ignore
    customer_chunk_size: Optional[int] = 100


@dataclass
class Config:
    """A complete list of parameters used in this codebase."""

    raw_data: Optional[RawData] = RawData()
    feature_setup: AllFeatureSetup = AllFeatureSetup()
    paths: Paths = Paths()  # need to always define this parameter in the config
    synthetic_data: Optional[SyntheticData] = None
    n_customers_sampled: Optional[int] = 100
    n_workers: Optional[int] = 4


def load_config_store():
    """
    Load the configuration store.

    The name 'base_config' is used for matching it with the main.yaml's default section,
    so that the main config becomes an instance of the defined Config class.

    Returns
    -------
        None
    """
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)


load_config_store()
