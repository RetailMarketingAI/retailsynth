import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from retailsynth.base_config import AllFeatureSetup
from retailsynth.feature_eng.feature_generator import (
    CategoryFeatureGenerator,
    ProductFeatureGenerator,
    StoreFeatureGenerator,
)
from retailsynth.feature_eng.feature_utils import map_feature_name_to_object
from retailsynth.feature_eng.features import FeatureCatalog
from retailsynth.feature_eng.response_tracker import ResponseTracker


@dataclass
class FeatureGenerationPipeline:
    chunk_idx: int
    customers: pd.DataFrame
    products: pd.DataFrame
    transactions: pd.DataFrame
    feature_setup: AllFeatureSetup
    txns_array_path: str
    customer_chunk_size: int

    def __post_init__(self):
        """Derive some attributes from given arguments and run the feature generation at different aggregation levels."""
        self.all_weeks = self.transactions.index.levels[0].values.tolist()
        chk_start = self.chunk_idx * self.customer_chunk_size
        chk_end = chk_start + self.customer_chunk_size
        self.customers = self.customers.iloc[chk_start:chk_end]
        self.all_customers = self.customers.index.values
        self.all_products = self.products.index.values.tolist()
        self._response_tracker = ResponseTracker(
            "week",
            self.all_customers,
            self.all_weeks,
            self.products.reset_index(),
        )
        self.run_all()

    def run_store_feature_generator(
        self,
        feature_path: Path = None,
    ):
        if feature_path is None:
            feature_path = Path(self.txns_array_path).parent / "store_features"
        logging.info(f"Saving store features to {feature_path}")
        """
        pipeline method to generate store-level features
        """
        features = {
            feature: map_feature_name_to_object(feature, FeatureCatalog)  # type: ignore
            for feature in self.feature_setup.customer_feature.features  # type: ignore
        }
        os.makedirs(feature_path, exist_ok=True)
        t1 = time.time()
        fg = StoreFeatureGenerator(
            customer_list=self.all_customers,
            week_list=self.all_weeks,
            product_list=self.all_products,
            features=features,
            feature_path=feature_path,
            chunk_idx=self.chunk_idx,
        )
        store_visit_array = fg.write_historical_features(
            txns_array_path=self.txns_array_path
        )

        self._response_tracker.condition_on_store_visit(
            store_visit_array.to_dataframe().reset_index().drop(columns=["store"])
        )
        logging.info(
            f"Write historical store features to {feature_path} in {time.time() - t1: .2f} seconds."
        )

    def run_category_feature_generator(
        self,
        feature_path: Path = None,
    ):
        """Pipeline method to generate category-level features."""
        if feature_path is None:
            feature_path = Path(self.txns_array_path).parent / "category_features"
        logging.info(f"Saving category features to {feature_path}")
        features = {
            feature: map_feature_name_to_object(feature, FeatureCatalog)  # type: ignore
            for feature in self.feature_setup.customer_category_feature.features  # type: ignore
        }
        os.makedirs(feature_path, exist_ok=True)
        t1 = time.time()
        fg = CategoryFeatureGenerator(
            customer_list=self.all_customers,
            week_list=self.all_weeks,
            product_list=self.all_products,
            features=features,
            feature_path=feature_path,
            product_category_mapping=self.products,
            chunk_idx=self.chunk_idx,
        )

        filtered_index = self._response_tracker.get_filtering_index(
            agg_key="category_nbr"
        )
        cat_choice_array = fg.write_historical_features(
            txns_array_path=self.txns_array_path,
            filtered_index=filtered_index,
        )
        self._response_tracker.condition_on_category_choice(
            cat_choice_array.to_dataframe().reset_index()
        )
        logging.info(
            f"Write historical categorical features for category to {feature_path} in {time.time() - t1: .2f} seconds."
        )

    def run_product_feature_generator(
        self,
        feature_path: Path = None,
    ):
        """Pipeline method to generate product-level features."""
        if feature_path is None:
            feature_path = Path(self.txns_array_path).parent / "product_features"
        logging.info(f"Saving product features to {feature_path}")

        features = {
            feature: map_feature_name_to_object(feature, FeatureCatalog)  # type: ignore
            for feature in self.feature_setup.customer_product_feature.features  # type: ignore
        }

        fg = ProductFeatureGenerator(
            customer_list=self.all_customers,
            week_list=self.all_weeks,
            product_list=self.all_products,
            features=features,
            feature_path=feature_path,
            chunk_idx=self.chunk_idx,
        )
        filtered_index = self._response_tracker.get_filtering_index(
            agg_key="product_nbr"
        )

        fg.write_historical_features(
            txns_array_path=self.txns_array_path, filtered_index=filtered_index
        )
        logging.info(f"Write historical product features to {feature_path}")

    def run_all(self):
        self.run_store_feature_generator()
        self.run_category_feature_generator()
        self.run_product_feature_generator()


def run_feature_generation(
    customers,
    products,
    transactions,
    feature_setup,
    txns_array_path,
    customer_chunk_size=None,
    n_workers=1,
):
    customer_list = customers.index.values
    n_customers = len(customer_list)

    logging.info(
        f"Preparing features on customers chunks of size {customer_chunk_size}"
    )
    fg_pipeline = partial(
        FeatureGenerationPipeline,
        customers=customers,
        products=products,
        transactions=transactions,
        feature_setup=feature_setup,
        txns_array_path=txns_array_path,
        customer_chunk_size=customer_chunk_size,
    )
    chunk_idxs = np.arange(int(n_customers / customer_chunk_size) + 1)
    if n_workers > 1:
        with Pool(n_workers) as p:
            p.map(fg_pipeline, chunk_idxs)
    else:
        for chunk_idx in chunk_idxs:
            fg_pipeline(chunk_idx)
