import logging
from pathlib import Path

import pandas as pd


class FeatureVizDataLayer:
    def _load_store_feature_df(self):
        logging.info("Loading store feature df")
        # prepare store visit features
        store_feature_df = self.feature_loader.load_store_features().dropna()
        store_feature_df.loc[:, "store_visit"] = (
            store_feature_df["item_qty"] > 0
        ).astype(int)
        _to_parquet(store_feature_df, self.data_path, "store_visit_df")
        self.store_feature_df = store_feature_df

    def _load_category_feature_df(self):
        logging.info("Loading category feature df")
        # prepare category choice features
        category_choice_df = self.feature_loader.load_category_features().dropna()
        category_choice_df["category_choice"] = (
            category_choice_df["item_qty"] > 0
        ).astype(int)
        _to_parquet(category_choice_df, self.data_path, "category_choice_df")
        self.category_choice_df = category_choice_df

    def _load_product_choice_df(self):
        logging.info("Loading product choice df")
        # prepare product choice features
        product_choice_df = self.feature_loader.load_product_features().dropna()
        product_choice_df["product_choice"] = (
            product_choice_df["item_qty"] > 0
        ).astype(int)
        _to_parquet(product_choice_df, self.data_path, "product_choice_df")
        self.product_choice_df = product_choice_df

    def _load_product_demand_df(self):
        # Assume we are using same features for product choice and demand
        product_demand_df = self.product_choice_df.query("item_qty > 0")
        _to_parquet(product_demand_df, self.data_path, "product_demand_df")
        self.product_demand_df = product_demand_df

    def prepare_visualization_features(
        self,
        feature_loader,
        category_product_mapping,
        data_path,
    ):
        """Query the feature store for features on the store, category, and product level in a data frame format.

        Parameters
        ----------
            feature loader (FeatureLoader): object to load features from local
            category_product_mapping (pd.DataFrame): mapping between product and category
            data_path (str): path to save the data frame

        Returns
        -------
            dict: dictionary of features with keys "store_visit", "category_choice", "product_choice", "product_demand"
        """
        assert set(category_product_mapping.columns) == set(
            ["product_nbr", "category_nbr"]
        )
        self.feature_loader = feature_loader
        self.data_path = data_path
        self.category_product_mapping = category_product_mapping

        self._load_store_feature_df()
        self._load_category_feature_df()
        self._load_product_choice_df()
        self._load_product_demand_df()

        return {
            "store_visit": self.store_feature_df,
            "category_choice": self.category_choice_df,
            "product_choice": self.product_choice_df,
            "product_demand": self.product_demand_df,
        }


def _to_parquet(df: pd.DataFrame, path: str, name: str):
    float32_cols = list(df.select_dtypes(include=["float32"]).columns)
    new_types = dict((col, "float") for col in float32_cols)
    df = df.astype(new_types)
    df.to_parquet(Path(path, f"{name}.parquet").resolve())
