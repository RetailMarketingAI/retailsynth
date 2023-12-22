import glob

import pandas as pd


class FeatureLoader:
    """Object to load feature arrays from a given storage directory

    Attributes
    ----------
    store_feature_path: str
        directory where stores the store-level feature arrays
    category_feature_path: str
        directory where stores the category-level feature arrays
    product_feature_path: str
        directory where stores the product-level feature arrays
    """

    def __init__(self, store_feature_path, category_feature_path, product_feature_path):
        self.store_feature_path = store_feature_path
        self.category_feature_path = category_feature_path
        self.product_feature_path = product_feature_path

    def _load_features(
        self,
        path: str,
    ):
        feature_paths = glob.glob(path + "/*.parquet")
        df_list = [pd.read_parquet(p) for p in feature_paths]
        return pd.concat(df_list, axis=1)

    def load_store_features(self):
        index_cols = ["week", "customer_key", "store"]
        store_featues = (
            self._load_features(
                path=self.store_feature_path,
            )
            .reset_index()
            .drop(columns=["store"])
        )
        return store_featues.set_index(index_cols[:-1])

    def load_category_features(self):
        index_cols = ["week", "customer_key", "category_nbr"]

        return self._load_features(
            path=self.category_feature_path,
        ).set_index(index_cols)

    def load_product_features(self):
        index_cols = ["week", "customer_key", "product_nbr"]

        return self._load_features(
            path=self.product_feature_path,
        ).set_index(index_cols)


def initialize_feature_loader(path_config):
    store_path = path_config.store_feature_path
    category_path = path_config.category_feature_path
    product_path = path_config.product_feature_path

    feature_loader = FeatureLoader(
        store_feature_path=store_path,
        category_feature_path=category_path,
        product_feature_path=product_path,
    )

    return feature_loader
