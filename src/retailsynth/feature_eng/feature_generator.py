import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import dask.dataframe as dd
import pandas as pd
import xarray as xr

from retailsynth.feature_eng.feature_utils import (
    convert_trx_df_to_numpy,
    convert_trx_df_to_numpy_fill_with_mean,
    generate_index_mapping,
)


def serialize_historical_transactions(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    transaction: pd.DataFrame,
    data_path: str,
    customer_chunk_size: int = 100,
):
    time_list = transaction.index.levels[0].tolist()
    customer_list = customers.index.values.tolist()
    product_list = products.index.values.tolist()

    indices = generate_index_mapping(
        transaction,
        time_list,
        customer_list,
        product_list,
        item_name="product_nbr",
    )
    dims = ["week", "customer_key", "product_nbr"]
    # convert it to an xarray datasets
    coords = {
        "week": time_list,
        "customer_key": customer_list,
        "product_nbr": product_list,
    }
    variables_agg_mean = [
        ("item_qty", False),
        ("sales_amt", False),
        ("unit_price", True),
        ("discount_portion", True),
    ]
    for var_agg_mean in variables_agg_mean:  # type: ignore
        var, mean_fill = var_agg_mean
        logging.info(f"Extracting {var} to zarr in {data_path}")
        if mean_fill:
            dataset = convert_trx_df_to_numpy_fill_with_mean(transaction, var, indices)
        else:
            dataset = convert_trx_df_to_numpy(transaction, var, indices)
        (
            xr.DataArray(dataset, dims=dims, coords=coords, name=var)
            .chunk(
                {
                    "customer_key": customer_chunk_size,
                    "week": len(time_list),
                    "product_nbr": len(product_list),
                }
            )
            .to_zarr(data_path + f"{var}.zarr", mode="w")
        )


@dataclass
class ProductFeatureGenerator:
    """Object to generate product-level features

    Attributes
    ----------
    customer_list: List[int]
        List of customer keys to consider
    week_list: List[int]
        List of weeks to consider
    product_list: List[int]
        List of product nbrs to consider
    features: Dict[str, BaseFeature]
        dictionary of identical feature or derived features to generate
    feature_path: str
        directory to store derived features
    chunk_idx: str
        chunk_idx used for naming file
    """

    customer_list: List[int]
    week_list: List[int]
    product_list: List[
        int
    ]  # being the complete list of all products in the trx or a subset of products
    features: Dict
    feature_path: Path
    chunk_idx: int

    def __post_init__(self):
        self.agg_level = "product_nbr"
        self.features = {
            feature_name: feature(self.agg_level, "week", feature_name)
            for feature_name, feature in self.features.items()
        }

    def _load_data_array(self, data_path: str, variable_name: str) -> xr.Dataset:
        """load the historical trx array from the given path

        Parameters
        ----------
        data_path: str
            directory where stores the historical trx array
        variable_name: str
            a specific variable to load

        Returns
        ---------
        vars: xr.Dataset
            historical trx array on the product level
        """
        vars = xr.open_mfdataset(
            data_path + f"{variable_name}.zarr",
            engine="zarr",
            combine="by_coords",
        )

        vars = vars.sel(
            customer_key=self.customer_list,
            week=self.week_list,
            product_nbr=self.product_list,
        ).load()

        return vars

    def write_historical_features(
        self, txns_array_path: str, filtered_index: pd.MultiIndex = None
    ) -> xr.DataArray:
        """Compute derived features and store them to the given feature path

        Parameters
        ---------
        txns_array_path: str
            directory where stores the historical trx array
        filtered_index: pd.MultiIndex
            index of the filtered historical trx array

        Returns
        --------
        item_qty_array: xr.DataArray
        """
        saved_features: List[str] = []
        # iterate through historical data trx arrays
        for data_var_name in [
            "item_qty",
            "sales_amt",
            "unit_price",
            "discount_portion",
        ]:
            trx_array = self._load_data_array(txns_array_path, data_var_name)
            for name, feature in self.features.items():
                # find if one feature is derived from the current historical trx array
                if feature.target_column == data_var_name:
                    feature.set_historical_data(trx_array[data_var_name].values)
                    # compute features
                    derived_feature = feature.get_historical_feature()
                    derived_feature = xr.DataArray(
                        derived_feature,
                        dims=["week", "customer_key", self.agg_level],
                        coords=trx_array.coords,
                        name=name,
                    )
                    file_path = Path(
                        os.getcwd(),
                        self.feature_path,
                        f"{name}.parquet",
                    )
                    df = derived_feature.to_dataframe()
                    if filtered_index is not None:
                        df = df.reindex(filtered_index, axis="index")
                    if len(saved_features) == 0:
                        index_df = df.index.to_frame().reset_index(drop=True)
                        index_path = (
                            Path(
                                os.getcwd(),
                                self.feature_path,
                                "index.parquet",
                            ),
                        )
                        dd.from_pandas(index_df, npartitions=1).to_parquet(
                            path=index_path,
                            engine="pyarrow",
                            append=False,
                            ignore_divisions=True,
                            name_function=lambda x: f"index_{self.chunk_idx}.parquet",
                        )
                    dd.from_pandas(df.reset_index(drop=True), npartitions=1).to_parquet(
                        path=file_path,
                        engine="pyarrow",
                        append=False,
                        ignore_divisions=True,
                        name_function=lambda x: f"{name}_{self.chunk_idx}.parquet",
                    )
                    saved_features.append(name)
                    if feature.name == "item_qty":
                        item_qty_array = derived_feature
        skipped_features = set(self.features.keys()) - set(saved_features)
        assert (
            len(skipped_features) == 0
        ), f"No raw trx array is available to derive features {skipped_features}"

        return item_qty_array


@dataclass
class StoreFeatureGenerator(ProductFeatureGenerator):
    """Object to generate store-level features

    Attributes
    -----------
    loading_method: Dict[str, str]
        dictionary of loading method for each historical trx array, sum or mean
    agg_level: str
        aggregation level, default to be store
    features: dict
        dictionary of identical feature or derived features to generate
    """

    def __post_init__(self):
        self.loading_method = {
            "item_qty": "sum",
            "sales_amt": "sum",
            "unit_price": "mean",
            "discount_portion": "mean",
        }
        self.agg_level = "store"
        self.features = {
            feature_name: feature(self.agg_level, "week", feature_name)
            for feature_name, feature in self.features.items()
        }

    def _load_data_array(self, data_path, variable_name) -> xr.Dataset:
        vars = xr.open_mfdataset(
            data_path + f"{variable_name}.zarr",
            engine="zarr",
            combine="by_coords",
        )

        vars = vars.sel(
            customer_key=self.customer_list,
            week=self.week_list,
            product_nbr=self.product_list,
        ).load()
        # load raw data on the product level and aggregate it to the store level
        loading_method = self.loading_method[variable_name]
        return self._aggregate_raw_trx_array(vars, loading_method)

    def _aggregate_raw_trx_array(
        self, trx_array: xr.DataArray, method="sum"
    ) -> xr.Dataset:
        """Aggregate the raw trx arrays to the store level

        Parameters
        ----------
        trx_array: xr.DataArray
            raw trx array on the product level
        method: str
            aggregation method, sum or mean

        Returns
        ---------
        trx_array: xr.DataArray
            aggregated trx array on the store level, including item_qty, unit_price, discount_portion
        """
        if method == "sum":
            trx_array = trx_array.sum(dim="product_nbr", keepdims=True).rename(
                {"product_nbr": self.agg_level}
            )
            trx_array = trx_array.assign_coords({self.agg_level: [0]})
            return trx_array
        elif method == "mean":
            trx_array = trx_array.mean(dim="product_nbr", keepdims=True).rename(
                {"product_nbr": self.agg_level}
            )
            trx_array = trx_array.assign_coords({self.agg_level: [0]})
            return trx_array
        else:
            raise ValueError(f"method {method} not supported")


@dataclass
class CategoryFeatureGenerator(StoreFeatureGenerator):
    """Object to generate store-level features

    Attributes
    -----------
    product_category_mapping: pd.DataFrame
        product info data frame, with product_nbr as index and category_nbr as column
    agg_level: str
        aggregation level, default to be category_nbr
    features: dict
        dictionary of identical feature or derived features to generate
    """

    product_category_mapping: pd.DataFrame

    def __post_init__(self):
        self.loading_method = {
            "item_qty": "sum",
            "sales_amt": "sum",
            "unit_price": "mean",
            "discount_portion": "mean",
        }
        self.agg_level = "category_nbr"
        self.features = {
            feature_name: feature(self.agg_level, "week", feature_name)
            for feature_name, feature in self.features.items()
        }

    def _aggregate_raw_trx_array(self, trx_array, method="sum") -> xr.Dataset:
        """Aggregate the raw trx arrays to the category level

        Parameters
        ----------
        trx_array: xr.DataArray
            raw trx array on the product level
        method: str
            aggregation method, sum or mean

        Returns
        ---------
        trx_array: xr.DataArray
            aggregated trx array on the category level, including item_qty, unit_price, discount_portion
        """
        # trx_array: (week, customer, product)
        trx_array = trx_array.assign_coords(
            {
                self.agg_level: (
                    "product_nbr",
                    self.product_category_mapping.loc[
                        trx_array.coords["product_nbr"].values, self.agg_level
                    ],
                )
            }
        )
        if method == "sum":
            trx_array = trx_array.groupby(self.agg_level).sum()
            trx_array = trx_array.transpose("week", "customer_key", self.agg_level)
            return trx_array
        elif method == "mean":
            trx_array = trx_array.groupby(self.agg_level).mean()
            trx_array = trx_array.transpose("week", "customer_key", self.agg_level)
            return trx_array
        else:
            raise ValueError(f"method {method} not supported")
