"""Utility functions for feature generation."""
import itertools
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from retailsynth.feature_eng.features import BaseFeature
from retailsynth.feature_eng.indices import IndexMap

# mypy: ignore-errors


class IndexNameMapping(Enum):
    """record a list of all supported indices."""

    week = "time_index"
    day = "day_index"
    customer_key = "customer_index"
    product_nbr = "item_index"
    category_nbr = "item_index"
    subcategory_desc = "item_index"
    all = "item_index"


def generate_index_mapping(
    trx: pd.DataFrame,
    time_list: Optional[List[int]] = None,
    customer_list: Optional[List] = None,
    item_list: Optional[List] = None,
    time_name: Optional[str] = "week",
    customer_name: Optional[str] = "customer_key",
    item_name: Optional[str] = "product_nbr",
):
    """Given a pandas data frame, generate a integer index mapping for index column.

    Parameters
    ----------
        trx (pd.DataFrame): input data frame with index being either pd.Index or pd.MultiIndex
        time_coords (Optional[List[int]]): list of unique time values
        customer_coords (Optional[List]): list of unique customer values
        item_coords (Optional[List]): list of unique item values
        time_name (Optional[str]): name of time index
        customer_name (Optional[str]): name of customer index
        item_name (Optional[str]): name of item index

    Raises
    ------
        NotImplementedError: if the index of pandas df is not pd.Index, could not generate index mapping

    Returns
    -------
        dict: dictionary of IndexMap objects, one for each level of index
    """
    if isinstance(trx.index, pd.MultiIndex):
        assert time_list is not None
        assert customer_list is not None
        assert item_list is not None
        indices = {
            "time_index": IndexMap(time_list, time_name),
            "customer_index": IndexMap(customer_list, customer_name),
            "item_index": IndexMap(item_list, item_name),
        }
    elif isinstance(trx.index, pd.Index):
        indices = {
            IndexNameMapping[trx.index.name].value: IndexMap(
                trx.index.values, trx.index.name
            )
        }
    else:
        raise NotImplementedError(f"Unknown index type: {type(trx.index)}")
    return indices


def _extract_index(
    trx: pd.DataFrame, indices: Dict[str, IndexMap], index_name: str
) -> np.ndarray:
    """Extract index from transaction data frame."""
    column_name = indices[index_name].name
    column_value = list(trx.index.get_level_values(column_name).values)
    index = indices[index_name].get_index(column_value)
    return index


def convert_trx_df_to_numpy(
    trx: pd.DataFrame,
    target_column: str,
    indices: Dict[str, IndexMap],
) -> np.ndarray:
    # used for item_qty, sales_amt
    # create placeholder for the transaction array
    shape = (
        len(indices["time_index"]),
        len(indices["customer_index"]),
        len(indices["item_index"]),
    )
    trx_array = np.zeros(shape, dtype=np.float32)

    # get index to put target value for each row from the transaction
    idx = (
        _extract_index(trx, indices, "time_index"),
        _extract_index(trx, indices, "customer_index"),
        _extract_index(trx, indices, "item_index"),
    )
    # assign values to transaction array
    trx_array[idx] = trx[target_column].astype("float32").values

    return trx_array


def convert_trx_df_to_numpy_fill_with_mean(
    trx: pd.DataFrame,
    target_column: str,
    indices: Dict[str, IndexMap],
) -> np.ndarray:
    """Convert a specific column of df to numpy array, and add negative data.

    Parameters
    ----------
        trx (pd.DataFrame): transaction data with (time, customer_key, product_nbr) being the index
        target_column (str): interested column from the transaction data frame
        indices (Dict[str, IndexMap]): dictionary of IndexMap objects, one for each level of index

    Returns
    -------
        np.ndarray: transaction array of target column in a shape of (n_time_step, n_customer, n_item)
    """
    # create placeholder for the transaction array
    shape = (
        len(indices["time_index"]),
        len(indices["customer_index"]),
        len(indices["item_index"]),
    )
    trx_array = np.zeros(shape, dtype=np.float32)
    trx_array[:] = np.nan

    # get index to put target value for each row from the transaction
    idx = (
        _extract_index(trx, indices, "time_index"),
        _extract_index(trx, indices, "customer_index"),
        _extract_index(trx, indices, "item_index"),
    )
    # assign values to transaction array
    trx_array[idx] = trx[target_column].values

    # create a data frame to record mean of target column
    fill_values = pd.DataFrame(
        itertools.product(indices["time_index"].values, indices["item_index"].values),
        columns=["week", indices["item_index"].name],
    )
    # get mean value within a week
    product_mean_each_week = (
        trx.reset_index()
        .groupby(["week", indices["item_index"].name])[target_column]
        .mean()
    )
    fill_values = fill_values.merge(
        product_mean_each_week,
        left_on=["week", indices["item_index"].name],
        right_index=True,
        how="left",
    )

    # if there're still nan values, fill the nan with global mean
    product_mean_global = (
        trx.reset_index().groupby(indices["item_index"].name)[target_column].mean()
    )
    fill_values = fill_values.merge(
        product_mean_global,
        left_on=indices["item_index"].name,
        right_index=True,
        how="left",
        suffixes=("", "_global"),
    )
    fill_values[target_column] = fill_values[target_column].fillna(
        fill_values[f"{target_column}_global"]
    )
    fill_values = (
        fill_values.pivot(
            index="week", columns=indices["item_index"].name, values=target_column
        )
        .astype("float32")
        .values
    )

    # actual filling step for the trx_array
    fill_values = np.expand_dims(fill_values, axis=1)
    trx_array = np.where(np.isnan(trx_array), fill_values, trx_array)

    return trx_array


def map_feature_name_to_object(feature_name: str, feature_catalog: Enum) -> Any:
    """Convert feature name to actual feature to create a callable object.

    Parameters
    ----------
        feature_name (str): name of a feature object
        feature_catalog (Enum, optional): a list of all supported features. Defaults to FeatureCatalog.

    Raises
    ------
        KeyError: if input feature is not defined in the feature catalog

    Returns
    -------
        BaseFeature: a feature object
    """
    map = feature_catalog[feature_name].value
    assert type(map) == type(BaseFeature)
    return map


def generate_product_hierarchy_indicator_matrix(
    products: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    """Generate a indicator matrix with each row representing a product, and each column representing a unique value from the target column.

        Each value in this matrix represents if the corresponding product belongs to the corresponding group of target columns.
        For example, when target_column is "category_desc", this matrix tells us
        if product A belongs to a category with 1 representing yes and 0 representing no.

    Parameters
    ----------
        products (pd.DataFrame): product hierarchy table in a pandas data frame
        target_column (str): column to create the indicator matrix for

    Returns
    -------
       pd.DataFrame: indicator matrix with each row representing a product,
                     and each column one-hot-encoded from the target column
    """
    assert products.index.name == "product_nbr"
    # one-hot-encode the corresponding column
    indicator_matrix = pd.get_dummies(products[target_column])
    if not (sorted(indicator_matrix.index) == indicator_matrix.index).all():
        raise ValueError(
            f"The index of the indicator matrix is not sorted for {target_column} in {products.head()}."
        )
    return indicator_matrix
