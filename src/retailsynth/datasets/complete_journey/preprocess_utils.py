"""Utility functions for preprocessing the data."""
import logging
import os
from pathlib import Path

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml import DoubleMLData
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from retailsynth.datasets.complete_journey.config import Preprocess


def detect_inactive_products(
    transactions: pd.DataFrame,
    product_selection_flag: pd.DataFrame,
    aggregation_level: str,
    product_inactive_threshold: int,
) -> pd.DataFrame:
    """Filter out products that have been inactive for a certain period of time.

    Parameters
    ----------
        transactions (pd.DataFrame): transaction table
        product_selection_flag (pd.DataFrame): product selection flag table, with column of `pricing_policy_eligible`
        aggregation_level (str): the level of aggregation of the transaction table, can be either 'day' or 'week'
        product_inactive_threshold (int): the number of days or weeks that a product has been inactive

    Returns
    -------
        pd.DataFrame: product selection flag table with column of `pricing_policy_eligible` updated
    """
    df = get_remaining_transaction_according_pricing_eligible_flag_df(
        product_selection_flag, transactions
    )
    end_trx_time = transactions[aggregation_level].max()
    inactive_period = end_trx_time - df.groupby("product_nbr")[aggregation_level].max()
    filtered_product_list = inactive_period[
        inactive_period > product_inactive_threshold
    ].index
    product_selection_flag = update_pricing_eligible_flag(
        product_selection_flag,
        inactive_period.index,
        filtered_product_list,
        "product_active_flag",
    )
    logging.info(
        "Flag products that have been inactive for {} {}s. {} products pass the filter.".format(
            product_inactive_threshold,
            aggregation_level,
            product_selection_flag[
                product_selection_flag.pricing_policy_eligible
            ].shape[0],
        )
    )
    return product_selection_flag


def update_pricing_eligible_flag(
    flag_df: pd.DataFrame,
    original_list: pd.Index,
    result_list: pd.Index,
    flag_name: str,
):
    """Update the pricing eligible flag for products that are not eligible for price adjustment.

    Parameters
    ----------
        flag_df (pd.DataFrame): pricing eligible flag table with product_nbr as index, with columns of each filtering flag, and final result

    Returns
    -------
        pd.DataFrame: updated pricing eligible flag table
    """
    flag_df.loc[original_list, flag_name] = True
    flag_df.loc[result_list, flag_name] = False
    flag_df = flag_df.fillna(False)
    flag_df.loc[:, "pricing_policy_eligible"] = flag_df.loc[
        :, flag_df.columns != "pricing_policy_eligible"
    ].all(axis=1)
    return flag_df


def get_remaining_transaction_according_pricing_eligible_flag_df(
    flag_df: pd.DataFrame, transactions: pd.DataFrame
):
    """
    get the product list that are eligible for price adjustment.

    Parameters
    ----------
        flag_df (pd.DataFrame): pricing eligible flag table, with columns of each filtering flag, and final result
        transactions (pd.DataFrame): transaction table
    Returns:
        pd.DataFrame: remaining transactions
    """
    selected_products = flag_df[flag_df.pricing_policy_eligible].index
    return transactions[transactions.product_nbr.isin(selected_products)]


def detect_products_with_small_sales(
    transactions: pd.DataFrame,
    product_selection_flag: pd.DataFrame,
    small_sales_threshold: float,
) -> pd.DataFrame:
    """Compute the sales of each product for each time step, and filter out products with 25% quantile sales below a certain threshold.

    Parameters
    ----------
        transactions (pd.DataFrame): transaction datasets
        product_selection_flag (pd.DataFrame): pricing eligible flag table, with columns of each filtering flag, and final flag for pricing policy eligibility
        small_sales_threshold (float): the threshold of sales below which a product is considered to have small sales

    Returns
    -------
        pd.DataFrame: updated product_selection_flag table, with small sales products being False
    """
    df = get_remaining_transaction_according_pricing_eligible_flag_df(
        product_selection_flag, transactions
    )
    prod_txns_count = df.groupby(["product_nbr"])["item_qty"].count()
    filtered_product_list = prod_txns_count[
        prod_txns_count < small_sales_threshold
    ].index
    product_selection_flag = update_pricing_eligible_flag(
        product_selection_flag,
        prod_txns_count.index,
        filtered_product_list,
        "product_sufficient_sales_flag",
    )
    logging.info(
        "Flag products with number of txns below {}. {} products pass the filter.".format(
            small_sales_threshold,
            product_selection_flag[
                product_selection_flag.pricing_policy_eligible
            ].shape[0],
        )
    )
    return product_selection_flag


def detect_cats_with_small_sales(
    transactions: pd.DataFrame,
    product_selection_flag: pd.DataFrame,
    small_sales_threshold: float,
) -> pd.DataFrame:
    """Compute the sales of each product for each time step, and filter out products with 25% quantile sales below a certain threshold.

    Parameters
    ----------
        transactions (pd.DataFrame): transaction datasets
        product_selection_flag (pd.DataFrame): pricing eligible flag table, with columns of each filtering flag, and final flag for pricing policy eligibility
        small_sales_threshold (float): the threshold of sales below which a product is considered to have small sales

    Returns
    -------
        pd.DataFrame: updated product_selection_flag table, with small sales products being False
    """
    df = get_remaining_transaction_according_pricing_eligible_flag_df(
        product_selection_flag, transactions
    )
    cat_txns_count = df.groupby(["category_nbr"])["item_qty"].count()
    filtered_cat_list = cat_txns_count[cat_txns_count < small_sales_threshold].index
    filtered_product_list = df[
        df.category_nbr.isin(filtered_cat_list)
    ].product_nbr.unique()
    product_selection_flag = update_pricing_eligible_flag(
        product_selection_flag,
        df.product_nbr.unique(),
        filtered_product_list,
        "category_sufficient_sales_flag",
    )
    logging.info(
        "Flag products with number of category txns below {}. {} products pass the filter.".format(
            small_sales_threshold,
            product_selection_flag[
                product_selection_flag.pricing_policy_eligible
            ].shape[0],
        )
    )
    return product_selection_flag


def detect_products_with_no_price_variation(
    transactions: pd.DataFrame, std_epsilon: float = 1e-6
) -> pd.DataFrame:
    """Filter out products that have no price variation.

    Parameters
    ----------
        transactions (pd.DataFrame): aggregated transaction datasets

    Returns
    -------
        pd.DataFrame: filtered transaction datasets
    """
    standard_deviation = transactions.groupby("product_nbr")["unit_price"].std()
    selected_product_list = standard_deviation[standard_deviation > std_epsilon].index
    return transactions[transactions.product_nbr.isin(selected_product_list)]


def doubleml_test_on_price(df: pd.DataFrame, product_nbr: str) -> tuple:
    """Calculate the p-value of the doubleml test on unit_price.

    Parameters
    ----------
        df (pd.DataFrame): the input dataframe with unit_price column and item_qty column
        product_nbr (str): product id

    Returns
    -------
        tuple: coefficient and p-value of the double ml test
    """
    try:
        # customer, product, time, item_qty, unit_price
        unit_price_level_df = df[df["product_nbr"] == product_nbr]
        data = DoubleMLData(
            data=unit_price_level_df,
            y_col="item_qty",
            d_cols="unit_price",
            x_cols=["customer_visit", "discount_portion"],
        )
        learner = LinearRegression()
        ml_g = learner
        ml_m = learner
        dml_plr_obj = dml.DoubleMLPLR(data, ml_g, ml_m)
        summary = dml_plr_obj.fit().summary
        return summary["coef"].unit_price, summary["P>|t|"].unit_price
    except ValueError:
        logging.debug(
            f"product {product_nbr} does not have enough data points across all price levels"
        )
        return np.nan, np.nan


def detect_products_doubleml_test_on_price(
    transactions: pd.DataFrame,
    product_selection_flag: pd.DataFrame,
    price_std_threshold: float = 1e-6,
    alpha: float = 0.05,
    aggregation_level: str = "week",
) -> pd.DataFrame:
    """Filter out products that do not have significant difference in sales under different prices.

    Parameters
    ----------
        transactions (pd.DataFrame): transaction datasets
        product_selection_flag (pd.DataFrame): pricing eligible flag table, with columns of each filtering flag, and final flag for pricing policy eligibility
        price_std_threshold (float): the threshold of minimum standard deviation of price distribution
        alpha (float): the significance level of double ml test

    Returns
    -------
        pd.DataFrame: updated product_selection_flag table, with products that do not pass the double ml test being False
    """
    df = get_remaining_transaction_according_pricing_eligible_flag_df(
        product_selection_flag, transactions
    )
    df = detect_products_with_no_price_variation(df, price_std_threshold)
    agg_df = (
        df.groupby([aggregation_level, "product_nbr"])
        .agg({"item_qty": "sum", "unit_price": "mean", "discount_portion": "mean"})
        .reset_index()
    )
    customer_visit = (
        df.groupby(aggregation_level)["customer_key"]
        .nunique()
        .to_frame(name="customer_visit")
    )
    agg_df = agg_df.merge(customer_visit, left_on=aggregation_level, right_index=True)
    result = []
    for product_nbr in tqdm(
        agg_df.product_nbr.unique(), desc="Evaluate pricing effect of product"
    ):
        coef, p_value = doubleml_test_on_price(agg_df, product_nbr)
        result.append((product_nbr, coef, p_value, (p_value < alpha) & (coef < 0)))
    test_df = pd.DataFrame(
        result, columns=["product_nbr", "coef", "p_value", "significant"]
    )
    filtered_product_list = test_df[~test_df.significant].loc[:, "product_nbr"]

    product_selection_flag = update_pricing_eligible_flag(
        product_selection_flag,
        test_df.product_nbr,
        filtered_product_list,
        "product_price_variation_flag",
    )
    logging.info(
        "Flag products that prices have no significant effect on sales. {} products pass the filter.".format(
            product_selection_flag[
                product_selection_flag.pricing_policy_eligible
            ].shape[0]
        )
    )
    return product_selection_flag


def clean_transactions(
    df: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    preprocess_params: Preprocess,
) -> pd.DataFrame:
    """Filter out the raw transaction based on a set of criterion defined in the class.

    Parameters
    ----------
    df(pd.DataFrame):
        the input dataframe we want to filter on
        columns: day, customer_id, product id, sales_value, quantity sold, dealt price, discount_portion
    customers (pd.DataFrame): customer demographic table
    products (pd.DataFrame): product hierarchy table
    preprocess_params: Preprocess
        the preprocess parameters defined in the config file

    Returns
    -------
    df(processed)
        the output dataframe has the exact same columns as the input dataframe
        but with less rows because we filter out some records based on the criterion defined
    customers(processed): pd.DataFrame
        the output customer demographic table, with only customers shown in the filtered transaction table
    products(processed): pd.DataFrame
        the output product hierarchy table, with only products shown in the filtered transaction table
    """
    # remove unused records from product and customer table
    customers = customers[customers.customer_key.isin(df.customer_key.unique())]
    products = products[products.product_nbr.isin(df.product_nbr.unique())]
    column_in_use = [
        preprocess_params.aggregation_level,
        "customer_key",
        "product_nbr",
        "item_qty",
        "sales_amt",
        "unit_price",
        "discount_portion",
    ]
    return df[column_in_use], customers, products


def aggregate_transactions(
    df: pd.DataFrame,
    time_aggregation_level: str,
    product_aggregation_level: str = "product_nbr",
    if_reset_index: bool = True,
) -> pd.DataFrame:
    """Aggregate the transaction datasets from individual level to segment level.

    Parameters
    ----------
        df (pd.DataFrame): raw transaction datasets
        time_aggregation_level (str): time aggregation level, can be "day" or "week"
        product_aggregation_level (str): product aggregation level, can be "product_nbr", "subcategory_desc", etc.
        if_reset_index (boolean): whether to reset the index of outcome dataframe

    Returns
    -------
        pd.DataFrame: aggregated transaction datasets on daily or weekly level
    """
    assert product_aggregation_level in df.columns
    time_column = "week" if time_aggregation_level == "week" else "day"
    result = (
        df.groupby(
            [
                time_column,
                "customer_key",
                product_aggregation_level,
            ]
        )
        .agg(
            {
                "item_qty": "sum",
                "sales_amt": "sum",
                "unit_price": "mean",
                "discount_portion": "mean",
            }
        )
        .sort_index()
    )

    return result.reset_index() if if_reset_index else result


def save_processed_dfs(
    path: str,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    transactions: pd.DataFrame,
):
    """store processed parquet files locally

    Parameters
    ----------
        path (str): path to store the files
        customers (pd.DataFrame): customer demographic table
        products (pd.DataFrame): product hierarchy table
        transactions (pd.DataFrame): transaction table
    """
    if not os.path.exists(path):
        os.makedirs(path)
    customers.to_parquet(Path(path, "customers.parquet"))
    products.to_parquet(Path(path, "products.parquet"))
    transactions.to_parquet(Path(path, "transactions.parquet"))
