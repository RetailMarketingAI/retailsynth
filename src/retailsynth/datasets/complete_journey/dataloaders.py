"""Data loaders for the complete journey datasets."""
import logging
from pathlib import Path

import pandas as pd
from completejourney_py import get_data

from retailsynth import REPO_ROOT_DIR
from retailsynth.base_config import RawData
from retailsynth.datasets.dataclass import RetailDataSet


def read_raw_tables(
    paths: Path,
    customer_table_name: str,
    product_table_name: str,
    transaction_table_name: str,
) -> pd.DataFrame:
    """Read raw tables from the directory.

    Parameters
    ----------
    paths : Path
        The raw data path.
    customer_table_name : str
        Name of the customer table.
    product_table_name : str
        Name of the product table.
    transaction_table_name : str
        Name of the transaction table.

    Returns
    -------
    pd.DataFrame
        Customer table, product table, and transaction table read from the CSV.
    """
    logging.info(f"Reading raw tables from the directory {paths.resolve()}")
    customers = pd.read_csv(paths / customer_table_name)
    transactions = pd.read_csv(paths / transaction_table_name)
    products = pd.read_csv(paths / product_table_name)
    return customers, products, transactions


def download_complete_journey_dataset():
    """Use Python API to download the dunnhumby (complete journey) datasets.

    Returns
    -------
    customers : pd.DataFrame
        Cleaned customer demographic table.
    products : pd.DataFrame
        Cleaned product hierarchy tables.
    transactions : pd.DataFrame
        Cleaned transaction-like table.
    """
    # Use the API to download the complete journey datasets
    complete_dataset = get_data()
    customers = complete_dataset["demographics"]
    products = complete_dataset["products"]
    transactions = complete_dataset["transactions"]

    # Rename columns to be compatible with the rest of the code
    customers.rename(columns={"household_id": "customer_key"}, inplace=True)
    products.rename(
        columns={
            "product_id": "product_nbr",
            "product_category": "category_desc",
            "product_type": "subcategory_desc",
        },
        inplace=True,
    )
    transactions.rename(
        columns={
            "household_id": "customer_key",
            "store_id": "store_nbr",
            "product_id": "product_nbr",
            "quantity": "item_qty",
            "sales_value": "sales_amt",
        },
        inplace=True,
    )
    transactions[["coupon_disc", "coupon_match_disc", "retail_disc"]] = (
        -1 * transactions[["coupon_disc", "coupon_match_disc", "retail_disc"]]
    )

    # Add day column to the transaction table
    start_date = transactions.transaction_timestamp.min()
    transactions["day"] = (transactions.transaction_timestamp - start_date).dt.days + 1

    return customers, products, transactions


def load_dataset(cfg_raw_data: RawData) -> RetailDataSet:
    """Load datasets from local or using a Python API.

    Parameters
    ----------
    cfg_raw_data : RawData
        Configuration for raw data.

    Returns
    -------
    customers : pd.DataFrame
        Cleaned customer demographic table.
    products : pd.DataFrame
        Cleaned product hierarchy tables.
    transactions : pd.DataFrame
        Cleaned transaction-like table.
    """
    if cfg_raw_data.if_local_data:
        # Load data from local files
        root_dir = REPO_ROOT_DIR
        customers, products, transactions = read_raw_tables(
            Path(root_dir, cfg_raw_data.raw_data_path),
            cfg_raw_data.customer_table_name,
            cfg_raw_data.product_table_name,
            cfg_raw_data.transaction_table_name,
        )
    else:
        # Download the complete journey datasets using Python API
        customers, products, transactions = download_complete_journey_dataset()

    return RetailDataSet(customers, products, transactions)
