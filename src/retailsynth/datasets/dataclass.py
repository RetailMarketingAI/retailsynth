"""Define the base dataset class."""
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from retailsynth.base_config import RawData


@dataclass
class RetailDataSet(object):
    """Define the dataset dictionary consisting of the customers, products, and transactions data frames."""

    customers: pd.DataFrame = None
    products: pd.DataFrame = None
    transactions: pd.DataFrame = None

    def get_sampled_dataset(self, n_customer: Optional[int] = None):
        """Get sample datasets for testing.

        Parameters
        ----------
        n_customer (int): number of customers to sample

        Returns
        -------
        BaseInputDataset
            The sampled dataset.
        """
        if n_customer is not None:
            self.customers = self.customers.sample(
                n_customer, random_state=0
            ).sort_index()
            self.transactions = self.transactions[
                self.transactions.index.get_level_values("customer_key").isin(
                    self.customers.index.unique()
                )
            ].sort_index()
            self.products = self.products.loc[
                self.transactions.index.get_level_values("product_nbr").unique()
            ].sort_index()

            logging.info("Sampling dataset")
            logging.info("Number of customers: {}".format(self.customers.shape[0]))
            logging.info("Number of products: {}".format(self.products.shape[0]))
            logging.info(
                "Number of transactions: {}".format(self.transactions.shape[0])
            )
        return self.customers, self.products, self.transactions


@dataclass
class BaseInputDataset(object):
    """
    BaseDataset class.

    This class provides a base class for the input dataset.

    Attributes
    ----------
        cfg_raw_data: The raw data configuration.
        dataset: The dataset dictionary consisting of the customers, products, and transactions data frames.

    Methods
    -------
        __post_init__(): This method is called after the object is initialized. It loads the data, renames the columns, casts the column types, and validates the schema.
        load_dataset(): This method loads the dataset from the raw data configuration.
        rename_table_columns(): This method renames the columns in the data frames.
        cast_column_type(): This method casts the column types in the data frames.
        schema_validator(): This method validates the schema of the data frames.
    """

    cfg_raw_data: RawData

    def __post_init__(self):
        """Is called after the object is initialized. It loads the data, renames the columns, casts the column types, and validates the schema."""
        self.dataset = self.load_dataset()
        self.rename_table_columns()
        self.cast_column_type()
        self.schema_validator()

    def load_dataset(self):
        """
        Load the dataset from the raw data configuration.

        Raises
        ------
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def rename_table_columns(self):
        """
        Rename the columns in the data frames.

        Raises
        ------
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def cast_column_type(self):
        """Casts the column types in the data frames."""
        pass

    def schema_validator(self):
        """
        Validate the schema of the data frames.

        Raises
        ------
            ValueError: If the schema is invalid.
        """
        raise NotImplementedError
