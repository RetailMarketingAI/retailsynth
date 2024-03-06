"""Preprocess pipeline for the complete journey datasets."""
import logging

import numpy as np
import pandas as pd

from retailsynth.base_config import Config, RawData
from retailsynth.datasets.complete_journey.dataclass import CompleteJourneyDataset
from retailsynth.datasets.complete_journey.product_analysis import annotate_products
from retailsynth.datasets.dataclass import RetailDataSet


class TransformerMixin(object):
    """Mixin class for data transformation."""

    def __init__(self) -> None:
        """Initialize the transformer class."""
        self.cfg_raw_data: RawData
        self.dataset: RetailDataSet

    def filter_invalid_transactions(self):
        """Filter out transactions with negative quantity or negative sales amount."""
        # filter out rows with non-positive quantity sold
        self.dataset.transactions = self.dataset.transactions[
            (self.dataset.transactions["item_qty"] > 0)
            & (self.dataset.transactions["sales_amt"] > 0)
        ]

        logging.info(
            "Filter out transactions with non-positive quantity sold or money spent. Number of transactions are decreased to {}.".format(
                len(self.dataset.transactions)
            )
        )

        return self

    def drop_duplicate_product_id(self):
        """Assign same id to products with the same hierarchy information."""
        subset = list(self.dataset.products.columns.drop("product_nbr"))

        # standardize string values in datasets
        string_column = self.dataset.products.select_dtypes(include=["object"]).columns
        self.dataset.products[string_column] = self.dataset.products[
            string_column
        ].apply(lambda x: x.str.strip().str.lower().replace("\s+", " ", regex=True))

        products_no_duplicates = self.dataset.products.drop_duplicates(
            subset=subset
        ).drop(columns=["product_nbr"])
        products_no_duplicates.loc[:, "unique_product_id"] = np.arange(
            len(products_no_duplicates)
        )
        self.dataset.products = self.dataset.products.merge(
            products_no_duplicates, left_on=subset, right_on=subset, how="left"
        )

        # add helper column to be able to aggregate across all products
        self.dataset.products.loc[:, "all"] = "all"

        self.dataset.transactions = self.dataset.transactions.merge(
            self.dataset.products[["product_nbr", "unique_product_id"]],
            on="product_nbr",
        )

        self.dataset.transactions = self.dataset.transactions.drop(
            columns=["product_nbr"]
        )
        self.dataset.products = self.dataset.products.drop(columns=["product_nbr"])
        self.dataset.transactions.rename(
            columns={"unique_product_id": "product_nbr"}, inplace=True
        )
        self.dataset.products = self.dataset.products.rename(
            columns={"unique_product_id": "product_nbr"}
        ).drop_duplicates()
        self.dataset.products = self.dataset.products.fillna("None")
        self.dataset.products.sort_values("product_nbr", inplace=True)

        logging.info(
            "Use the same label for products with the same hierarchy information. Number of products are decreased to {}.".format(
                len(products_no_duplicates)
            )
        )
        return self

    def add_category_id(self):
        """Assign category id to products with the same category desc."""
        categories = self.dataset.products.category_desc.unique()
        categories = pd.DataFrame(
            np.vstack((categories, np.arange(len(categories)))).T,
            columns=["category_desc", "category_nbr"],
        ).astype({"category_nbr": int})
        self.dataset.products = self.dataset.products.join(
            categories.set_index("category_desc"), on="category_desc"
        )

        logging.info("Added category numbers.")
        return self

    def record_unrecognized_customer(self):
        """Record unrecognized customer ids.

        Parameters
        ----------
            customers (pd.DataFrame): customer demographic table
            transactions (pd.DataFrame): transaction table

        Returns
        -------
            customers (pd.DataFrame): customer demographic table
        """
        unrecognized_customer = self.dataset.transactions.loc[
            ~self.dataset.transactions["customer_key"].isin(
                self.dataset.customers["customer_key"].unique()
            )
        ]["customer_key"].unique()
        if len(unrecognized_customer) > 0:
            logging.info(
                "There are unrecognized customer ids in the transactions table. Number of unrecognized customer ids: {}".format(
                    len(unrecognized_customer)
                )
            )

            self.dataset.customers = pd.concat(
                [
                    self.dataset.customers,
                    pd.DataFrame(
                        unrecognized_customer, columns=["customer_key"]
                    ).drop_duplicates(),
                ]
            ).reset_index(drop=True)
        self.dataset.customers = self.dataset.customers.fillna("None").sort_values(
            "customer_key"
        )
        return self

    def add_pricing_columns(self):
        """Read the raw transaction datasets.

        Add date column to prepare the prophet demand model fitting.

        Parameters
        ----------
        raw_transaction
            the base raw transaction that current object deals with

        Returns
        -------
        raw_transaction(processed)
            resulting dataframe contains two extra columns: unit price and discount portion
        """
        # use dealt price / quantity to get unit price
        self.dataset.transactions["unit_price"] = (
            self.dataset.transactions["sales_amt"]
            / self.dataset.transactions["item_qty"]
        )
        # compute shelf price, and use unit_price / shelf_price to get the discount portion
        base_price = (
            self.dataset.transactions["sales_amt"]
            - self.dataset.transactions[["coupon_match_disc", "retail_disc"]].sum(
                axis=1
            )
        ) / self.dataset.transactions["item_qty"]
        retail_discount_portion = (
            1 - self.dataset.transactions["unit_price"] / base_price
        )
        manufacturer_discount_portion = np.clip(
            (
                -self.dataset.transactions["coupon_disc"]
                / (self.dataset.transactions["item_qty"] * base_price)
            ),
            a_min=0,
            a_max=1,
        )
        self.dataset.transactions["discount_portion"] = np.clip(
            retail_discount_portion + manufacturer_discount_portion,
            a_min=0,
            a_max=1,
        )
        self.dataset.transactions = (
            self.dataset.transactions.groupby(
                ["basket_id", "product_nbr", "customer_key", "week", "day"]
            )
            .agg(
                {
                    "item_qty": "sum",
                    "sales_amt": "sum",
                    "unit_price": "mean",
                    "discount_portion": "mean",
                }
            )
            .reset_index()
        )
        # day, customer_id, product id, sales_value, quantity sold, unit price, discount_portion
        return self

    def clean_transactions(self):
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
        self.dataset.customers = self.dataset.customers[
            self.dataset.customers.customer_key.isin(
                self.dataset.transactions.customer_key.unique()
            )
        ]
        self.dataset.products = self.dataset.products[
            self.dataset.products.product_nbr.isin(
                self.dataset.transactions.product_nbr.unique()
            )
        ]
        column_in_use = [
            self.cfg_raw_data.preprocess.aggregation_level,
            "customer_key",
            "product_nbr",
            "item_qty",
            "sales_amt",
            "unit_price",
            "discount_portion",
        ]
        self.dataset.transactions = self.dataset.transactions[column_in_use]
        return self

    def aggregate_transactions(
        self,
        product_aggregation_level: str = "product_nbr",
        if_reset_index: bool = True,
    ):
        """Aggregate the transaction datasets by time, customer, and product.

        Parameters
        ----------
            product_aggregation_level (str): product aggregation level, can be "product_nbr", "category_nbr", etc.
            if_reset_index (boolean): whether to reset the index of outcome dataframe

        Returns
        -------
            pd.DataFrame: aggregated transaction datasets on daily or weekly level
        """
        assert product_aggregation_level in self.dataset.transactions.columns
        time_column = (
            "week"
            if self.cfg_raw_data.preprocess.aggregation_level == "week"
            else "day"
        )
        self.dataset.transactions = (
            self.dataset.transactions.groupby(
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

        if if_reset_index:
            self.dataset.transactions = self.dataset.transactions.reset_index()

        return self

    def drop_gasoline_related_products(self):
        """Drop products that are related to fuel."""
        self.dataset.products = self.dataset.products[
            ~self.dataset.products.subcategory_desc.str.contains("gasoline")
        ]
        self.dataset.transactions = self.dataset.transactions[
            self.dataset.transactions.product_nbr.isin(
                self.dataset.products.product_nbr
            )
        ]
        self.dataset.customers = self.dataset.customers[
            self.dataset.customers.customer_key.isin(
                self.dataset.transactions.customer_key.unique()
            )
        ]
        logging.info(
            "Drop products that are related to fuel. Number of products are decreased to {}.".format(
                len(self.dataset.products)
            )
        )
        return self

    def apply_all_transformations(self):
        """Apply all transformations in the pipeline to the datasets.

        This method applies all the transformation methods from the transformer class to the datasets.

        Returns
        -------
        BaseInputDataset
            The transformed datasets.
        """
        (
            self.filter_invalid_transactions()
            .drop_duplicate_product_id()
            .add_category_id()
            .record_unrecognized_customer()
            .add_pricing_columns()
            .clean_transactions()
            .aggregate_transactions()
        )
        if self.cfg_raw_data.preprocess.remove_gasoline_product:
            self.drop_gasoline_related_products()
        self.dataset.customers = self.dataset.customers.set_index(
            "customer_key"
        ).sort_index()

        self.dataset.products = self.dataset.products.set_index(
            "product_nbr"
        ).sort_index()

        self.dataset.transactions = self.dataset.transactions.set_index(
            [
                self.cfg_raw_data.preprocess.aggregation_level,
                "customer_key",
                "product_nbr",
            ]
        ).sort_index()


class PreprocessPipeline(CompleteJourneyDataset, TransformerMixin):  # type: ignore
    def __init__(self, raw_data_config: RawData) -> None:
        super().__init__(raw_data_config)


def run_preprocess(config: Config):
    """Run the preprocess pipeline.

    This function applies the preprocess pipeline to the raw data based on the provided configuration.

    Parameters
    ----------
    config : Config
        Configuration for the whole run.

    Returns
    -------
    tuple
        A tuple containing the preprocessed customers, products, and transactions dataframes.
    """
    logging.info("Start preprocess pipeline")

    complete_journey_preprocess_pipeline = PreprocessPipeline(
        raw_data_config=config.raw_data,  # type: ignore
    )
    complete_journey_preprocess_pipeline.apply_all_transformations()

    logging.info("Preprocess pipeline finished")
    logging.info(
        "Number of customers: {}".format(
            complete_journey_preprocess_pipeline.dataset.customers.shape[0]  # type: ignore
        )
    )
    logging.info(
        "Number of products: {}".format(
            complete_journey_preprocess_pipeline.dataset.products.shape[0]  # type: ignore
        )
    )
    logging.info(
        "Number of transactions: {}".format(
            complete_journey_preprocess_pipeline.dataset.transactions.shape[0]  # type: ignore
        )
    )

    logging.info("Annotate products")
    if config.raw_data.product_annotation.include_annotation:  # type: ignore
        complete_journey_preprocess_pipeline.dataset.products = annotate_products(
            complete_journey_preprocess_pipeline.dataset.products,
            complete_journey_preprocess_pipeline.dataset.transactions,
            config.raw_data.product_annotation,  # type: ignore
        )
    return complete_journey_preprocess_pipeline.dataset.get_sampled_dataset(
        config.n_customers_sampled
    )
