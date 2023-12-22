"""CompleteJourneyDataset class definition."""
from dataclasses import dataclass

from retailsynth.datasets.complete_journey.data_schemas import (
    CUSTOMER_SCHEMA,
    PRODUCT_SCHEMA,
    TRANSACTION_SCHEMA,
)
from retailsynth.datasets.complete_journey.dataloaders import load_dataset
from retailsynth.datasets.dataclass import BaseInputDataset, RetailDataSet


@dataclass
class CompleteJourneyDataset(BaseInputDataset):
    """CompleteJourneyDataset class."""

    def load_dataset(self) -> RetailDataSet:
        """
        Load the datasets using the provided configuration object.

        Returns
        -------
        dictionary of:
            customers : pd.DataFrame
                DataFrame containing customer data.

            products : pd.DataFrame
                DataFrame containing product data.

            transactions : pd.DataFrame
                DataFrame containing transaction data.
        """
        return load_dataset(self.cfg_raw_data)

    def rename_table_columns(self):
        """
        Rename the columns of the customers, products, and transactions DataFrames.

        Returns
        -------
        self : CompleteJourneyDataset
            Returns the updated instance of the class.
        """
        name_mapping = self.cfg_raw_data.colnames.__dict__  # value: key pair
        key_value_name_mapping = {
            origin_name: new_name for new_name, origin_name in name_mapping.items()
        }
        self.dataset.transactions.rename(columns=key_value_name_mapping, inplace=True)
        self.dataset.customers.rename(columns=key_value_name_mapping, inplace=True)
        self.dataset.products.rename(columns=key_value_name_mapping, inplace=True)
        return self

    def schema_validator(self):
        """
        Perform schema validation on the customers, products, and transactions DataFrames.

        Returns
        -------
        self : CompleteJourneyDataset
            Returns the updated instance of the class.
        """
        CUSTOMER_SCHEMA.validate(self.dataset.customers.reset_index())
        TRANSACTION_SCHEMA.validate(self.dataset.transactions.reset_index())
        PRODUCT_SCHEMA.validate(self.dataset.products.reset_index())
        return self
