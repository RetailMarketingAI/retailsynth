import pytest

from retailsynth.datasets.complete_journey.data_schemas import (
    CUSTOMER_SCHEMA,
    PRODUCT_SCHEMA,
    TRANSACTION_SCHEMA,
)
from retailsynth.datasets.complete_journey.dataclass import CompleteJourneyDataset


@pytest.fixture(scope="class")
def complete_journey_dataset(sample_config):
    return CompleteJourneyDataset(sample_config.raw_data)


def test_init_dataset(complete_journey_dataset):
    # Assert that the customers, products, and transactions attributes have been updated
    assert complete_journey_dataset.dataset.customers is not None
    assert complete_journey_dataset.dataset.products is not None
    assert complete_journey_dataset.dataset.transactions is not None


def test_load_dataset(complete_journey_dataset):
    assert complete_journey_dataset.dataset.customers.shape == (4, 8)
    assert complete_journey_dataset.dataset.products.shape == (4, 7)
    assert complete_journey_dataset.dataset.transactions.shape == (58, 15)


def test_schema_validator(complete_journey_dataset):
    CUSTOMER_SCHEMA.validate(complete_journey_dataset.dataset.customers)
    PRODUCT_SCHEMA.validate(complete_journey_dataset.dataset.products)
    TRANSACTION_SCHEMA.validate(complete_journey_dataset.dataset.transactions)
