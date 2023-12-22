from dataclasses import dataclass

import pandas as pd
import pytest

from retailsynth.datasets.complete_journey.preprocess_pipeline import PreprocessPipeline


@pytest.fixture(scope="class")
def pipeline_class(sample_config):
    pipeline_class = PreprocessPipeline(
        raw_data_config=sample_config.raw_data,
    )
    return pipeline_class


def test_apply_all(pipeline_class):
    complete_journey_preprocess_pipeline = pipeline_class
    complete_journey_preprocess_pipeline.apply_all_transformations()
    assert complete_journey_preprocess_pipeline.dataset.customers.shape == (4, 7)
    assert complete_journey_preprocess_pipeline.dataset.products.shape == (4, 8)
    assert complete_journey_preprocess_pipeline.dataset.transactions.shape == (56, 4)


@dataclass
class MockDataset:
    products = pd.DataFrame(
        {
            "product_nbr": [1, 2, 3, 3],
            "category": ["A", "B", "C", "D"],
            "sub_category": ["A", "B", "C", "C"],
        }
    )
    transactions = pd.DataFrame(
        {
            "basket_id": [1, 2, 3, 4],
            "product_nbr": [1, 2, 3, 3],
            "customer_key": [1, 2, 3, 5],
            "sales_amt": [1, 2, -3, 4],
            "item_qty": [1, 2, 3, -4],
            "retail_disc": [1, 2, 3, 4],
            "coupon_disc": [1, 2, 3, 4],
            "coupon_match_disc": [1, 2, 3, 4],
            "week": [1, 2, 3, 4],
            "day": ["1", "2", "3", "4"],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_key": [1, 2, 3, 10],
            "name": ["John", "Jane", "Doe", "Smith"],
        }
    )


def test_drop_duplicate_product_id(pipeline_class):
    pipeline_class.dataset = MockDataset()

    transformed_dataset = pipeline_class.drop_duplicate_product_id().dataset

    assert (transformed_dataset.products["product_nbr"] == [0, 1, 2, 3]).all()
    assert (transformed_dataset.products["category"] == ["a", "b", "c", "d"]).all()
    assert (transformed_dataset.transactions["product_nbr"] == [0, 1, 2, 3, 2, 3]).all()
    assert (
        transformed_dataset.transactions["customer_key"] == [1, 2, 3, 3, 5, 5]
    ).all()
    assert (transformed_dataset.transactions["basket_id"] == [1, 2, 3, 3, 4, 4]).all()


def test_record_unrecognized_customer(pipeline_class):
    pipeline_class.dataset = MockDataset()

    transformed_dataset = pipeline_class.record_unrecognized_customer().dataset
    assert set(transformed_dataset.customers["customer_key"]) == set([1, 2, 3, 5, 10])


def test_add_pricing_columns(pipeline_class):
    pipeline_class.dataset = MockDataset()

    transformed_dataset = pipeline_class.add_pricing_columns()
    assert (
        transformed_dataset.dataset.transactions.columns
        == [
            "basket_id",
            "product_nbr",
            "customer_key",
            "week",
            "day",
            "item_qty",
            "sales_amt",
            "unit_price",
            "discount_portion",
        ]
    ).all()


def test_clean_transactions(pipeline_class):
    pipeline_class.dataset = MockDataset()

    transformed_dataset = (
        pipeline_class.add_pricing_columns().clean_transactions().dataset
    )
    assert transformed_dataset.transactions.shape == (4, 7)
    assert (
        transformed_dataset.transactions["customer_key"].values == [1, 2, 3, 5]
    ).all()


def test_aggregate_transactions(pipeline_class):
    pipeline_class.dataset = MockDataset()

    transformed_dataset = (
        pipeline_class.add_pricing_columns().aggregate_transactions().dataset
    )
    assert transformed_dataset.transactions.shape == (4, 7)
