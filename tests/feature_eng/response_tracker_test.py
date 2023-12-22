import numpy as np
import pandas as pd
import xarray as xr


def test_reset(tracker):
    tracker.selected_time = [1, 2]
    tracker.reset()
    assert (tracker.selected_time == [1, 2, 3, 4, 5, 6, 7, 8]).all()


def test_get_selected_times(tracker):
    tracker.selected_time = [1, 2]
    selected_times = tracker.get_selected_times()
    assert isinstance(selected_times, xr.DataArray)
    assert (selected_times.values == np.array([1, 2])).all()


def test_condition_on_store_visit(tracker):
    tracker.reset()
    store_visit_result = (
        xr.DataArray(
            np.array([[1, 0], [1, 1]]),
            dims=["week", "customer_key"],
            coords={"week": [1, 2], "customer_key": [1001, 1002]},
            name="item_qty",
        )
        .to_dataframe()
        .reset_index()
    )

    tracker.condition_on_store_visit(store_visit_result)
    assert tracker.selected_time == [1] * 2 + [2] * 2 * 2

    assert tracker.selected_customer == [1001] * 2 + [1001] * 2 + [1002] * 2
    assert tracker.selected_categories == [0, 1] * 3


def test_condition_on_category_choice(tracker):
    cat_choice_result = (
        xr.DataArray(
            np.array(
                [
                    [
                        [1, 1],
                    ],
                    [
                        [0, 1],
                    ],
                ]
            ),
            dims=["week", "customer_key", "category_nbr"],
            coords={
                "week": [1, 2],
                "customer_key": [
                    1001,
                ],
                "category_nbr": [0, 1],
            },
            name="item_qty",
        )
        .to_dataframe()
        .reset_index()
    )

    tracker.condition_on_category_choice(cat_choice_result, 0)
    assert tracker.selected_time == [1, 1, 1]
    assert tracker.selected_customer == [1001, 1001, 1001]
    assert tracker.selected_product == [0, 2, 4]

    exp_time = [
        1,
        1,
        1,
        1,
        1,
        2,
        2,
    ]
    tracker.condition_on_category_choice(cat_choice_result, None)
    assert tracker.selected_time == exp_time

    exp_customer = [1001, 1001, 1001, 1001, 1001, 1001, 1001]
    assert tracker.selected_customer == exp_customer

    exp_product = [0, 2, 4, 1, 3, 1, 3]
    assert tracker.selected_product == exp_product

    index = tracker.get_filtering_index()
    expected_indx = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "week": exp_time,
                "customer_key": exp_customer,
                "product_nbr": exp_product,
            }
        )
    )
    assert index.equals(expected_indx)

    tracker.condition_on_category_choice(cat_choice_result)

    index = tracker.get_filtering_index()
    pd.testing.assert_index_equal(index, expected_indx, check_order=False)


def test_condition_on_product_choice(tracker):
    tracker.reset()

    product_choice_result = xr.DataArray(
        np.array([[[1, 0], [0, 1]], [[1, 0], [0, 0]]]),
        dims=["week", "customer_key", "product_nbr"],
        coords={"week": [1, 2], "customer_key": [1001, 1002], "product_nbr": [4, 1]},
        name="item_qty",
    )

    tracker.condition_on_product_choice(
        product_choice_result.to_dataframe().reset_index()
    )
    assert tracker.selected_time == [
        1,
        1,
        2,
    ]
    assert tracker.selected_customer == [
        1001,
        1002,
        1001,
    ]
    assert tracker.selected_product == [
        4,
        1,
        4,
    ]


def test_get_selected_customers(tracker):
    tracker.selected_customer = [1001, 1002]
    selected_customers = tracker.get_selected_customers()
    assert isinstance(selected_customers, xr.DataArray)
    assert (selected_customers.values == np.array([1001, 1002])).all()


def test_get_selected_products(tracker):
    tracker.selected_product = [0, 1]
    selected_products = tracker.get_selected_products()
    assert isinstance(selected_products, xr.DataArray)
    assert (selected_products.values == np.array([0, 1])).all()
