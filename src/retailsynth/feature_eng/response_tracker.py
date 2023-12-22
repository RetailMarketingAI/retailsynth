from typing import Any, List, Optional

import numpy as np
import pandas as pd
import xarray as xr


class ResponseTracker:
    """This object records the customers (and products) that have positive decisions on the choice events for each step
    so that the multistage model will only proceed to the next step with selected customers and products at each time t.

    Attributes
    ----------
    time_column : str
        The name of the time column.
    customer_list : List[Any]
        The list of all customers.
    product_list : List[Any]
        The list of all products.

    selected_time: List[Any]
        The list of selected time to be used for the next step
    selected_customer: List[Any]
        The list of selected customers to be used for the next step
    selected_product: List[Any]
        The list of selected products to be used for the next step
    """

    def __init__(
        self,
        time_column: str,
        customer_list: List[Any],
        time_list: List[Any],
        product_table: pd.DataFrame,
    ):
        self.time_column = time_column
        self.customer_list = np.array(customer_list)
        self.product_list = np.array(product_table.product_nbr)
        self.category_list = np.array(sorted(product_table.category_nbr.unique()))
        self.time_list = np.array(time_list)
        self.product_table = product_table

        self.reset()

    def reset(self):
        """reset the object for new predictions"""
        self.selected_time = self.time_list
        self.selected_customer = self.customer_list
        self.selected_product = self.product_list

    def condition_on_store_visit(self, store_visit_result: xr.DataArray):
        # update selected time and selected customers based on store visit

        assert (
            store_visit_result.columns.values
            == [
                "week",
                "customer_key",
                "item_qty",
            ]
        ).all()

        category_df = pd.DataFrame({"category_nbr": self.category_list})

        choice_df = (
            store_visit_result.reset_index()
            .query("item_qty > 0")
            .merge(
                category_df,
                how="cross",
            )
        )

        self.selected_time = choice_df[self.time_column].values.tolist()
        self.selected_customer = choice_df["customer_key"].values.tolist()
        self.selected_categories = choice_df["category_nbr"].values.tolist()

    def condition_on_category_choice(
        self,
        category_choice_result: xr.DataArray,
        category_idx: Optional[int] = None,
    ):
        """
         Update selected time, customers, and products based on if customer
         purchased in the category using category choice result for a single input category with index category_idx

        Parameters
        ----------
            category_choice_result (xr.DataArray): shape=n_time * n_customer, 1
        """
        assert (
            category_choice_result.columns.values
            == [
                "week",
                "customer_key",
                "category_nbr",
                "item_qty",
            ]
        ).all()
        product_category_map = self.product_table.reset_index().loc[
            :, ["product_nbr", "category_nbr"]
        ]
        if category_idx is not None:
            category_choice_result = category_choice_result.query(
                f"category_nbr=={category_idx}"
            )

        choice_df = category_choice_result.query("item_qty > 0").merge(
            product_category_map, how="left", on="category_nbr"
        )

        self.selected_time = choice_df[self.time_column].values.tolist()
        self.selected_customer = choice_df["customer_key"].values.tolist()
        self.selected_product = choice_df["product_nbr"].values.tolist()

    def condition_on_product_choice(self, product_choice_result: xr.DataArray):
        assert (
            product_choice_result.columns.values
            == [
                "week",
                "customer_key",
                "product_nbr",
                "item_qty",
            ]
        ).all()
        self.product_table.reset_index().loc[:, ["product_nbr", "category_nbr"]]

        choice_df = product_choice_result.query("item_qty > 0")

        self.selected_time = choice_df[self.time_column].values.tolist()
        self.selected_customer = choice_df["customer_key"].values.tolist()
        self.selected_product = choice_df["product_nbr"].values.tolist()

    def get_selected_times(self) -> xr.DataArray:
        """return selected times to be used as a filter to extract sample of features

        Returns:
            xr.DataArray: the selected times
        """
        return xr.DataArray(self.selected_time)

    def get_selected_customers(self) -> xr.DataArray:
        """return selected customers to be used as a filter to extract sample of features

        Returns:
            xr.DataArray: the selected customers
        """
        return xr.DataArray(self.selected_customer)

    def get_selected_products(self) -> xr.DataArray:
        """return selected products to be used as a filter to extract sample of features

        Returns:
            xr.DataArray: the selected products
        """
        return xr.DataArray(self.selected_product)

    def get_filtering_index(self, agg_key="product_nbr"):
        if agg_key == "product_nbr":
            return pd.MultiIndex.from_frame(
                pd.DataFrame(
                    np.array(
                        [
                            self.selected_time,
                            self.selected_customer,
                            self.selected_product,
                        ]
                    ).T,
                    columns=[self.time_column, "customer_key", "product_nbr"],
                )
            )
        elif agg_key == "category_nbr":
            return pd.MultiIndex.from_frame(
                pd.DataFrame(
                    np.array(
                        [
                            self.selected_time,
                            self.selected_customer,
                            self.selected_categories,
                        ]
                    ).T,
                    columns=[self.time_column, "customer_key", "category_nbr"],
                )
            )
        else:
            raise ValueError(f"{agg_key} not supported for filtering")
