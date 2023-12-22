import pandas as pd

from retailsynth.datasets.complete_journey.config import ProductAnnotation
from retailsynth.datasets.complete_journey.preprocess_utils import (
    detect_cats_with_small_sales,
    detect_inactive_products,
    detect_products_doubleml_test_on_price,
    detect_products_with_small_sales,
)


def annotate_products(
    products: pd.DataFrame, df: pd.DataFrame, preprocess_params: ProductAnnotation
):
    df = df.reset_index().join(products, on="product_nbr")
    """Annotate products with the number of transactions and the number of customers."""
    # create placeholder for pricing_policy_eligible flag
    product_selection_flag = pd.DataFrame(
        True, columns=["pricing_policy_eligible"], index=df.product_nbr.unique()
    )
    # run detections
    product_selection_flag = detect_inactive_products(
        df,
        product_selection_flag,
        preprocess_params.aggregation_level,
        preprocess_params.product_inactive_threshold,
    )
    product_selection_flag = detect_cats_with_small_sales(
        df,
        product_selection_flag,
        preprocess_params.min_num_txns,
    )
    product_selection_flag = detect_products_with_small_sales(
        df,
        product_selection_flag,
        preprocess_params.min_num_txns,
    )
    product_selection_flag = detect_products_doubleml_test_on_price(
        df,
        product_selection_flag,
        preprocess_params.product_price_std_threshold,
        preprocess_params.product_price_dependence_significance_level,
        preprocess_params.aggregation_level,
    )
    # merge the pricing_policy_eligible to the product table
    products = products.merge(
        product_selection_flag.fillna(False),
        left_on="product_nbr",
        right_index=True,
        how="inner",
    )

    return products
