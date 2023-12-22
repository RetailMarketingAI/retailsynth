from retailsynth.feature_eng.feature_generator import (
    CategoryFeatureGenerator,
    StoreFeatureGenerator,
)
from retailsynth.feature_eng.feature_utils import map_feature_name_to_object
from retailsynth.feature_eng.features import FeatureCatalog


def test_store_feature_loading_raw_array(
    sample_config, raw_sample_tables, serialized_trx_arrays
):
    feature_setup = sample_config.feature_setup
    customers, products, transactions = raw_sample_tables
    data_path = sample_config.paths.product_feature_path
    feature_path = sample_config.paths.store_feature_path

    customer_list = customers.index.values
    week_list = transactions.index.levels[0].values.tolist()
    product_list = products.index.values.tolist()

    features = {
        feature: map_feature_name_to_object(feature, FeatureCatalog)  # type: ignore
        for feature in feature_setup.customer_feature.features
    }
    fg = StoreFeatureGenerator(
        chunk_idx=0,
        customer_list=customer_list,
        week_list=week_list,
        product_list=product_list,
        features=features,
        feature_path=feature_path,
    )
    item_qty = fg._load_data_array(data_path, "item_qty")
    unit_price = fg._load_data_array(data_path, "unit_price")
    assert item_qty.sum() == transactions.item_qty.sum()
    assert unit_price.mean() == transactions.unit_price.mean()


def test_category_feature_loading_raw_array(
    sample_config, raw_sample_tables, serialized_trx_arrays
):
    feature_setup = sample_config.feature_setup
    customers, products, transactions = raw_sample_tables
    data_path = sample_config.paths.product_feature_path
    feature_path = sample_config.paths.category_feature_path

    customer_list = customers.index.values
    week_list = transactions.index.levels[0].values.tolist()
    product_list = products.index.values.tolist()

    features = {
        feature: map_feature_name_to_object(feature, FeatureCatalog)  # type: ignore
        for feature in feature_setup.customer_category_feature.features
    }
    fg = CategoryFeatureGenerator(
        chunk_idx=0,
        customer_list=customer_list,
        week_list=week_list,
        product_list=product_list,
        features=features,
        feature_path=feature_path,
        product_category_mapping=products,
    )
    from pathlib import Path

    assert (len(list(Path(data_path).glob("*/*")))) == 28
    item_qty = fg._load_data_array(data_path, "item_qty")
    unit_price = fg._load_data_array(data_path, "unit_price")
    assert item_qty.sum() == transactions.item_qty.sum()
    assert unit_price.mean() == transactions.unit_price.mean()
