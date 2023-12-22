import logging
import os
import pickle
import time
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from retailsynth.datasets.complete_journey.preprocess_pipeline import run_preprocess
from retailsynth.feature_eng.feature_generator import serialize_historical_transactions
from retailsynth.feature_eng.feature_loader import initialize_feature_loader
from retailsynth.feature_eng.feature_pipeline import run_feature_generation
from retailsynth.feature_eng.viz_data_layer import FeatureVizDataLayer
from retailsynth.synthesizer.synthesizer_pipeline import run_synthesizer
from retailsynth.utils.storage import clear_cwd


def prepare_txns(cfg):
    data_path = cfg.paths.processed_data
    os.makedirs(data_path, exist_ok=True)
    if cfg.synthetic_data is not None:
        logging.info("Running data synthesis")
        t1 = time.time()
        (
            customers,
            products,
            transactions,
            synthesizer,
        ) = run_synthesizer(cfg.synthetic_data)
        logging.info(
            f"Generated synthetic data for {len(customers)} customers, {products.category_nbr.nunique()} categories, {len(products)} products in {time.time() - t1} seconds."
        )
        stats_path = Path(data_path, "synthesizer.pickle").resolve()
        with open(stats_path, "wb") as file:
            pickle.dump(synthesizer, file)
    else:
        customers, products, transactions = run_preprocess(cfg)
    products.to_parquet(Path(data_path, "annotated_products.parquet"))
    return customers, products, transactions


def prepare_viz_features_from_txns(cfg, customers, products, transactions):
    data_path = cfg.paths.processed_data
    os.makedirs(data_path, exist_ok=True)

    """ 
    Serialize individual customer trx in zarr files
    """
    t1 = time.time()
    customer_chunk_size = cfg.feature_setup.customer_chunk_size
    serialize_historical_transactions(
        customers,
        products,
        transactions,
        cfg.paths.txns_array_path,
        customer_chunk_size,
    )

    """ 
    Aggregate the individual product trx to the category level and the store level
    """
    t1 = time.time()
    txns_array_path = cfg.paths.txns_array_path

    customer_chunk_size = cfg.feature_setup.customer_chunk_size
    run_feature_generation(
        customers,
        products,
        transactions,
        cfg.feature_setup,
        txns_array_path=txns_array_path,
        customer_chunk_size=customer_chunk_size,
        n_workers=cfg.n_workers,
    )
    logging.info(f"Computing aggregated features in {time.time() - t1} seconds.")

    """
    Generate features for visualization and store in parquet files
    """
    t1 = time.time()
    feature_loader = initialize_feature_loader(cfg.paths)
    product_category_mapping = products.reset_index()[["product_nbr", "category_nbr"]]
    data_path = cfg.paths.processed_data
    results = FeatureVizDataLayer().prepare_visualization_features(
        feature_loader, product_category_mapping, data_path=data_path
    )
    logging.info(
        f"Preparing consolidated data frame for visualizations in {time.time() - t1} seconds."
    )

    return results


def viz_main(cfg):
    cfg = OmegaConf.to_object(cfg)

    customers, products, transactions = prepare_txns(cfg)
    _ = prepare_viz_features_from_txns(cfg, customers, products, transactions)


@hydra.main(
    config_path="../../../analysis_workflow/cfg",
    config_name="data_viz_real_data",
)
def run_real_data(cfg):
    clear_cwd(cfg.paths.processed_data)
    viz_main(cfg)


@hydra.main(
    config_path="../../../analysis_workflow/cfg",
    config_name="data_viz_synthetic_data_in_sample",
)
def run_synthetic_data_in_sample(cfg):
    clear_cwd(cfg.paths.processed_data)
    viz_main(cfg)


if __name__ == "__main__":
    run_real_data()
    run_synthetic_data_in_sample()
