import logging
import hydra
from omegaconf import OmegaConf
import seaborn as sns
from calib_utils.plot_purchases import Plotter
from sdmetrics.single_column import KSComplement

from retailsynth import REPO_ROOT_DIR
from retailsynth.utils.dataset_viz import viz_main
from retailsynth.utils.storage import clear_cwd, load_result

# This python script sets up the entry point for parameter sweeping
# and find the optimal config to calibrate the synthesizer to the real data.

REAL_DATA_PATH = (
    REPO_ROOT_DIR
    / "analysis_workflow/2_synthesizer_calibration/outputs/data/processed/complete_journey_data"
)
OPTIMIZATION_MODE = OmegaConf.load(
    REPO_ROOT_DIR
    / "analysis_workflow/2_synthesizer_calibration/cfg/parameter_sweeping.yaml"
).hydra.sweeper.study_name
ACQ_WEEK_THR = 3


def report_ks_metrics(real_result, synthetic_result):
    """Gather similarity statistics from real and synthetic data.

    Parameters
    ----------
        real_result (dict): the dictionary of real feature df for each step
        synthetic_result (dict): the dictionary of synthetic feature df for each step

    Returns
    -------
        tuple: ks metrics for basket size, category choice probability and product choice probability
    """
    # get basket size in category counts
    basket_size_ks_data = []
    for data_sample in [real_result, synthetic_result]:
        data_sample = (
            data_sample["category_choice"]
            .groupby(["week", "customer_key"])["category_choice"]
            .sum()
        )
        basket_size_ks_data.append(data_sample)
    basket_size_ks_data = KSComplement.compute(
        real_data=basket_size_ks_data[0],
        synthetic_data=basket_size_ks_data[1],
    )

    # get category purchase probability
    category_purchase_ks_data = []
    for data_sample in [real_result, synthetic_result]:
        data_sample = (
            data_sample["category_choice"]
            .groupby("category_nbr")["category_choice"]
            .mean()
        )
        category_purchase_ks_data.append(data_sample)
    category_purchase_ks_data = KSComplement.compute(
        real_data=category_purchase_ks_data[0],
        synthetic_data=category_purchase_ks_data[1],
    )

    # get product purchase probability
    product_purchase_ks_data = []
    for data_sample in [real_result, synthetic_result]:
        data_sample = (
            data_sample["product_choice"]
            .groupby("product_nbr")["product_choice"]
            .mean()
        )
        product_purchase_ks_data.append(data_sample)
    product_purchase_ks_data = KSComplement.compute(
        real_data=product_purchase_ks_data[0],
        synthetic_data=product_purchase_ks_data[1],
    )

    # get store visit probability
    store_visit_ks_data = []
    for data_sample in [real_result, synthetic_result]:
        data_sample = (
            data_sample["store_visit"].groupby("customer_key")["store_visit"].mean()
        )
        store_visit_ks_data.append(data_sample)
    store_visit_ks_data = KSComplement.compute(
        real_data=store_visit_ks_data[0],
        synthetic_data=store_visit_ks_data[1],
    )

    # get product demand distribution
    product_demand_ks_data = []
    for data_sample in [real_result, synthetic_result]:
        data_sample = data_sample["product_demand"].item_qty
        product_demand_ks_data.append(data_sample)
    product_demand_ks_data = KSComplement.compute(
        real_data=product_demand_ks_data[0],
        synthetic_data=product_demand_ks_data[1],
    )

    return (
        basket_size_ks_data,
        category_purchase_ks_data,
        product_purchase_ks_data,
        store_visit_ks_data,
        product_demand_ks_data,
    )


def get_optimization_goal(
    basket_ks,
    category_ks,
    product_ks,
    store_ks,
    demand_ks,
    mode: str = "best_category_fit",
):
    """Define the optimization goal for parameter sweeping.

    Parameters
    ----------
        basket_ks (float): ks metric for basket size
        category_ks (float): ks metric for category choice probability
        product_ks (float): ks metric for product choice probability
        store_ks (float): ks metric for store visit probability
        demand_ks (float): ks metric for product demand distribution
        mode (str, optional): the mode to define the optimization goal. Defaults to "best_category_fit".

    Returns
    -------
        float: the sum of the 3 ks metrics
    """
    if mode == "best_category_fit":
        return basket_ks + category_ks
    elif mode == "best_product_fit":
        return product_ks
    elif mode == "best_overall_fit":
        return basket_ks + product_ks
    elif mode == "best_store_fit":
        return store_ks
    elif mode == "best_demand_fit":
        return demand_ks
    else:
        raise ValueError(
            "Unknown mode. Supported modes include best_category_fit, best_product_fit, best_overall_fit, best_store_fit, best_demand_fit."
        )


def compute_objective(cfg):
    try:
        viz_main(cfg)
    except AssertionError:
        return -1
    real_result = load_result(REAL_DATA_PATH)
    synthetic_result = load_result(cfg.paths.processed_data)

    basket_ks, category_ks, product_ks, store_ks, demand_ks = report_ks_metrics(
        real_result, synthetic_result
    )

    """
    Get distribution plots
    """
    results = [real_result, synthetic_result]
    sns.color_palette(n_colors=len(results))

    plotter = Plotter(
        results=[real_result, synthetic_result],
        labels=["real_data", "synthetic_data"],
        row_size=20,
    )
    # Look at customer visit behavior after burn-in period as set below
    plotting_dirs = [
        (
            plotter.plot_time_since_last_purchase,
            dict(binwidth=1, xlim_min=1, xlim_max=7, legend=True),
        ),
        (
            plotter.plot_basket_size_category_count,
            dict(binwidth=1, xlim_min=0, xlim_max=50),
        ),
        (plotter.plot_basket_size, dict(binwidth=1, xlim_min=0, xlim_max=50)),
        (plotter.plot_product_sales, dict(binwidth=1, xlim_min=0, xlim_max=50)),
    ]
    aggregate_fig, _ = plotter.examine_datasets(plotting_dirs, row_size=8)

    plotting_dirs = [
        (
            plotter.plot_customer_visits,
            dict(
                binwidth=0.05,
                xlim_min=0,
                xlim_max=1,
                acq_week=ACQ_WEEK_THR,
                legend=True,
            ),
        ),
        (
            plotter.plot_category_purchase_freq_by_hierarchy,
            dict(binwidth=0.02, xlim_min=0, xlim_max=1),
        ),
        (
            plotter.plot_product_purchase_freq_by_hierarchy,
            dict(binwidth=0.01, xlim_min=0, xlim_max=1),
        ),
        (plotter.plot_individual_demand, dict(binwidth=1, xlim_min=1, xlim_max=15)),
        (
            plotter.plot_product_price_variation,
            dict(binwidth=1, xlim_min=0, xlim_max=20),
        ),
        (
            plotter.plot_product_counts_per_category,
            dict(binwidth=10, xlim_min=0, xlim_max=200),
        ),
    ]
    individual_fig, _ = plotter.examine_datasets(plotting_dirs, row_size=8)
    # clear stored parquet files in the directory
    clear_cwd(exclude=".log")
    # store plots and log
    aggregate_fig.savefig("aggregate_fig.png")
    individual_fig.savefig("individual_fig.png")
    logging.info(cfg.synthetic_data.synthetic_data_setup)
    logging.info(f"basket ks: {basket_ks}")
    logging.info(f"category ks: {category_ks}")
    logging.info(f"product ks: {product_ks}")
    logging.info(f"store ks: {store_ks}")
    logging.info(f"demand ks: {demand_ks}")

    # define the sum of 5 ks metric as the maximization goal for parameter sweeping
    obj = get_optimization_goal(
        basket_ks, category_ks, product_ks, store_ks, demand_ks, mode=OPTIMIZATION_MODE
    )
    return obj


@hydra.main(
    config_path="cfg",
    config_name="parameter_sweeping",
)
def main(cfg):
    return compute_objective(cfg)


if __name__ == "__main__":
    main()
