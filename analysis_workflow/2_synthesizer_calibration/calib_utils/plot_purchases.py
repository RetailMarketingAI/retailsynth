from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from sdmetrics.single_column import KSComplement, RangeCoverage


def gather_summary_statistics(data: pd.DataFrame, target_column: str):
    """Compute summary statistics.

    Parameters
    ----------
        data (pd.DataFrame): decision dataframe with identifiers of week, customer, item (store / category / product)
        target_column (str): column to collect summary statistics

    Returns
    -------
        dictionary: dictionary with summary statistics
    """
    mean = data[target_column].mean()
    std = data[target_column].std()
    return {"mean": mean, "std": std}


def format_stats_dict(result: Dict) -> str:
    """Convert dictionary of summary statistics to readable string.

    Parameters
    ----------
        result (Dict): dictionary of summary statistics

    Returns
    -------
        str: formatted summary statistics
    """
    string = ""
    for name, value in result.items():
        string = string + "%s: %.6f, " % (name, value)
    return string


def gather_report_one_step(
    results: List,
    labels: List,
    step_name: str = "store_visit",
    target_name: str = "store_visit",
) -> str:
    """Prepare summary statistics report for one decision step.

    Parameters
    ----------
        results (List): list of datasets to gather summary statistics
        labels (List): list of names of the datasets
        step_name (str, optional): the name of the decision step. Defaults to 'store_visit'.
        target_name (str, optional): the column to collect summary statistics on. Defaults to 'store_visit'.

    Returns
    -------
        str: formatted report for one step
    """
    report = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    report = report + step_name + ", " + target_name + "\n"
    for i in range(len(results)):
        stats = gather_summary_statistics(results[i][step_name], target_name)
        report = report + labels[i] + " " + format_stats_dict(stats) + "\n"
    return report


def gather_summary_stats_report(results: List, labels: List):
    """Prepare summary report for all steps, store visit, category choice, product choice and product demand.

    Parameters
    ----------
        results (List): list of datasets to gather summary statistics
        labels (List): list of names of the datasets

    Returns
    -------
        str: a complete report
    """
    report = ""
    report += gather_report_one_step(results, labels, "store_visit", "store_visit")
    report += gather_report_one_step(
        results, labels, "category_choice", "category_choice"
    )
    report += gather_report_one_step(
        results, labels, "product_choice", "product_choice"
    )
    report += gather_report_one_step(results, labels, "product_demand", "item_qty")
    return report


def compute_similarity_metrics(real_data, synthetic_data):
    ks = KSComplement.compute(
        real_data=real_data,
        synthetic_data=synthetic_data,
    )
    rc = RangeCoverage.compute(
        real_data=real_data,
        synthetic_data=synthetic_data,
    )
    return f"KS complement: {ks:.3f}\nRange coverage: {rc:.3f}"


@dataclass
class Plotter:
    """plotting objects to get figures of distribution comparison."""

    results: List[Dict]
    labels: List[str]
    row_size: int

    def __post_init__(self):
        self.colors = sns.color_palette()[: len(self.results)]
        for result in self.results:
            assert "store_visit" in result
            assert "category_choice" in result
            assert "product_choice" in result
            assert "product_demand" in result

    def examine_datasets(
        self,
        plotting_directives,
        row_size=20,
        share_axes=True,
        label_args: dict = {},
        legend_args: dict = {},
        text_args: dict = {},
        text_loc="upper right",
    ):
        # method to draw a series of plots at the same time
        fig, axes = plt.subplots(
            len(plotting_directives),
            1,
            figsize=(row_size, row_size // 3 * len(plotting_directives)),
        )

        for i, plotdir in enumerate(plotting_directives):
            fn, arg = plotdir
            fn(
                axes=[axes[i], axes[i]],
                **arg,
                label_args=label_args,
                legend_args=legend_args,
                text_args=text_args,
                text_loc=text_loc,
            )
        plt.tight_layout()
        return fig, axes

    def plot_time_since_last_purchase(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        acq_week=3,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare time since last purchase from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))

        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = data_sample["store_visit"]
            df = data_sample.reset_index()
            segment = (
                df.query(f"week <= {acq_week}")
                .query("store_visit==1")
                .customer_key.unique()
            )
            acquired_customer_visits = df.loc[df.customer_key.isin(segment)].query(
                f"week > {acq_week}"
            )
            time_since_last_purchase = acquired_customer_visits.time_since_last_purchase
            ks_data.append(time_since_last_purchase)
            sns.histplot(
                time_since_last_purchase,
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
            )
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Time since last purchase", **label_args)
            ax.set_ylabel("", **label_args)
            ax.set(yscale="log")
            if legend:
                ax.legend(**legend_args)
        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)

    def plot_category_purchase_freq_by_hierarchy(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare category choice probability from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))

        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = (
                data_sample["category_choice"]
                .groupby("category_nbr")["category_choice"]
                .mean()
            )
            ks_data.append(data_sample)
            sns.histplot(
                data_sample,
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
            )
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Category purchase probability", **label_args)

            ax.set_ylabel("", **label_args)
            if legend:
                ax.legend(**legend_args)
        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)

    def plot_product_purchase_freq_by_hierarchy(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare product choice probability from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))

        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = (
                data_sample["product_choice"]
                .groupby("product_nbr")["product_choice"]
                .mean()
            )
            ks_data.append(data_sample)
            sns.histplot(
                data_sample,
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
            )
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Product purchase probability", **label_args)

            ax.set_ylabel("", **label_args)
            if legend:
                ax.legend(**legend_args)

        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)

    def plot_individual_demand(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare quantity demand from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))
        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = data_sample["product_demand"].item_qty
            ks_data.append(data_sample)
            sns.histplot(
                data_sample,
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
            )
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Item quantity", **label_args)

            ax.set_ylabel("", **label_args)
            if legend:
                ax.legend(**legend_args)

        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)

    def plot_product_price_variation(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare product price from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))

        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = (
                data_sample["product_demand"].groupby("product_nbr").unit_price.mean()
            )
            ks_data.append(data_sample)
            sns.histplot(
                data_sample,
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
            )
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Product price", **label_args)
            ax.set_ylabel("", **label_args)
            if legend:
                ax.legend(**legend_args)

        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)

    def plot_basket_size_category_count(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare number of categories purchased from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))

        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = (
                data_sample["category_choice"]
                .groupby(["week", "customer_key"])["category_choice"]
                .sum()
            )
            ks_data.append(data_sample)
            n_missing_entries = (
                data_sample.reset_index().week.nunique()
                * data_sample.reset_index().customer_key.nunique()
                - len(data_sample.index)
            )
            sns.histplot(
                np.hstack((data_sample.values, np.zeros(n_missing_entries))),
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
            )
            ax.set_xlim(xlim_min, xlim_max)
            ax.set(yscale="log")

            ax.set_xlabel("Categories purchased per customer per week", **label_args)
            ax.set_ylabel("", **label_args)
            if legend:
                ax.legend(**legend_args)
        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)

    def plot_basket_size(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare weekly basket size from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))

        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = (
                data_sample["product_demand"]
                .groupby(["week", "customer_key"])["item_qty"]
                .sum()
            )
            ks_data.append(data_sample)
            n_missing_entries = (
                data_sample.reset_index().week.nunique()
                * data_sample.reset_index().customer_key.nunique()
                - len(data_sample.index)
            )
            sns.histplot(
                np.hstack((data_sample.values, np.zeros(n_missing_entries))),
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
            )
            ax.set_xlim(xlim_min, xlim_max)
            ax.set(yscale="log")

            ax.set_xlabel("Items purchased per customer per week", **label_args)
            ax.set_ylabel("", **label_args)
            if legend:
                ax.legend(**legend_args)
        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)

    def plot_product_sales(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare weekly units of product sold from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))

        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = (
                data_sample["product_demand"]
                .groupby(["week", "product_nbr"])["item_qty"]
                .sum()
            )
            n_missing_entries = (
                data_sample.reset_index().week.nunique()
                * data_sample.reset_index().product_nbr.nunique()
                - len(data_sample)
            )
            plot_data = np.hstack((data_sample.values, np.zeros(n_missing_entries)))
            ks_data.append(plot_data)

            sns.histplot(
                plot_data,
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
            )
            ax.set(yscale="log")
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Items purchased per product per week", **label_args)
            ax.set_ylabel("", **label_args)
            if legend:
                ax.legend(**legend_args)
        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)

    def plot_customer_visits(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        acq_week=3,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare store visit probability from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))
        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = (
                (data_sample["store_visit"].item_qty > 0).groupby("week").mean()
            )
            data_sample = (
                data_sample.reset_index()
                .query(f"week > {acq_week}")
                .drop("week", axis=1)
            )
            data_sample = data_sample.values.flatten()
            ks_data.append(data_sample)
            sns.histplot(
                data_sample,
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
                legend=False,
            )
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Store visit probability", **label_args)
            ax.set_ylabel("", **label_args)
            if legend:
                ax.legend(**legend_args)
        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)

    def plot_product_counts_per_category(
        self,
        binwidth,
        xlim_min,
        xlim_max,
        axes=None,
        legend=False,
        label_args={},
        legend_args={},
        text_args={},
        text_loc="upper right",
    ):
        # method to compare product counts for categories from the real data and the synthetic data
        if axes is None:
            _, axes = plt.subplots(1, len(self.results))
        ks_data = []
        for data_sample, label_sample, color_sample, ax in zip(
            self.results, self.labels, self.colors, axes
        ):
            data_sample = (
                data_sample["products"]
                .reset_index()
                .groupby("category_nbr")
                .product_nbr.nunique()
            )
            data_sample = data_sample.values.flatten()
            ks_data.append(data_sample)
            sns.histplot(
                data_sample,
                ax=ax,
                stat="probability",
                label=label_sample,
                color=color_sample,
                edgecolor=color_sample,
                bins=np.arange(xlim_min, xlim_max + binwidth, binwidth),
                element="step",
                legend=False,
            )
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Category size", **label_args)
            ax.set_ylabel("", **label_args)
            if legend:
                ax.legend(**legend_args)
        metric = compute_similarity_metrics(ks_data[0], ks_data[1])
        anc = AnchoredText(metric, loc=text_loc, frameon=True, prop=text_args)
        axes[0].add_artist(anc)
        axes[0].set_ylabel("Proportion", **label_args)
