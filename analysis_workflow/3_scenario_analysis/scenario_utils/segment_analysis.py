import pickle
from enum import Enum
from pathlib import Path, PosixPath

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic.dataclasses import dataclass

from retailsynth.synthesizer.data_synthesizer import DataSynthesizer
from retailsynth.utils.storage import load_result


@dataclass
class Loader:
    """Load the data or synthesizer for a given policy.

    Parameters
    ----------
        policy_name (str): the name of the policy
        data_path (PosixPath): the path to the synthetic data.
        data (dict, optional): the synthetic data. Defaults to None.
        load_synthesizer (bool, optional): whether to load the synthesizer. Defaults to True.
        synthesizer ([type], optional): the synthesizer. Defaults to None.
    """

    policy_name: str
    data_path: PosixPath
    data: dict = None
    load_synthesizer: bool = True
    synthesizer = None

    def __post_init__(self):
        """Load the feature dataframes and the synthesizer object."""
        self.data = load_result(self.data_path)
        if self.load_synthesizer:
            with open(Path(self.data_path, "synthesizer.pickle"), "rb") as fp:
                self.synthesizer = pickle.load(fp)

    def __getitem__(self, index: str):
        """Make the model config subscriptable."""
        try:
            return getattr(self, index)
        except AttributeError:
            raise KeyError(f"Invalid attribute: {index}")


def segment_customer(
    synthesizer: DataSynthesizer, n_segment: int = 4, segment_name=None
) -> pd.DataFrame:
    """Segment customers according their price sensitivity.

    Parameters
    ----------
        synthesizer (DataSynthesizer)
        n_segment (int, optional): number of segments to divide. Defaults to 4.
        segment_name (list, optional): list of names for segments. Defaults to None.

    Returns
    -------
        pd.DataFrame: customer info data frame with columns of customer_key and segment
    """
    utility_beta_ui_w = synthesizer.utility_beta_ui_w.mean(axis=-1)
    sorted_indices = jnp.argsort(utility_beta_ui_w)
    segment_size = len(sorted_indices) // n_segment
    segment_indices = [
        sorted_indices[i * segment_size : (i + 1) * segment_size]
        for i in range(n_segment)
    ]

    if segment_name is None:
        segment_name = [f"segment_{i}" for i in range(n_segment)]
    else:
        assert (
            len(segment_name) == n_segment
        ), "segment_name must have the same length as n_segment"
    customers = []
    for segment_no in range(n_segment):
        df = pd.DataFrame(
            {
                "customer_key": segment_indices[segment_no],
                "segment": segment_name[segment_no],
            }
        )
        customers.append(df)
    customers = pd.concat(customers)
    return customers


def compute_steady_state(transition_matrix: jnp.ndarray):
    """Compute the steady state for pct of products on discount."""
    # Transpose the matrix and find its eigenvalues and eigenvectors
    transposed_matrix = jnp.transpose(transition_matrix)
    eigenvalues, eigenvectors = jnp.linalg.eig(transposed_matrix)

    # Find the index of the eigenvalue closest to 1 (within a small threshold)
    steady_state_index = jnp.argmin(jnp.abs(eigenvalues - 1.0))

    # Extract the corresponding eigenvector as the steady state vector
    steady_state = jnp.abs(eigenvectors[:, steady_state_index])
    steady_state /= jnp.sum(steady_state)  # Normalize the probabilities to sum up to 1

    return steady_state


def get_discount_desc(synthesizer: DataSynthesizer):
    # compute discount frequency, discount depth, and discount level in one step
    transition_prob = synthesizer.transition_prob.mean.mean(axis=0)
    transition_matrix = jnp.array(
        [
            [1 - transition_prob[0], transition_prob[0]],
            [1 - transition_prob[1], transition_prob[1]],
        ]
    )
    steady_state = compute_steady_state(transition_matrix)
    discount_frequency = round(steady_state[1].item(), 2)
    discount_depth = synthesizer.discount_depth_distribution.mean
    discount_level = (
        jnp.array(synthesizer.choice_decision_stats["discount"]).mean().item()
    )
    return discount_frequency, discount_depth, discount_level


def prepare_customer_product_trx(
    customers: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame
):
    # merge customers, products, and transactions into one data frame
    transactions.loc[:, "sales_amt"] = (
        transactions["item_qty"] * transactions["unit_price"]
    )
    transactions = transactions.reset_index().merge(
        products, left_on="product_nbr", right_index=True
    )
    transactions = transactions.merge(customers, on="customer_key")
    return transactions.sort_values(by=["week", "segment"])


def calculate_active_customers_per_week(df, window_length: int = 4):
    df = df.sort_values(by=["week", "customer_key"])

    weeks = np.linspace(
        df.week.min(), df.week.max(), df.week.max() - df.week.min() + 1, dtype=int
    )

    active_customers_per_week = []
    # Iterate through each week
    for week in weeks:
        # Filter transactions for the past 4 weeks including the current week
        recent_weeks = df[df["week"].between(week - (window_length - 1), week)]

        # Get unique customers in the recent weeks
        active_customers = (
            recent_weeks.groupby("segment")["customer_key"]
            .nunique()
            .to_frame(name="active_customers")
        )
        active_customers.loc[:, "week"] = week

        # Store the count of active customers for the current week
        active_customers_per_week.append(active_customers.reset_index())

    return pd.concat(active_customers_per_week)


def compute_retention_rate_time_series(
    results: list, labels: list, customers: pd.DataFrame, window_length: int = 4
):
    customer_segment_size = (
        customers.groupby("segment")
        .customer_key.nunique()
        .to_frame(name="segment_size")
    )
    stats_df = []
    for result, label in zip(results, labels):
        result = result.merge(customers, on="customer_key")
        active_customers = calculate_active_customers_per_week(result, window_length)
        active_customers = active_customers.merge(
            customer_segment_size, left_on="segment", right_index=True
        )
        active_customers.loc[:, "retention_rate"] = (
            active_customers.loc[:, "active_customers"]
            / active_customers.loc[:, "segment_size"]
        )
        active_customers.loc[:, "policy"] = label
        stats_df.append(active_customers)
    stats_df = pd.concat(stats_df)

    return stats_df


def compute_total_item_qty(trx):
    return trx.groupby("segment").item_qty.sum()


def compute_total_revenue(trx):
    return trx.groupby("segment").sales_amt.sum()


def compute_weekly_avg_item_qty(trx):
    weekly_item_qty = trx.groupby(["week", "segment"]).item_qty.sum().reset_index()
    return weekly_item_qty.groupby("segment").item_qty.mean()


def compute_weekly_avg_revenue(trx):
    weekly_revenue = trx.groupby(["week", "segment"]).sales_amt.sum().reset_index()
    return weekly_revenue.groupby("segment").sales_amt.mean()


def compute_weekly_avg_item_qty_per_customer(trx):
    weekly_item_qty = trx.groupby(["week", "segment"]).item_qty.sum()
    weekly_customer_visit = trx.groupby(["week", "segment"]).customer_key.nunique()
    return (weekly_item_qty / weekly_customer_visit).groupby("segment").mean()


def compute_weekly_avg_revenue_per_customer(trx):
    weekly_revenue = trx.groupby(["week", "segment"]).sales_amt.sum()
    weekly_customer_visit = trx.groupby(["week", "segment"]).customer_key.nunique()
    return (weekly_revenue / weekly_customer_visit).groupby("segment").mean()


def compute_weekly_avg_category_sold(trx):
    # if customer a and customer b both purchase category 1, we count the number of category sold as 2
    trx = trx[["week", "customer_key", "segment", "category_nbr"]].drop_duplicates()
    weekly_category_sold = trx.groupby(["week", "segment"]).category_nbr.count()
    return weekly_category_sold.groupby("segment").mean()


def compute_weekly_avg_category_sold_per_customer(trx):
    # compute the number of categories sold normalized by unique number of customers visit
    trx = trx[["week", "customer_key", "segment", "category_nbr"]].drop_duplicates()
    weekly_category_sold = trx.groupby(["week", "segment"]).category_nbr.count()
    weekly_customer_visit = trx.groupby(["week", "segment"]).customer_key.nunique()
    return (weekly_category_sold / weekly_customer_visit).groupby("segment").mean()


def compute_weekly_avg_pct_customer_visit(trx):
    # number of unique customer visits normalized by number of customers shown up in the trajectory
    weekly_customer_visit = (
        trx.groupby(["week", "segment"]).customer_key.nunique().reset_index()
    )
    weekly_avg_customer_visit = weekly_customer_visit.groupby(
        ["segment"]
    ).customer_key.mean()
    segment_size = trx.groupby(["segment"]).customer_key.nunique()
    return weekly_avg_customer_visit / segment_size


def compute_weekly_avg_customer_visit(trx):
    weekly_customer_visit = (
        trx.groupby(["week", "segment"]).customer_key.nunique().reset_index()
    )
    return weekly_customer_visit.groupby(["segment"]).customer_key.mean()


def compute_final_week_retention_rate(trx, window_length: int = 3):
    # count the pct of customers still make a purchase at the final week
    last_day = trx.week.max()
    active_customers = trx[
        trx["week"].between(last_day - (window_length - 1), last_day)
    ]
    retention = active_customers.groupby("segment").customer_key.nunique()
    segment_size = trx.groupby(["segment"]).customer_key.nunique()
    return retention / segment_size


def compute_effective_discount(trx):
    # count the mean discount attracts customers to purchase
    return trx.groupby("segment").discount_portion.mean()


def compute_mean_elasticity(synthesizer):
    # compute the mean elasticity for each customer
    mean_elasticity = jnp.array(
        synthesizer.elasticity_stats["overall_elasticity"]
    ).mean(axis=(0, -1))
    result = pd.Series(mean_elasticity, index=jnp.arange(synthesizer.n_customer))
    return result


def compute_mean_utility_beta_w(synthesizer):
    # compute the mean utility_beta_ui_w for each customer
    mean_beta = synthesizer.utility_beta_ui_w.mean(axis=-1)
    result = pd.Series(mean_beta, index=jnp.arange(synthesizer.n_customer))
    return result


class ReportMetric(Enum):
    total_item_qty = compute_total_item_qty
    total_revenue = compute_total_revenue
    weekly_avg_item_qty = compute_weekly_avg_item_qty
    weekly_avg_revenue = compute_weekly_avg_revenue
    weekly_avg_item_qty_per_customer = compute_weekly_avg_item_qty_per_customer
    weekly_avg_revenue_per_customer = compute_weekly_avg_revenue_per_customer
    weekly_avg_category_sold = compute_weekly_avg_category_sold
    weekly_avg_category_sold_per_customer = (
        compute_weekly_avg_category_sold_per_customer
    )
    weekly_avg_pct_customer_visit = compute_weekly_avg_pct_customer_visit
    weekly_avg_customer_visit = compute_weekly_avg_customer_visit
    final_week_retention_rate = compute_final_week_retention_rate
    effective_discount = compute_effective_discount
    mean_elasticity = compute_mean_elasticity
    mean_utility_beta_w = compute_mean_utility_beta_w


def generate_segment_report_from_trx_for_one_policy(
    data_loader, n_segment, segment_name=None
):
    # return a data frame for segment analysis report, containing the following metrics
    metrics = [
        "total_item_qty",
        "total_revenue",
        "weekly_avg_item_qty",
        "weekly_avg_revenue",
        "weekly_avg_item_qty_per_customer",
        "weekly_avg_revenue_per_customer",
        "weekly_avg_category_sold",
        "weekly_avg_category_sold_per_customer",
        "weekly_avg_pct_customer_visit",
        "weekly_avg_customer_visit",
        "final_week_retention_rate",
        "effective_discount",
    ]
    customers = segment_customer(data_loader.synthesizer, n_segment, segment_name)
    trx = prepare_customer_product_trx(
        customers, data_loader.data["products"], data_loader.data["product_demand"]
    )
    (discount_frequency, discount_depth, discount_level) = get_discount_desc(
        data_loader.synthesizer
    )
    report = pd.DataFrame(
        {metric: getattr(ReportMetric, metric)(trx) for metric in metrics}
    )
    report.loc[:, "discount_frequency"] = discount_frequency
    report.loc[:, "discount_depth"] = discount_depth
    report.loc[:, "discount_level"] = discount_level
    report.loc[:, "policy"] = data_loader.policy_name
    # columns: segment, policy, ... (metrics)
    return report.reset_index()


def generate_segment_report_from_synthesizer_for_one_policy(
    data_loader, n_segment, segment_name=None
):
    # return a data frame for segment analysis report, containing the following metrics
    metrics = [
        "mean_elasticity",
        "mean_utility_beta_w",
    ]
    customers = segment_customer(data_loader.synthesizer, n_segment, segment_name)
    synthesizer = data_loader.synthesizer
    report = pd.DataFrame(
        data={metric: getattr(ReportMetric, metric)(synthesizer) for metric in metrics}
    )
    report = report.merge(customers, left_index=True, right_on="customer_key")
    report = report.groupby("segment")[metrics].mean()
    report.loc[:, "policy"] = data_loader.policy_name
    # columns: segment, policy, ... (metrics)
    return report.reset_index()


def plot_retention_rate_time_series_for_one_segment(
    df, target_segment=None, ax=None, title=None
):
    if ax is None:
        ax = plt.gca()
    if target_segment is None:
        target_segment = df.segment.iloc[0]  # take the first segment
    df = df[df.segment == target_segment]
    sns.lineplot(
        data=df, x="week", y="retention_rate", errorbar=("ci", 0), hue="policy", ax=ax
    )
    ax.set_ylabel("Retention rate")
    ax.set_xlabel("Week")
    title = title or f"{target_segment} segment"
    ax.set_title(title)
    return ax


def plot_retention_rate_time_series(df, row_size: int = 8, column_size: int = 3):
    # plot the retention rate time series for each segment in one panel
    segment_name = df.segment.unique()
    n_segment = df.segment.nunique()
    fig, axes = plt.subplots(
        n_segment, 1, figsize=(row_size, column_size * n_segment), sharey=True
    )
    for segment, ax in zip(segment_name, axes):
        plot_retention_rate_time_series_for_one_segment(df, segment, ax=ax)
    plt.subplots_adjust(hspace=0.5)
    return fig, axes


def plot_final_retention_rate_for_one_segment(
    df, target_segment=None, ax=None, title=None, order: list = None
):
    if ax is None:
        ax = plt.gca()
    if target_segment is None:
        target_segment = df.segment.iloc[0]  # take the first segment
    df = df[df.segment == target_segment]
    sns.barplot(
        data=df, x="policy", y="retention_rate", order=order, hue="policy", ax=ax
    )
    ax.set_ylabel("Retention rate")
    ax.set_xlabel("Policy")
    title = title or f"{target_segment} segment"
    ax.set_title(title)
    return ax


def plot_final_retention_rate(
    df, row_size: int = 10, column_size: int = 10, order: list = None
):
    # plot the final retention rate for each segment in one panel
    segment_name = df.segment.unique()
    fig, axes = plt.subplots(2, 2, figsize=(row_size, column_size), sharey=True)
    for segment, ax in zip(segment_name, axes.flatten()):
        plot_final_retention_rate_for_one_segment(df, segment, ax=ax, order=order)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle("Retention rate at the final week")
    return fig, axes


def plot_purchase_summary_for_segments(
    reports: pd.DataFrame,
    row_size: int = 15,
    column_size: int = 3,
    palette: list = None,
    axes: list = None,
    label_args={},
    title_args={},
    legend_args={},
):
    assert "total_revenue" in reports.columns
    assert "final_week_retention_rate" in reports.columns
    assert "weekly_avg_category_sold_per_customer" in reports.columns
    assert "effective_discount" in reports.columns
    assert "discount_level" in reports.columns

    if axes is None:
        fig, axes = plt.subplots(1, 5, figsize=(row_size, column_size))

    sns.lineplot(
        data=reports,
        x="policy",
        y="effective_discount",
        hue="segment",
        palette=palette,
        ax=axes[0],
        legend=False,
        marker="o",
    )
    axes[0].set_title("Average Discount", **title_args)
    axes[0].set_ylabel("Effective discount", **label_args)
    axes[0].set_xlabel("Policy", **label_args)

    sns.lineplot(
        data=reports,
        x="policy",
        y="total_item_qty",
        hue="segment",
        legend=False,
        ax=axes[1],
        palette=palette,
        marker="o",
    )
    axes[1].set_title("Demand", **title_args)
    axes[1].set_ylabel("Quantity sold", **label_args)
    axes[1].ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
    axes[1].set_xlabel("Policy", **label_args)

    sns.lineplot(
        data=reports,
        x="policy",
        y="total_revenue",
        hue="segment",
        legend=False,
        ax=axes[2],
        palette=palette,
        marker="o",
    )
    axes[2].set_title("Revenue", **title_args)
    axes[2].set_ylabel("Revenue", **label_args)
    axes[2].ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
    axes[2].set_xlabel("Policy", **label_args)

    sns.lineplot(
        data=reports,
        x="policy",
        y="weekly_avg_category_sold_per_customer",
        hue="segment",
        legend=False,
        palette=palette,
        ax=axes[3],
        marker="o",
    )
    axes[3].set_title("Category Penetration", **title_args)
    axes[3].set_ylabel("Categories sold", **label_args)
    axes[3].set_xlabel("Policy", **label_args)

    sns.lineplot(
        data=reports,
        x="policy",
        y="final_week_retention_rate",
        hue="segment",
        ax=axes[4],
        palette=palette,
        marker="o",
    )
    axes[4].set_title("Customer Retention", **title_args)
    axes[4].set_ylabel("Retention rate", **label_args)
    axes[4].set_xlabel("Policy", **label_args)

    handles, _ = axes[4].get_legend_handles_labels()
    axes[4].legend(
        title="Price sensitivity",
        handles=handles,
        labels=["High", "Medium", "Low"],
        **legend_args,
    )

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fig = axes[0].get_figure()

    return fig, axes


def plot_revenue_retention_tradeoff(
    reports,
    ax=None,
    ci: int = 80,
    palette=None,
):
    # return a scatter plot between revenue and retention rate
    segment_names = reports.segment.unique()
    for i in range(len(segment_names)):
        segment = segment_names[i]
        color = palette[segment]
        sns.regplot(
            data=reports.query(f'segment == "{segment}"'),
            x="final_week_retention_rate",
            y="total_revenue",
            ax=ax,
            ci=ci,
            label=segment,
            color=color,
        )
    plt.legend()
    plt.title("Tradeoff between revenue and retention rate")
    plt.ylabel("Revenue")
    plt.xlabel("Customer retention: retention rate")

    return ax
