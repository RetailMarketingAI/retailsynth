import itertools
from typing import Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from retailsynth.synthesizer.data_synthesizer import DataSynthesizer


class ElasticityVizDataLayer:
    """Prepare data frame used to visualize the elasticity.

    Parameters
    ----------
        n_customer (int): number of customer
        n_category (int): number of category
        n_product (int): number of product
        product_category_mapping (np.array): mapping from product to category
        elasticity_stats (dict): dictionary of elasticity statistics, obtained from the synthesizer
        prob_stats (dict): dictionary of probability statistics, obtained from the synthesizer
    """

    def __init__(
        self,
        n_customer,
        n_category,
        n_product,
        product_category_mapping,
        elasticity_stats,
        prob_stats,
    ):
        self.prob_stats = {
            step_name: jnp.array(array) for step_name, array in prob_stats.items()
        }
        self.elasticity_stats = {
            step_name: jnp.array(array) for step_name, array in elasticity_stats.items()
        }
        self.n_customer = n_customer
        self.n_category = n_category
        self.n_product = n_product
        self.product_category_mapping = product_category_mapping

    def _agg_stats(self, axis):
        """Compute the mean the elasticity or probability of the given axis."""
        category_choice_elasticity = self.elasticity_stats[
            "category_choice_elasticity"
        ].mean(axis=axis)
        product_choice_elasticity = self.elasticity_stats[
            "product_choice_elasticity"
        ].mean(axis=axis)
        product_choice_prob = self.prob_stats["product_choice_prob"].mean(axis=axis)
        product_demand_elasticity = self.elasticity_stats[
            "product_demand_elasticity"
        ].mean(axis=axis)
        product_demand = self.prob_stats["demand_mean"].mean(axis=axis)
        overall_elasticity = self.elasticity_stats["overall_elasticity"].mean(axis=axis)
        expected_demand = self.prob_stats["expected_demand"].mean(axis=axis)

        if axis == (0,) or axis == (0, 1):
            category_choice_prob = jnp.matmul(
                self.prob_stats["category_choice_prob"].mean(axis=axis),
                self.product_category_mapping.T,
            )
        elif axis == (0, 2):
            category_choice_prob = jnp.matmul(
                self.prob_stats["category_choice_prob"], self.product_category_mapping.T
            ).mean(axis=axis)

        return (
            category_choice_prob,
            category_choice_elasticity,
            product_choice_prob,
            product_choice_elasticity,
            product_demand,
            product_demand_elasticity,
            expected_demand,
            overall_elasticity,
        )

    def get_elasticity_per_customer_per_product(self):
        axis = (0,)
        (
            category_choice_prob,
            category_choice_elasticity,
            product_choice_prob,
            product_choice_elasticity,
            product_demand,
            product_demand_elasticity,
            expected_demand,
            overall_elasticity,
        ) = self._agg_stats(axis)
        # all result array has shape (n_customer, n_product)
        customer_list = jnp.arange(self.n_customer).tolist()
        product_list = jnp.arange(self.n_product).tolist()
        index = list(itertools.product(customer_list, product_list))
        index = pd.MultiIndex.from_tuples(index, names=["customer_key", "product_nbr"])
        category_nbr = jnp.where(self.product_category_mapping == 1)[1]
        category_nbr = jnp.concatenate([category_nbr] * self.n_customer)
        result = pd.DataFrame(
            {
                "category_nbr": category_nbr,
                "category_choice_elasticity": category_choice_elasticity.reshape(
                    -1,
                ),
                "category_choice_prob": category_choice_prob.reshape(
                    -1,
                ),
                "product_choice_elasticity": product_choice_elasticity.reshape(
                    -1,
                ),
                "product_choice_prob": product_choice_prob.reshape(
                    -1,
                ),
                "product_demand_elasticity": product_demand_elasticity.reshape(
                    -1,
                ),
                "product_demand": 1
                + product_demand.reshape(
                    -1,
                ),  # add 1 to account for shifted Poission
                "overall_elasticity": overall_elasticity.reshape(
                    -1,
                ),
                "expected_demand": expected_demand.reshape(
                    -1,
                ),
            },
            index=index,
        ).reset_index()

        return result

    def get_elasticity_per_week_per_customer_per_product(
        self,
        product_idx: int,
    ):
        (
            category_choice_prob,
            category_choice_elasticity,
            product_choice_prob,
            product_choice_elasticity,
            product_demand,
            product_demand_elasticity,
            expected_demand,
            overall_elasticity,
        ) = (
            self.prob_stats["category_choice_prob"][:, :, product_idx],
            self.elasticity_stats["category_choice_elasticity"][:, :, product_idx],
            self.prob_stats["product_choice_prob"][:, :, product_idx],
            self.elasticity_stats["product_choice_elasticity"][:, :, product_idx],
            1
            + self.prob_stats["demand_mean"][
                :, :, product_idx
            ],  # add 1 to account for shifted Poission
            self.elasticity_stats["product_demand_elasticity"][:, :, product_idx],
            self.prob_stats["expected_demand"][:, :, product_idx],
            self.elasticity_stats["overall_elasticity"][:, :, product_idx],
        )
        week_list = jnp.arange(category_choice_prob.shape[0]).tolist()
        customer_list = jnp.arange(self.n_customer).tolist()
        product_list = [product_idx]
        index = list(itertools.product(week_list, customer_list, product_list))
        index = pd.MultiIndex.from_tuples(
            index, names=["week", "customer_key", "product_nbr"]
        )
        result = pd.DataFrame(
            {
                "category_choice_elasticity": category_choice_elasticity.reshape(
                    -1,
                ),
                "category_choice_prob": category_choice_prob.reshape(
                    -1,
                ),
                "product_choice_elasticity": product_choice_elasticity.reshape(
                    -1,
                ),
                "product_choice_prob": product_choice_prob.reshape(
                    -1,
                ),
                "product_demand_elasticity": product_demand_elasticity.reshape(
                    -1,
                ),
                "product_demand": product_demand.reshape(
                    -1,
                ),
                "overall_elasticity": overall_elasticity.reshape(
                    -1,
                ),
                "expected_demand": expected_demand.reshape(
                    -1,
                ),
            },
            index=index,
        ).reset_index()

        return result


def initialize_elasticity_viz_data_layer(synthesizer: DataSynthesizer):
    # helper method to initialize a data layer object from a given synthesizer
    n_customer = synthesizer.n_customer
    n_category = synthesizer.n_category
    n_product = synthesizer.n_product
    product_category_mapping = synthesizer.product_category_mapping
    elasticity_stats = synthesizer.elasticity_stats
    prob_stats = synthesizer.choice_decision_stats

    dl = ElasticityVizDataLayer(
        n_customer,
        n_category,
        n_product,
        product_category_mapping,
        elasticity_stats,
        prob_stats,
    )

    return dl


def list_elasticity_one_step_heatmap(
    data,
    values: str,
    title: str,
    customer_list=None,
    product_list=None,
    category_nbr=None,
    ax=None,
    color=None,
    grid_line_width: float = 1,
):
    # plot the heatmap of the elasticities for specified customers and products
    if ax is None:
        ax = plt.gca()
    if category_nbr is not None:
        data = data[data["category_nbr"] == category_nbr]
    if customer_list is not None:
        data = data[data["customer_key"].isin(customer_list)]
    if product_list is not None:
        data = data[data["product_nbr"].isin(product_list)]
    pivot_data = data.pivot(index="product_nbr", columns="customer_key", values=values)
    sum_product_elasticity = pivot_data.sum(axis=1).sort_values()
    pivot_data = pivot_data.loc[sum_product_elasticity.index]
    pivot_data = pivot_data.loc[:, pivot_data.iloc[0].sort_values().index]
    sns.heatmap(
        pivot_data,
        cmap=color,
        ax=ax,
        cbar_kws={"label": "Price elasticity"},
        linewidths=grid_line_width,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("Customer")
    ax.set_ylabel("Product")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def list_elasticity_one_step_kdeplot(
    data,
    x: str,
    y: str,
    xlabel: str = "Probability",
    ylabel: str = "Elasticity",
    customer_key: list = [],
    ax=None,
    color=None,
    legend_loc: Optional[str] = "lower right",
    ylim_bottom: float = None,
    ylim_top: float = None,
    xlim_left: float = None,
    xlim_right: float = None,
    hue: str = None,
    log_scale: Tuple[bool, bool] = (True, False),
    bw=None,
    bw_adjust=1,
):
    # plot the relationship between choice probability and price elasticity
    # with kde approximation
    if ax is None:
        ax = plt.gca()
    data = data[data.customer_key.isin(customer_key)]
    ax = sns.kdeplot(
        data,
        x=x,
        y=y,
        palette=color,
        ax=ax,
        log_scale=log_scale,
        bw_adjust=bw_adjust,
        bw=bw,
        gridsize=500,
        fill=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if hue is not None:
        ax.legend(loc=legend_loc, title=hue)
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    ax.set_xlim(left=xlim_left, right=xlim_right)
    return ax


def list_elasticity_one_step_scatterplot(
    data,
    x: str,
    y: str,
    title: str,
    xlabel: str = "Probability",
    ylabel: str = "Elasticity",
    customer_key: list = [],
    ax=None,
    color=None,
    legend_loc: Optional[str] = "lower right",
    ylim_bottom: float = None,
    ylim_top: float = None,
    xlim_left: float = None,
    xlim_right: float = None,
    hue: str = None,
    scientific_axis: bool = False,
):
    # plot the relationship between choice probability and price elasticity
    # in pure scatter plot
    if ax is None:
        ax = plt.gca()
    data = data[data.customer_key.isin(customer_key)]
    sns.scatterplot(data, x=x, y=y, palette=color, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if hue is not None:
        ax.legend(loc=legend_loc, title=hue)
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    ax.set_xlim(left=xlim_left, right=xlim_right)
    if scientific_axis:
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    return ax


def examine_elasticity_per_customer_per_product(
    data: pd.DataFrame,
    color: str = None,
    row_size: int = 20,
    customer_key: list = [],
    product_nbr: int = 0,
    title: str = "Elasticity scatterplot summary",
    legend_loc: str = None,
    hue: str = None,
    ylim_bottom: list = [None, None, None, None],
    ylim_top: list = [None, None, None, None],
    xlim_left: list = [None, None, None, None],
    xlim_right: list = [None, None, None, None],
):
    # one-step method to plot elasticity for steps of category choice, product choice, product demand and overall demand
    assert "product_nbr" in data.columns
    # assert "category_nbr" in data.columns
    assert "customer_key" in data.columns
    fig, axes = plt.subplots(2, 2, figsize=(row_size, row_size))
    list_elasticity_one_step_kdeplot(
        data,
        "category_choice_prob",
        "category_choice_elasticity",
        xlabel="Conditional category choice probability",
        ylabel="Category choice elasticity",
        color=color,
        ax=axes[0][0],
        customer_key=customer_key,
        legend_loc=legend_loc,
        ylim_bottom=ylim_bottom[0],
        ylim_top=ylim_top[0],
        xlim_left=xlim_left[0],
        xlim_right=xlim_right[0],
        hue=hue,
    )

    list_elasticity_one_step_kdeplot(
        data,
        "product_choice_prob",
        "product_choice_elasticity",
        xlabel="Conditional product choice probability",
        ylabel="Product choice elasticity",
        color=color,
        ax=axes[0][1],
        customer_key=customer_key,
        legend_loc=legend_loc,
        ylim_bottom=ylim_bottom[1],
        ylim_top=ylim_top[1],
        xlim_left=xlim_left[1],
        xlim_right=xlim_right[1],
        hue=hue,
    )

    list_elasticity_one_step_kdeplot(
        data,
        "product_demand",
        "product_demand_elasticity",
        xlabel="Conditional quantity preference",
        ylabel="Quantity elasticity",
        color=color,
        ax=axes[1][0],
        customer_key=customer_key,
        legend_loc=legend_loc,
        ylim_bottom=ylim_bottom[2],
        ylim_top=ylim_top[2],
        xlim_left=xlim_left[2],
        xlim_right=xlim_right[2],
        hue=hue,
    )

    list_elasticity_one_step_kdeplot(
        data,
        "expected_demand",
        "overall_elasticity",
        xlabel="Overall demand mean",
        ylabel="Overall elasticity",
        color=color,
        ax=axes[1][1],
        customer_key=customer_key,
        legend_loc=legend_loc,
        ylim_bottom=ylim_bottom[3],
        ylim_top=ylim_top[3],
        xlim_left=xlim_left[3],
        xlim_right=xlim_right[3],
        hue=hue,
        log_scale=(True, False),
    )

    fig.suptitle(f"Product {product_nbr}: " + title)

    return axes
