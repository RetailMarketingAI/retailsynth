import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandera import Column, DataFrameSchema, Index, MultiIndex

REQUIRED_TXN_SCHEMA = DataFrameSchema(
    columns={
        "item_qty": Column(float),
        "unit_price": Column(float),
        "discount_portion": Column(float),
    },
    index=MultiIndex(
        [
            Index(int, name="week"),
            Index(int, name="customer_key"),
            Index(int, name="product_nbr"),
        ]
    ),
)

REQUIRED_PRODUCT_SCHEMA = DataFrameSchema(
    columns={"category_nbr": Column(int)},
    index=Index(int, name="product_nbr"),
)


def prepare_aggregate_data(transactions, products):
    # merge the transactions and the products table to get a giant info table
    transactions = REQUIRED_TXN_SCHEMA.validate(transactions)
    products = REQUIRED_PRODUCT_SCHEMA.validate(products)

    transactions.loc[:, "sales_amt"] = (
        transactions["item_qty"] * transactions["unit_price"]
    )
    trx = transactions.reset_index().merge(
        products, left_on="product_nbr", right_index=True
    )
    return trx


def convert_to_transition_matrix(synthesizer):
    # convert the transition probability to a transition matrix
    transition_prob = synthesizer.transition_prob.mean.mean(axis=0)
    transition_matrix = jnp.array(
        [
            [1 - transition_prob[0], transition_prob[0]],
            [1 - transition_prob[1], transition_prob[1]],
        ]
    )
    return transition_matrix


def compute_steady_state(transition_matrix):
    # Transpose the matrix and find its eigenvalues and eigenvectors
    transposed_matrix = jnp.transpose(transition_matrix)
    eigenvalues, eigenvectors = jnp.linalg.eig(transposed_matrix)

    # Find the index of the eigenvalue closest to 1 (within a small threshold)
    steady_state_index = jnp.argmin(jnp.abs(eigenvalues - 1.0))

    # Extract the corresponding eigenvector as the steady state vector
    steady_state = jnp.abs(eigenvectors[:, steady_state_index])
    steady_state /= jnp.sum(steady_state)  # Normalize the probabilities to sum up to 1

    return steady_state


def compute_discount_depth(synthesizer):
    return synthesizer.discount_depth_distribution.mean


def compute_discount_level(synthesizer):
    return jnp.array(synthesizer.choice_decision_stats["discount"]).mean().item()


def get_discount_summary_df(synthesizers: list, labels: list):
    # Create a summary df to report the offers of different scenarios
    df = []
    for synthesizer, label in zip(synthesizers, labels):
        transition_matrix = convert_to_transition_matrix(synthesizer)
        steady_state = compute_steady_state(transition_matrix)
        discount_frequency = round(steady_state[1].item(), 2)
        discount_depth = compute_discount_depth(synthesizer)
        discount_level = compute_discount_level(synthesizer)
        df.append([discount_frequency, discount_depth, discount_level, label])
    discount_df = pd.DataFrame(
        df,
        columns=["discount_frequency", "discount_depth", "discount_level", "policy"],
    )
    return discount_df


def plot_discount_state_proportion(
    synthesizers: list, synthesizers_names: list, ax=None
):
    """compare proportion of products in discounts for different pricing strategies

    Parameters
        synthesizers (list): list of synthesizer running different pricing strategies
        synthesizers_names (list): list of names of different pricing strategies
    """
    discount_state_proposition = pd.DataFrame()
    for synthesizer, name in zip(synthesizers, synthesizers_names):
        discount_prop = jnp.array(
            synthesizer.choice_decision_stats["discount_state"]
        ).mean(axis=(1))
        discount_state_proposition.loc[:, name] = discount_prop
    if ax is None:
        ax = plt.gca()
    sns.lineplot(discount_state_proposition, ax=ax)
    plt.xlabel("Week")
    plt.ylabel("Proportion")
    plt.title("Products in discount state")
    return ax


def plot_discount_distribution(
    synthesizers: list, synthesizers_names: list, ax=None, binwidth=0.05
):
    """plot the actual discount that the synthesizer generate internally

    Parameters
    ----------
        synthesizers (list): list of synthesizer running different pricing strategies
        synthesizers_names (list): list of names of different pricing strategies
    """
    discounts = []
    for synthesizer, name in zip(synthesizers, synthesizers_names):
        discount = jnp.array(synthesizer.choice_decision_stats["discount"])
        discount = jnp.nanmean(jnp.where(discount > 0, discount, jnp.nan), axis=0)
        discount = pd.DataFrame({"discount": discount, "label": name})
        discounts.append(discount)
    discounts = pd.concat(discounts).reset_index()
    if ax is None:
        ax = plt.gca()
    sns.histplot(
        data=discounts,
        binwidth=binwidth,
        stat="probability",
        x="discount",
        hue="label",
    )
    ax.set_title("Discount depth")
    return ax


def plot_price_distribution(
    synthesizers: list, labels: list, xlim_min: int = 0, xlim_max: int = 10, ax=None
):
    # draw the price distribution that the synthesizer offered as a kde plot
    stat_dfs = []
    for synthesizer, label in zip(synthesizers, labels):
        result = jnp.array(synthesizer.choice_decision_stats["product_price"]).mean(
            axis=0
        )
        result = pd.DataFrame({"unit_price": result, "policy": label})
        stat_dfs.append(result)
    stat_dfs = pd.concat(stat_dfs)
    if ax is None:
        ax = plt.gca()
    sns.kdeplot(data=stat_dfs, x="unit_price", alpha=1, hue="policy", ax=ax)
    ax.set_xlabel("Product price")
    ax.set_title("Price distribution by pricing strategy")
    ax.set_xlim(xlim_min, xlim_max)
    return ax


def plot_purchase_summary_for_all_customers(
    reports: pd.DataFrame,
    row_size: int = 18,
    column_size: int = 3,
    policy_plotting_order: list = None,
    axes: list = None,
    label_args={},
    title_args={},
):
    assert "total_revenue" in reports.columns
    assert "total_item_qty" in reports.columns
    assert "final_week_retention_rate" in reports.columns
    assert "weekly_avg_category_sold_per_customer" in reports.columns
    assert "effective_discount" in reports.columns
    assert "discount_level" in reports.columns

    if axes is None:
        fig, axes = plt.subplots(1, 5, figsize=(row_size, column_size))

    sns.barplot(
        data=reports,
        x="policy",
        y="effective_discount",
        order=policy_plotting_order,
        ax=axes[0],
    )
    axes[0].set_title("Average Discount", **title_args)
    axes[0].set_ylabel("Realized discount", **label_args)
    axes[0].set_xlabel("Policy", **label_args)

    sns.barplot(
        data=reports,
        x="policy",
        y="total_item_qty",
        order=policy_plotting_order,
        ax=axes[1],
    )
    axes[1].set_title("Demand", **title_args)
    axes[1].set_ylabel("Quantity sold", **label_args)
    axes[1].ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
    axes[1].set_xlabel("Policy", **label_args)

    sns.barplot(
        data=reports,
        x="policy",
        y="total_revenue",
        order=policy_plotting_order,
        ax=axes[2],
    )
    axes[2].set_title("Revenue", **title_args)
    axes[2].set_ylabel("Revenue", **label_args)
    axes[2].ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
    axes[2].set_xlabel("Policy", **label_args)

    sns.barplot(
        data=reports,
        x="policy",
        y="weekly_avg_category_sold_per_customer",
        order=policy_plotting_order,
        ax=axes[3],
    )
    axes[3].set_title("Category Penetration", **title_args)
    axes[3].set_ylabel("Categories sold", **label_args)
    axes[3].set_xlabel("Policy", **label_args)

    sns.barplot(
        data=reports,
        x="policy",
        y="final_week_retention_rate",
        order=policy_plotting_order,
        ax=axes[4],
    )
    axes[4].set_title("Customer Retention", **title_args)
    axes[4].set_ylabel("Retention rate", **label_args)
    axes[4].set_xlabel("Policy", **label_args)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fig = axes[0].get_figure()

    return fig, axes
