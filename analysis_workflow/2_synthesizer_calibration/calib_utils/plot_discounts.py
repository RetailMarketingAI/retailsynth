import itertools

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def prepare_discount_df(synthesizer, p_index):
    # query the discount array for specified products from synthesizer and plot the price
    # no matter whether the customer purchase this product or not
    discount = jnp.array(synthesizer.choice_decision_stats["discount"])[:, p_index]
    discount_state = jnp.array(synthesizer.choice_decision_stats["discount_state"])[
        :, p_index
    ]
    price = jnp.array(synthesizer.choice_decision_stats["product_price"])[:, p_index]

    index = list(itertools.product(range(discount.shape[0]), p_index))
    index = pd.MultiIndex.from_tuples(index, names=["week", "product_nbr"])
    result = pd.DataFrame(columns=["discount", "discount_state", "price"], index=index)
    result.loc[:, "discount"] = discount.flatten()
    result.loc[:, "discount_state"] = discount_state.flatten()
    result.loc[:, "price"] = price.flatten()

    return result


def plot_discount_series(df):
    # plot the discount states, discount depth, and actual price in a time series
    fig, axes = plt.subplots(3, 1, figsize=(10, 7))
    sns.lineplot(df, x="week", y="discount_state", hue="product_nbr", ax=axes[0])
    axes[0].set_xlabel("week")
    axes[0].set_ylabel("discount_state")
    sns.lineplot(df, x="week", y="discount", hue="product_nbr", ax=axes[1])
    axes[1].set_xlabel("week")
    axes[1].set_ylabel("discount")
    sns.lineplot(df, x="week", y="price", hue="product_nbr", ax=axes[2])
    axes[2].set_xlabel("week")
    axes[2].set_ylabel("price")
    fig.suptitle("Discount and price over time")
    return axes


def plot_discount_depth(df):
    # plot the discount distribution
    fig, axes = plt.subplots(1, 1, figsize=(7, 3))
    df = df.query("discount_state > 0")
    sns.histplot(df, x="discount", hue="product_nbr", ax=axes, kde=True)
    axes.set_xlabel("discount")
    axes.set_ylabel("density")
    axes.set_xlim(
        0,
    )
    fig.suptitle("Discount distribution")
    return axes


def plot_status_transition(discount_state, n_product: int = -1):
    # plot the composition of discount states VS no discount state
    # in a time series
    fig, axes = plt.subplots(1, 1, figsize=(10, 3))
    with_discount = discount_state[:, :n_product].mean(axis=1)
    no_discount = 1 - with_discount
    with_discount = pd.DataFrame(
        {
            "prob": with_discount,
            "label": "with_discount",
            "week": range(len(with_discount)),
        }
    )
    no_discount = pd.DataFrame(
        {"prob": no_discount, "label": "no_discount", "week": range(len(with_discount))}
    )
    sns.lineplot(
        data=pd.concat([with_discount, no_discount]),
        x="week",
        y="prob",
        hue="label",
        ax=axes,
    )
    axes.set_xlabel("week")
    axes.set_ylabel("probability")
    fig.suptitle("State composition")
    return axes
