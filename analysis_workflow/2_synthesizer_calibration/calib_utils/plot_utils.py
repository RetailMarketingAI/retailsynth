import altair as alt
from pandera import Column, DataFrameSchema, Index, MultiIndex

alt.themes.enable(
    "default"
)  # set your plotting theme https://github.com/vega/vega-themes/
alt.data_transformers.enable("vegafusion")

# Defining expected schemas for transaction and product dataframes
# Note that the dataframes can have additional columns; but they are ignored by this script
REQUIRED_TXN_SCHEMA = DataFrameSchema(
    columns={"item_qty": Column(int), "unit_price": Column(float)},
    index=MultiIndex(
        [
            Index(int, name="week"),
            Index(int, name="customer_key"),
            Index(int, name="product_nbr"),
        ]
    ),
)

REQUIRED_PRODUCT_SCHEMA = DataFrameSchema(
    columns={"category_desc": Column(str), "subcategory_desc": Column(str)},
    index=Index(int, name="product_nbr"),
)


def make_plot_data(transactions, products=None):
    """
    This utility function aggregates the transaction data to a weekly level. Each row of data
        is at a product-week level with the following features
            - total product demand normalized by the number of customers visiting the store by week  + change in this
            value from its mean across weeks
            - average unit price for the product for the week + change in this value from its mean across weeks
    Additional useful product metadata (category, subcategory) is also added to the dataframe

    Parameters
    ----------
    transactions: DataFrame of transactions output by the `run_preprocess` function
        with one row per customer transaction specifying week of purchase, product number,
        unit price and quantity purchased.
    products: DataFrame of products output by the `run_preprocess` function
        with one row per product with metadata about product category and subcategory

    Returns
    -------
    weekly_aggregates: DataFrame with demand and price features aggregated at product-week level

    """

    # Validate dataframes against specified schema
    transactions = REQUIRED_TXN_SCHEMA.validate(transactions)

    # calculate aggregate demand and average price per week for each product
    weekly_aggregates = (
        transactions.groupby(["week", "product_nbr"])
        .agg({"item_qty": "sum", "unit_price": "mean", "sales_amt": "sum"})
        .reset_index()
    )

    # calculate number of unique customers purchasing any item across all stores in a given week
    customer_visit = (
        transactions.reset_index()
        .groupby("week")["customer_key"]
        .nunique()
        .to_frame(name="customer_visit")
    )

    # calculate average price per product across weeks
    mean_price = (
        transactions.reset_index()
        .groupby(["product_nbr"])["unit_price"]
        .mean()
        .to_frame(name="avg_unit_price")
    )

    # calculate average normalized demand product across weeks
    normalized_item_qty_mean = (
        weekly_aggregates.groupby("product_nbr")["item_qty"]
        .mean()
        .to_frame(name="normalized_item_qty_mean")
    ) / customer_visit.mean().values
    normalized_sales_amt_mean = (
        weekly_aggregates.groupby("product_nbr")["sales_amt"]
        .mean()
        .to_frame(name="normalized_sales_amt_mean")
    ) / customer_visit.mean().values

    # join all the datasets
    weekly_aggregates = weekly_aggregates.merge(
        customer_visit, left_on="week", right_index=True
    )
    weekly_aggregates = weekly_aggregates.merge(
        mean_price, left_on="product_nbr", right_index=True
    )
    weekly_aggregates = weekly_aggregates.merge(
        normalized_item_qty_mean, left_on="product_nbr", right_index=True
    )
    weekly_aggregates = weekly_aggregates.merge(
        normalized_sales_amt_mean, left_on="product_nbr", right_index=True
    )

    # calculate price and demand change variables
    weekly_aggregates.loc[:, "normalized_item_qty"] = (
        weekly_aggregates.item_qty / weekly_aggregates.customer_visit
    )
    weekly_aggregates.loc[:, "normalized_sales_amt"] = (
        weekly_aggregates.sales_amt / weekly_aggregates.customer_visit
    )
    weekly_aggregates.loc[:, "unit_price_change"] = (
        weekly_aggregates.unit_price - weekly_aggregates.avg_unit_price
    ) / weekly_aggregates.avg_unit_price
    weekly_aggregates.loc[:, "normalized_item_qty_change"] = (
        weekly_aggregates.normalized_item_qty
        - weekly_aggregates.normalized_item_qty_mean
    ) / weekly_aggregates.normalized_item_qty_mean
    weekly_aggregates.loc[:, "normalized_sales_amt_change"] = (
        weekly_aggregates.normalized_sales_amt
        - weekly_aggregates.normalized_sales_amt_mean
    ) / weekly_aggregates.normalized_sales_amt_mean

    if products is not None:
        products = REQUIRED_PRODUCT_SCHEMA.validate(products)
        # add product metadata
        weekly_aggregates = weekly_aggregates.merge(
            products.loc[:, ["category_desc", "subcategory_desc"]],
            left_on="product_nbr",
            right_index=True,
        )
        weekly_aggregates.loc[:, "product_nbr"] = weekly_aggregates.product_nbr.astype(
            str
        )
        weekly_aggregates.loc[:, "product_description"] = (
            weekly_aggregates.loc[:, "category_desc"]
            + "-"
            + weekly_aggregates.loc[:, "subcategory_desc"]
            + ":"
            + weekly_aggregates.loc[:, "product_nbr"]
        )

    return weekly_aggregates


def make_exploratory_plot(
    weekly_aggregates,
    category,
    subcategory,
    plot_poly_order=2,
    y: str = "normalized_item_qty_change",
    ylabel="item_qty",
):
    """
    Helper function to plot the demand curve for a specific subcategory of products

    Parameters
    ----------
    weekly_aggregates: Dataframe output from the `make_plot_data` function
    category: string specifying the category label
    subcategory: string specifying the subcategory label
    plot_poly_order: int The polynomial order (number of coefficients) for the ‘poly’ method

    Returns
    -------
    altair plot object - scatter plot + curve showing change in demand as a function of change in price

    """

    plot_df = weekly_aggregates.query(f"category_desc=='{category}'").query(
        f"subcategory_desc == '{subcategory}'"
    )

    options = list(plot_df.product_nbr.astype("int").sort_values().unique().astype(str))
    labels = [option + " " for option in options]  # for spacing

    input_dropdown = alt.binding_select(
        # Add the empty selection which shows all when clicked
        options=options + [None],
        labels=labels + ["All"],
        name="Product Number: ",
    )

    selection = alt.selection_point(fields=["product_nbr"], bind=input_dropdown)

    base = (
        alt.Chart(plot_df, title="Demand Curve")
        .mark_point()
        .encode(
            x=alt.X("unit_price_change", scale=alt.Scale(domain=[-1.0, 1.0]))
            .axis(format="%")
            .title("Change in unit price %"),
            y=alt.Y(y)
            .axis(format="%")
            .title(
                ["Change in weekly Demand", f"(normalized {ylabel} by customer visits)"]
            ),
            color="product_nbr:N",
        )
    )

    polynomial_fit = [
        base.transform_regression(
            "unit_price_change",
            y,
            method="poly",
            order=plot_poly_order,
        ).mark_line()
    ]

    return (
        alt.layer(base, *polynomial_fit)
        .add_params(selection)
        .transform_filter(selection)
    )


def plot_demand_curve(
    weekly_aggregates,
    product_list,
    plot_poly_order=2,
    y: str = "normalized_item_qty_change",
    show_error: bool = False,
    title: str = "Demand Curve",
):
    """
    Helper function to plot the demand curve for a specific subset of products

    Parameters
    ----------
    weekly_aggregates: Dataframe output from the `make_plot_data` function
    product_list: list of product numbers
    plot_poly_order: int The polynomial order (number of coefficients) for the 'ploy' method

    Returns
    -------
    altair plot object - scatter plot + curve showing change in demand as a function of change in price
        for subset of product specified in product_list

    """

    plot_df = weekly_aggregates.loc[weekly_aggregates.product_nbr.isin(product_list), :]
    plot_df.loc[:, "unit_price_change"] = plot_df.loc[:, "unit_price_change"].round(
        decimals=1
    )

    plot_df_price_aggregated = (
        plot_df.groupby(["product_description", "unit_price_change"])
        .agg({y: ["mean", "std"]})
        .reset_index()
    )
    plot_df_price_aggregated.columns = [
        "_".join(col) if col[1] != "" else " ".join(col).strip()
        for col in plot_df_price_aggregated.columns.values
    ]

    base = (
        alt.Chart(plot_df, title=title)
        .mark_point()
        .encode(
            x=(
                alt.X("unit_price_change", scale=alt.Scale(domain=[-1.0, 1.0]))
                .axis(format="%")
                .title("Change in unit price %")
            ),
            y=(alt.Y(y).axis(format="%").title(["Change in weekly demand %"])),
            color=(
                alt.Color("product_description:N")
                .title("Product Description")
                .scale(scheme="magma")
            ),
        )
    )

    polynomial_fit = [
        base.transform_regression(
            "unit_price_change",
            y,
            method="poly",
            order=plot_poly_order,
            groupby=["product_description"],
        ).mark_line()
    ]

    point = (
        alt.Chart(plot_df_price_aggregated)
        .mark_point(opacity=0.4)
        .encode(
            alt.X("unit_price_change"),
            alt.Y(f"{y}_mean"),
            color=(
                alt.Color("product_description:N")
                .title("Product Description")
                .scale(scheme="dark2")
            ),
        )
    )

    bar = (
        alt.Chart(plot_df_price_aggregated)
        .mark_errorbar()
        .encode(
            x=alt.X("unit_price_change"),
            y=alt.Y(f"{y}_mean").title(""),
            yError=(f"{y}_std"),
            color=(
                alt.Color("product_description:N")
                .title("Product Description")
                .scale(scheme="dark2")
            ),
        )
    )

    if show_error:
        return alt.layer(*polynomial_fit) + point + bar
    else:
        return alt.layer(*polynomial_fit) + point


def plot_price_history(
    weekly_aggregates, product_list, category="", color_scheme="magma"
):
    """
    Generates a plot of the price history for a given list of products.

    Parameters
    ----------
    weekly_aggregates : pandas.DataFrame
        The weekly aggregates data containing price information.
    product_list : list
        The list of product numbers to be included in the plot.
    category : str, optional
        The category of products to filter by. Defaults to an empty string.

    Returns
    -------
    plot : alt.Chart
        The plot object representing the price history for the given products.
    """
    plot_df = weekly_aggregates.loc[weekly_aggregates.product_nbr.isin(product_list), :]
    plot = (
        alt.Chart(plot_df, title=f"Price history for {category}")
        .mark_line(size=2)
        .encode(
            x=alt.X("week").title("Week"),
            y=alt.Y(
                "unit_price",
            ).title("Unit Price"),
            color=alt.Color("product_nbr:O")
            .scale(scheme=color_scheme)
            .title("Product Number"),
        )
        .properties(width=500, height=150)
    )

    return plot
