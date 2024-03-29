---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python pycharm={"name": "#%%\n"}
import logging
import warnings

from hydra import compose, initialize
from omegaconf import OmegaConf

from retailsynth.datasets.complete_journey.preprocess_pipeline import (
    run_preprocess,
    PreprocessPipeline,
)
from retailsynth.base_config import load_config_store

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

warnings.filterwarnings("ignore", category=FutureWarning)
```

<!-- #region pycharm={"name": "#%% md\n"} -->

## Introduction

This notebook provides a deep dive exploratory analysis and documentation of the data preprocessing and cleaning steps
required to leverage The Complete Journey dataset.[1] We note that several demand prediction papers[2][3] have used this
same dataset and provide additional insights on the data quality beyond what is presented here.


## Data loading

We begin by instantiating the pipeline and downloading the raw data. We use three main tables: transactions, customer
demographics and product information. For more information about this data source, please refer to the Dunnhumby
website.

<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
with initialize(version_base=None, config_path="cfg"):
    load_config_store()
    cfg = compose(config_name="real_dataset")
    cfg = OmegaConf.to_object(cfg)

preprocess_pipeline = PreprocessPipeline(
    raw_data_config=cfg.raw_data,
)
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
customers = preprocess_pipeline.dataset.customers
transactions = preprocess_pipeline.dataset.transactions
products = preprocess_pipeline.dataset.products
```

<!-- #region pycharm={"name": "#%% md\n"} -->

## Data cleaning

In this section, we will describe the contents of each table and the cleaning steps we implemented to mitigate data
quality issues.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->

### Customer table cleaning

The `customers` table holds demographic attributes for each customer. Here is a sample of the data:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
customers.set_index("customer_key").head(2)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
However, in the transactions table there are orders attributed to customers customers not recorded in the demographic
table. With `record_unrecognized_customer`, we add these customers to the customer table.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# complete list customers in our record
recorded_customers = customers.customer_key.unique()
# transactions from customer not in our record
print(
    "Number of transactions from unrecognizable customers: ",
    transactions[~transactions.customer_key.isin(recorded_customers)].shape[0],
)
_ = preprocess_pipeline.record_unrecognized_customer()
```

<!-- #region pycharm={"name": "#%% md\n"} -->

### Product table cleaning

The `products` table contains information about products, where the `product_nbr` column serves as the unique
identifier. The table provides information, including product department, category description, subcategory description,
manufacturer, brand, and package size.

There are groups of products that share the same attributes but have different product IDs. These are likely very
similar from the customers' perspective. We show an example below:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
package_size = "13.7 OZ"
manufacturer_id = 6380
subcategory_desc = "TOMATOES: STEWED/DICED/CRMD"

products[
    (products.package_size == package_size)
    & (products.manufacturer_id == manufacturer_id)
    & (products.subcategory_desc == subcategory_desc)
]
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As the digital twin simulation focuses on the customer perspective, we overwrite the duplicate `product_nbrs` with the
first unique found in the product table.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
_ = preprocess_pipeline.drop_duplicate_product_id()
```

```python pycharm={"name": "#%%\n"}
products = preprocess_pipeline.dataset.products
products[
    (products.package_size == package_size.lower())
    & (products.manufacturer_id == manufacturer_id)
    & (products.subcategory_desc == subcategory_desc.lower())
]
```

<!-- #region pycharm={"name": "#%% md\n"} -->

### Transaction table cleaning

The `transactions` table contains records of each customer's purchase behavior, providing information such as
transaction date, discount policies, quantity sold, and more. The original dataset includes three sources of discount:
retail discount (from a loyalty card program), coupon discount (supplied by the manufacturer), and coupon match
discount (supplied by the retailer's match of the manufacturer coupon).
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
transactions.head(5)
```

<!-- #region pycharm={"name": "#%% md\n"} -->

#### 1. Product price calculation

The original dataset does not provide the discount portion. Thus we implement a helper method to compute the actual
dealt price, which includes unit price (after applying all discounts), and empirical discount percentage.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
_ = preprocess_pipeline.add_pricing_columns()
transactions = preprocess_pipeline.dataset.transactions
columns_in_use = [
    "customer_key",
    "product_nbr",
    "week",
    "item_qty",
    "sales_amt",
    "unit_price",
    "discount_portion",
]
transactions.loc[:, columns_in_use].head(5)
```

<!-- #region pycharm={"name": "#%% md\n"} -->

#### 2. Invalid transactions

To ensure that every row in the `transactions` table contains meaningful data, we need to inspect the values in each
column to determine if they are interpretable in the context of a purchase event. However, we have observed transactions
with negative quantity sold and negative money spent. These records likely represent product returns instead of actual
purchases or valid store visits.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
transactions[(transactions.item_qty <= 0) | (transactions.sales_amt <= 0)].head(2)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We use the following method to filter out these transactions and remove customer and product records that are not
associated with any valid transactions.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
_ = preprocess_pipeline.clean_transactions()
```

<!-- #region pycharm={"name": "#%% md\n"} -->

### Putting it all together

<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
We have encapsulated all filters introduced in this method into a helper method called `run_preprocess`. The method
signature is shown below.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
help(run_preprocess)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The following codes run all preprocessing steps described above, and outputs the customer demographic table, the product
hierarchy table, and the transaction table.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
customers, products, transactions = run_preprocess(cfg)
```

<!-- #region pycharm={"name": "#%% md\n"} -->

## References

1. Dunnhumby Source Files. https://www.dunnhumby.com/source-files/
2. Maasakkers, et al. Next-basket prediction in a high-dimensional setting using gated recurrent units, Expert Systems
   with Applications, Volume 212, 2023, 118795, ISSN 0957-4174. https://doi.org/10.1016/j.eswa.2022.118795.
3. Ariannezhad, Mozhdeh, et al. "ReCANet: A Repeat Consumption-Aware Neural Network for Next Basket Recommendation in
   Grocery Shopping." Proceedings of the 45th International ACM SIGIR Conference on Research and Development in
   Information Retrieval. 2022.
<!-- #endregion -->
