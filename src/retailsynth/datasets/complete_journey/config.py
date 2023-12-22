"""Define the following dataclass to limit the data types of parameters and provide function to validate the range."""
from pydantic.dataclasses import dataclass


@dataclass
class Colnames:
    """A class representing the mapping between column names in the real datasets and the ones used in this codebase."""

    visit_date: str = "Visit_date"
    store_nbr: str = "STORE_ID"
    customer_key: str = "household_key"
    product_nbr: str = "PRODUCT_ID"
    sales_amt: str = "SALES_VALUE"
    item_qty: str = "QUANTITY"
    unit_price: str = "Unit_Price"
    basket_id: str = "BASKET_ID"
    discount_portion: str = "Discount_Portion"
    day: str = "DAY"
    week: str = "WEEK_NO"
    coupon_disc: str = "COUPON_DISC"
    coupon_match_disc: str = "COUPON_MATCH_DISC"
    retail_disc: str = "RETAIL_DISC"
    department: str = "DEPARTMENT"
    category_desc: str = "COMMODITY_DESC"
    subcategory_desc: str = "SUB_COMMODITY_DESC"

    @property
    def product_description_columns(self):
        """Return a list of product description-related column names."""
        return [self.department, self.category_desc, self.subcategory_desc]

    @property
    def replay_memory_columns(self):
        """Return a list of column names used for replay memory."""
        return [
            self.visit_date,
            self.product_nbr,
            self.discount_portion,
            self.unit_price,
            self.item_qty,
            self.sales_amt,
        ]


@dataclass
class Preprocess:
    """A class representing the preprocessing options for data preparation."""

    start_date: str = "2023-01-01"
    aggregation_level: str = "week"
    remove_gasoline_product: bool = False


@dataclass
class ProductAnnotation:
    include_annotation: bool = False
    min_num_txns: int = 100
    aggregation_level: str = "week"
    product_inactive_threshold: int = 10
    product_price_std_threshold: float = 1.0e-6
    product_price_dependence_significance_level: float = 0.05


@dataclass
class TimeSlices:
    train_test_ratio: float = 0.7
    start_week: int = 1
    end_week: int = 53
