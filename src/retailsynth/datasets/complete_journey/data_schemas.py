"""Schemas for required datasets and fields."""
import pandera as pa

CUSTOMER_SCHEMA = pa.DataFrameSchema(
    {
        "customer_key": pa.Column(pa.Int),
    }
)

TRANSACTION_SCHEMA = pa.DataFrameSchema(
    {
        "customer_key": pa.Column(pa.Int),
        "product_nbr": pa.Column(pa.Int),
        "item_qty": pa.Column(pa.Int, checks=pa.Check.ge(0)),
        "sales_amt": pa.Column(pa.Float, checks=pa.Check.ge(0)),
        "retail_disc": pa.Column(pa.Float, checks=pa.Check.le(0)),
        "coupon_disc": pa.Column(pa.Float, checks=pa.Check.le(0)),
        "coupon_match_disc": pa.Column(pa.Float, checks=pa.Check.le(0)),
        "week": pa.Column(pa.Int, checks=pa.Check.greater_than(0)),
        "day": pa.Column(pa.Int, checks=pa.Check.greater_than(0)),
    }
)

PRODUCT_SCHEMA = pa.DataFrameSchema(
    {
        "product_nbr": pa.Column(pa.Int),
    }
)
