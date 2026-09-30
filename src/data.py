"""
src/data.py
Data-access layer — mirrors Notebook 01 logic as reusable functions.

Responsibility: Read raw tables from PostgreSQL, aggregate, join, compute
geographic features. Returns a single DataFrame per order.

Requirement 2: Separate concerns — data access is isolated here.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DB_URL
from src.logger import get_logger

logger = get_logger(__name__)


# Haversine distance (same logic as Notebook 01)
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two lat/lon points."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Read helpers
def get_engine():
    """Create a SQLAlchemy engine from config."""
    logger.debug("Creating DB engine: %s", DB_URL.split("@")[-1])
    return create_engine(DB_URL)


def load_orders(engine=None) -> pd.DataFrame:
    """Load olist_orders_dataset."""
    if engine is None:
        engine = get_engine()
    logger.info("Loading orders table …")
    return pd.read_sql("SELECT * FROM olist_orders_dataset", engine)


def load_customers(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    logger.info("Loading customers table …")
    return pd.read_sql("SELECT * FROM olist_customers_dataset", engine)


def load_order_items(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    logger.info("Loading order_items table …")
    return pd.read_sql("SELECT * FROM olist_order_items_dataset", engine)


def load_order_payments(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    logger.info("Loading order_payments table …")
    return pd.read_sql("SELECT * FROM olist_order_payments_dataset", engine)


def load_products(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    logger.info("Loading products table …")
    return pd.read_sql("SELECT * FROM olist_products_dataset", engine)


def load_sellers(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    logger.info("Loading sellers table …")
    return pd.read_sql("SELECT * FROM olist_sellers_dataset", engine)


def load_geolocation(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    logger.info("Loading geolocation table …")
    return pd.read_sql("SELECT * FROM olist_geolocation_dataset", engine)


# Aggregation helpers (mirror Notebook 01)
def aggregate_order_items(items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order_items to one row per order_id."""
    logger.debug("Aggregating order_items …")
    return (
        items.groupby("order_id")
        .agg(
            order_items_count=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            total_weight_g=("product_weight_g", "sum"),
            total_volume_cm3=("product_volume_cm3", "sum"),
            seller_id=("seller_id", "first"),
            product_id=("product_id", "first"),)
        .reset_index())


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order_payments to one row per order_id."""
    logger.debug("Aggregating payments …")
    agg = (
        payments.groupby("order_id")
        .agg(
            total_payment_value=("payment_value", "sum"),
            max_payment_installments=("payment_installments", "max"),
            dominant_payment_type=("payment_type", lambda x: x.value_counts().index[0]),)
        .reset_index())
    return agg


def build_geo_lookup(geo: pd.DataFrame) -> pd.DataFrame:
    """Return one (lat, lon) per zip_code_prefix (deduplicated median)."""
    logger.debug("Building geolocation lookup …")
    return (
        geo.groupby("geolocation_zip_code_prefix")
        [["geolocation_lat", "geolocation_lng"]]
        .median()
        .reset_index()
        .rename(
            columns={
                "geolocation_zip_code_prefix": "zip_prefix",
                "geolocation_lat": "lat",
                "geolocation_lng": "lng",}))


# Master join — replicates Notebook 01 end-to-end
def build_joined_dataframe(engine=None) -> pd.DataFrame:
    """
    Read all tables and return the fully-joined orders DataFrame
    (same schema as artifacts/data/01_joined_orders.parquet).
    """
    if engine is None:
        engine = get_engine()

    orders = load_orders(engine)
    customers = load_customers(engine)
    items = load_order_items(engine)
    payments = load_order_payments(engine)
    products = load_products(engine)
    sellers = load_sellers(engine)
    geo = load_geolocation(engine)

    # Aggregate
    items_agg = aggregate_order_items(items)
    payments_agg = aggregate_payments(payments)
    geo_lookup = build_geo_lookup(geo)

    # Add product volume
    products["product_volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"])

    # Join items with products
    items_agg = items_agg.merge(
        products[["product_id", "product_volume_cm3"]],
        on="product_id",
        how="left",)
    # Fix total_volume_cm3 using actual product volume
    items_full = items.merge(
        products[["product_id", "product_volume_cm3"]],
        on="product_id",
        how="left",)
    vol_per_order = (
        items_full.groupby("order_id")["product_volume_cm3"].sum().reset_index()
        .rename(columns={"product_volume_cm3": "total_volume_cm3_recalc"}))
    
    items_agg = items_agg.merge(vol_per_order, on="order_id", how="left")
    items_agg["total_volume_cm3"] = items_agg["total_volume_cm3_recalc"].fillna(
        items_agg["total_volume_cm3"])
    items_agg.drop(columns=["total_volume_cm3_recalc", "product_id"], inplace=True, errors="ignore")

    # Join sellers with geo
    sellers = sellers.merge(
        geo_lookup,
        left_on="seller_zip_code_prefix",
        right_on="zip_prefix",
        how="left",).rename(columns={"lat": "seller_lat", "lng": "seller_lng"})

    # Join customers with geo
    customers = customers.merge(
        geo_lookup,
        left_on="customer_zip_code_prefix",
        right_on="zip_prefix",
        how="left",
    ).rename(columns={"lat": "customer_lat", "lng": "customer_lng"})

    # Join items with sellers
    items_agg = items_agg.merge(
        sellers[["seller_id", "seller_state", "seller_lat", "seller_lng"]],
        on="seller_id",
        how="left",)

    # Master join
    df = (
        orders
        .merge(customers[["customer_id", "customer_state", "customer_lat", "customer_lng"]], on="customer_id", how="left")
        .merge(items_agg, on="order_id", how="left")
        .merge(payments_agg, on="order_id", how="left"))

    # Parse timestamps
    ts_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",]
    for col in ts_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Compute derived columns
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days

    df["is_same_state"] = (
        (df["customer_state"] == df["seller_state"]).astype(int))

    df["distance_km"] = df.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lng"],
            r["seller_lat"], r["seller_lng"],)

        if all(pd.notna([r["customer_lat"], r["customer_lng"], r["seller_lat"], r["seller_lng"]]))
        else float("nan"),
        axis=1,)

    logger.info("Joined DataFrame: %d rows, %d cols", len(df), len(df.columns))
    return df
