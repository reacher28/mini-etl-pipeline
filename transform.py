import pandas as pd
import numpy as np
import re
from numbers import Integral
from decimal import Decimal, InvalidOperation
from datetime import date


def merge_orders_products(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    products_to_merge = products.loc[:, ["id", "title", "category"]]
    products_to_merge = products_to_merge.rename(
        columns={"id": "product_id", "title": "product_name"}
    )
    result = orders.merge(
        products_to_merge,
        on="product_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    result["revenue"] = result["quantity"] * result["unit_price"]
    return result


def parse_unit_price(value: object) -> Decimal | None:
    if isinstance(value, (bool, np.bool_)):
        return None

    try:
        price = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None

    if not price.is_finite():
        return None

    if not Decimal("0") <= price <= Decimal("9999999999.99"):
        return None

    try:
        normalized_price = price.quantize(Decimal("0.01"))
    except InvalidOperation:
        return None

    if price != normalized_price:
        return None

    return normalized_price


def parse_order_date(value: object) -> date | None:
    if type(value) is date:
        return value

    if not isinstance(value, str):
        return None

    text = value.strip()

    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text) is None:
        return None

    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def parse_bigint(value: object) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None

    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?[0-9]+", text) is None:
            return None

        digits = text.lstrip("+-").lstrip("0") or "0"
        if len(digits) > 19:
            return None

        number = int(digits)
        if text.startswith("-"):
            number = -number
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or value % 1 != 0:
            return None
        number = int(value)
    else:
        return None

    return number if -(2**63) <= number <= (2**63 - 1) else None


def parse_bigint_column(column: pd.Series) -> pd.Series:
    return pd.Series(
        [parse_bigint(value) for value in column],
        index=column.index,
        dtype=object,
    ).astype("Int64")


def validate_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_orders = orders
    orders = orders.copy()

    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["unit_price"] = orders["unit_price"].map(parse_unit_price)
    orders["date"] = orders["date"].map(parse_order_date)

    quantity = orders["quantity"]
    unit_price = orders["unit_price"]
    raw_order_item_ids = original_orders["order_item_id"]
    raw_order_ids = original_orders["order_id"]
    raw_product_ids = original_orders["product_id"]

    order_item_ids = parse_bigint_column(raw_order_item_ids)
    order_ids = parse_bigint_column(raw_order_ids)
    product_ids = parse_bigint_column(raw_product_ids)

    missing_order_item_id = raw_order_item_ids.isna()
    missing_order_id = raw_order_ids.isna()
    missing_product_id = raw_product_ids.isna()

    invalid_order_item_id = ~missing_order_item_id & (
        order_item_ids.isna() | order_item_ids.le(0).fillna(False)
    )

    invalid_order_id = ~missing_order_id & (
        order_ids.isna() | order_ids.le(0).fillna(False)
    )

    invalid_product_id = ~missing_product_id & (
        product_ids.isna() | product_ids.le(0).fillna(False)
    )

    invalid_quantity = (
        quantity.isna()
        | (quantity <= 0)
        | ~np.isfinite(quantity)
        | (quantity % 1 != 0)
        | (quantity > 2**31 - 1)
    )

    invalid_unit_price = unit_price.isna()

    max_revenue = Decimal("999999999999.99")

    invalid_revenue = pd.Series(
        [
            (False if bad_quantity or bad_price else int(qty) * price > max_revenue)
            for qty, price, bad_quantity, bad_price in zip(
                quantity,
                unit_price,
                invalid_quantity,
                invalid_unit_price,
                strict=True,
            )
        ],
        index=orders.index,
        dtype=bool,
    )

    error_masks = {
        "missing_order_item_id": missing_order_item_id,
        "invalid_order_item_id": invalid_order_item_id,
        "missing_order_id": missing_order_id,
        "invalid_order_id": invalid_order_id,
        "missing_product_id": missing_product_id,
        "invalid_product_id": invalid_product_id,
        "invalid_quantity": invalid_quantity,
        "invalid_unit_price": invalid_unit_price,
        "invalid_revenue": invalid_revenue,
        "duplicate_order_item_id": (
            order_item_ids.notna() & order_item_ids.duplicated(keep=False)
        ),
        "invalid_date": orders["date"].isna(),
    }
    orders["order_item_id"] = order_item_ids
    orders["order_id"] = order_ids
    orders["product_id"] = product_ids

    errors = pd.DataFrame(error_masks)
    invalid_mask = errors.any(axis=1)

    valid_orders = orders[~invalid_mask].copy()
    valid_orders["quantity"] = valid_orders["quantity"].astype("Int64")
    invalid_orders = original_orders[invalid_mask].copy()
    invalid_orders["validation_errors"] = [
        [name for name, failed in row.items() if failed]
        for row in errors.loc[invalid_mask].to_dict(orient="records")
    ]

    return valid_orders, invalid_orders


def validate_products(
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    original_ids = products["id"]
    ids = parse_bigint_column(original_ids)

    titles = products["title"].str.strip()
    categories = products["category"].str.strip()

    missing_id = original_ids.isna()

    invalid_id = ~missing_id & (ids.isna() | ids.le(0).fillna(False))

    error_masks = {
        "missing_id": missing_id,
        "invalid_id": invalid_id,
        "duplicate_id": (ids.notna() & ids.duplicated(keep=False)),
        "invalid_title": titles.isna() | (titles == ""),
        "invalid_category": categories.isna() | (categories == ""),
    }

    errors = pd.DataFrame(error_masks)
    invalid_mask = errors.any(axis=1)

    valid_products = products[~invalid_mask].copy()

    valid_products["id"] = ids.loc[~invalid_mask]
    valid_products["title"] = titles.loc[~invalid_mask]
    valid_products["category"] = categories.loc[~invalid_mask]

    invalid_products = products[invalid_mask].copy()

    invalid_products["validation_errors"] = [
        [name for name, failed in row.items() if failed]
        for row in errors.loc[invalid_mask].to_dict(orient="records")
    ]

    return valid_products, invalid_products


def transform_data(
    orders: pd.DataFrame, products: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_orders, invalid_orders = validate_orders(orders)
    valid_products, invalid_products = validate_products(products)

    orders_and_products = merge_orders_products(valid_orders, valid_products)

    processed_orders = orders_and_products[
        orders_and_products["_merge"] == "both"
    ].copy()

    processed_orders = processed_orders.drop(columns="_merge")

    unmatched_valid_orders = orders_and_products.loc[
        orders_and_products["_merge"] == "left_only"
    ]

    raw_order_item_ids = parse_bigint_column(orders["order_item_id"])

    source_positions = {
        int(order_item_id): position
        for position, order_item_id in enumerate(raw_order_item_ids)
        if pd.notna(order_item_id)
    }

    unmatched_positions = [
        source_positions[int(order_item_id)]
        for order_item_id in unmatched_valid_orders["order_item_id"]
    ]

    unmatched_orders = orders.iloc[unmatched_positions].copy()

    unmatched_orders["validation_errors"] = [
        ["unmatched_product_id"] for _ in range(len(unmatched_orders))
    ]

    invalid_orders = pd.concat(
        [invalid_orders, unmatched_orders],
        ignore_index=True,
    )

    return processed_orders, invalid_orders, invalid_products
