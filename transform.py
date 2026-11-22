import pandas as pd


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


def validate_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_orders = orders
    orders = orders.copy()

    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["unit_price"] = pd.to_numeric(orders["unit_price"], errors="coerce")

    quantity = orders["quantity"]
    unit_price = orders["unit_price"]
    order_item_ids = orders["order_item_id"]

    error_masks = {
        "missing_order_item_id": order_item_ids.isna(),
        "missing_product_id": orders["product_id"].isna(),
        "invalid_quantity": quantity.isna() | (quantity <= 0),
        "invalid_unit_price": unit_price.isna() | (unit_price < 0),
        "duplicate_order_item_id": (
            order_item_ids.notna() & order_item_ids.duplicated(keep=False)
        ),
    }

    errors = pd.DataFrame(error_masks)
    invalid_mask = errors.any(axis=1)

    valid_orders = orders[~invalid_mask].copy()
    invalid_orders = original_orders[invalid_mask].copy()
    invalid_orders["validation_errors"] = [
        [name for name, failed in row.items() if failed]
        for row in errors.loc[invalid_mask].to_dict(orient="records")
    ]

    return valid_orders, invalid_orders


def validate_products(products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = products["id"]
    titles = products["title"].str.strip()
    categories = products["category"].str.strip()
    missing_id = ids.isna()

    error_masks = {
        "missing_id": missing_id,
        "duplicate_id": ~missing_id & ids.duplicated(keep=False),
        "invalid_title": titles.isna() | (titles == ""),
        "invalid_category": categories.isna() | (categories == ""),
    }

    errors = pd.DataFrame(error_masks)
    invalid_mask = errors.any(axis=1)

    valid_products = products[~invalid_mask].copy()
    valid_products["title"] = titles[~invalid_mask]
    valid_products["category"] = categories[~invalid_mask]
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
    unmatched_orders = orders_and_products.loc[
        orders_and_products["_merge"] == "left_only", orders.columns
    ].copy()
    unmatched_orders["validation_errors"] = [
        ["unmatched_product_id"] for _ in range(len(unmatched_orders))
    ]
    invalid_orders = pd.concat([invalid_orders, unmatched_orders], ignore_index=True)
    return processed_orders, invalid_orders, invalid_products
