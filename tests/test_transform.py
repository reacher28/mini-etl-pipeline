from transform import validate_products, transform_data, validate_orders
import pandas as pd
from pandas.api.types import is_numeric_dtype


def test_validate_products_multiple_errors():
    products = pd.DataFrame([{"id": None, "title": "   ", "category": None}])

    valid, invalid = validate_products(products)

    assert valid.empty
    assert len(invalid) == 1
    assert set(invalid["validation_errors"].iloc[0]) == {
        "missing_id",
        "invalid_title",
        "invalid_category",
    }


def test_validate_products_duplicate_ids():
    products = pd.DataFrame(
        [
            {"id": 1, "title": "Keyboard", "category": "Electronics"},
            {"id": 2, "title": "Mouse", "category": "Electronics"},
            {"id": 1, "title": "Monitor", "category": "Electronics"},
        ]
    )
    valid, invalid = validate_products(products)

    assert len(valid) == 1 and valid.iloc[0]["id"] == 2
    assert invalid["id"].tolist() == [1, 1]
    assert all("duplicate_id" in errors for errors in invalid["validation_errors"])


def test_transform_data():
    orders = pd.DataFrame(
        [
            {
                "order_item_id": 501,
                "order_id": 101,
                "product_id": 5,
                "quantity": 2,
                "unit_price": 2500,
                "date": "2026-11-01",
            },
            {
                "order_item_id": 502,
                "order_id": 101,
                "product_id": 8,
                "quantity": 1,
                "unit_price": 1200,
                "date": "2026-11-01",
            },
            {
                "order_item_id": 503,
                "order_id": 102,
                "product_id": 5,
                "quantity": 3,
                "unit_price": 2500,
                "date": "2026-11-02",
            },
            {
                "order_item_id": 504,
                "order_id": 103,
                "product_id": 999,
                "quantity": 1,
                "unit_price": 3500,
                "date": "2026-11-03",
            },
            {
                "order_item_id": 505,
                "order_id": 104,
                "product_id": 9,
                "quantity": 1,
                "unit_price": 9000,
                "date": "2026-11-04",
            },
            {
                "order_item_id": 506,
                "order_id": 105,
                "product_id": 8,
                "quantity": -1,
                "unit_price": 1000,
                "date": "2026-11-05",
            },
            {
                "order_item_id": 507,
                "order_id": 106,
                "product_id": None,
                "quantity": 1,
                "unit_price": 700,
                "date": "2026-11-06",
            },
            {
                "order_item_id": 508,
                "order_id": 107,
                "product_id": 6,
                "quantity": 2,
                "unit_price": 3000,
                "date": "2026-11-07",
            },
            {
                "order_item_id": 509,
                "order_id": 108,
                "product_id": 10,
                "quantity": 1,
                "unit_price": 1300,
                "date": "2026-11-08",
            },
            {
                "order_item_id": 510,
                "order_id": 109,
                "product_id": 5,
                "quantity": 0,
                "unit_price": -100,
                "date": "2026-11-09",
            },
            {
                "order_item_id": 511,
                "order_id": 110,
                "product_id": 5,
                "quantity": 1,
                "unit_price": 0,
                "date": "2026-11-10",
            },
            {
                "order_item_id": 512,
                "order_id": 111,
                "product_id": 11,
                "quantity": 1,
                "unit_price": 2000,
                "date": "2026-11-11",
            },
        ]
    )
    products = pd.DataFrame(
        [
            {"id": 5, "title": "Keyboard", "category": "Electronics"},
            {"id": 8, "title": "Mouse", "category": "Electronics"},
            {"id": 9, "title": "Monitor", "category": "Electronics"},
            {"id": 6, "title": "Tablet", "category": "Electronics"},
            {"id": 6, "title": "Phone", "category": "Electronics"},
            {"id": 10, "title": "   ", "category": "Electronics"},
            {"id": 11, "title": "Headphones", "category": None},
        ]
    )
    processed, invalid_orders, invalid_products = transform_data(orders, products)
    assert len(processed) == 5
    assert len(invalid_orders) == 7
    assert len(invalid_products) == 4
    assert processed["revenue"].sum() == 22700
    assert len(processed) + len(invalid_orders) == len(orders)
    assert "_merge" not in processed.columns
    assert set(processed["order_item_id"]) == {501, 502, 503, 505, 511}
    errors_by_item = invalid_orders.set_index("order_item_id")["validation_errors"]
    for order_item_id in (504, 508, 509, 512):
        assert "unmatched_product_id" in errors_by_item.at[order_item_id], order_item_id

    assert set(errors_by_item.at[510]) == {"invalid_quantity", "invalid_unit_price"}


def test_validate_orders_numeric_conversion():
    orders = pd.DataFrame(
        [
            {
                "order_item_id": 1,
                "product_id": 5,
                "quantity": "2",
                "unit_price": "100.5",
            },
            {
                "order_item_id": 2,
                "product_id": 5,
                "quantity": "unknown",
                "unit_price": "-50",
            },
        ]
    )

    valid_orders, invalid_orders = validate_orders(orders)

    assert valid_orders["order_item_id"].tolist() == [1]
    assert valid_orders["quantity"].iloc[0] == 2
    assert valid_orders["unit_price"].iloc[0] == 100.5
    assert is_numeric_dtype(valid_orders["quantity"])
    assert is_numeric_dtype(valid_orders["unit_price"])

    assert invalid_orders["order_item_id"].tolist() == [2]
    assert invalid_orders["quantity"].iloc[0] == "unknown"
    errors = invalid_orders["validation_errors"].iloc[0]
    assert "invalid_quantity" in errors
    assert "invalid_unit_price" in errors
    assert orders["quantity"].tolist() == ["2", "unknown"]
    assert orders["unit_price"].tolist() == ["100.5", "-50"]


def test_validate_orders_duplicate_and_missing_item_ids():
    orders = pd.DataFrame(
        {
            "order_item_id": [1, 1, None, None],
            "product_id": 5,
            "quantity": 2,
            "unit_price": 100,
        }
    )

    valid_orders, invalid_orders = validate_orders(orders)

    assert valid_orders.empty
    assert len(invalid_orders) == 4

    missing_id = invalid_orders["order_item_id"].isna()
    assert invalid_orders.loc[~missing_id, "validation_errors"].tolist() == [
        ["duplicate_order_item_id"],
        ["duplicate_order_item_id"],
    ]
    assert invalid_orders.loc[missing_id, "validation_errors"].tolist() == [
        ["missing_order_item_id"],
        ["missing_order_item_id"],
    ]


def test_validate_products_normalization():
    products = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "  Keyboard  ",
                "category": " Electronics ",
            },
            {
                "id": 2,
                "title": "   ",
                "category": " Electronics ",
            },
        ]
    )

    valid, invalid = validate_products(products)

    assert valid.iloc[0]["title"] == "Keyboard"
    assert valid.iloc[0]["category"] == "Electronics"

    assert invalid.iloc[0]["title"] == "   "
    assert "invalid_title" in invalid.iloc[0]["validation_errors"]

    # Исходный DataFrame не изменился
    assert products.iloc[0]["title"] == "  Keyboard  "
