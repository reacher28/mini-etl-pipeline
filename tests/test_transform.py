from transform import (
    validate_products,
    transform_data,
    validate_orders,
    parse_bigint,
    parse_unit_price,
    parse_order_date,
)
from pandas.api.types import is_numeric_dtype
import numpy as np
import pytest
from decimal import Decimal
import pandas as pd
from datetime import date


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
                "order_id": 1001,
                "product_id": 5,
                "quantity": "2",
                "unit_price": "100.5",
                "date": "2026-11-26",
            },
            {
                "order_item_id": 2,
                "order_id": 1002,
                "product_id": 5,
                "quantity": "unknown",
                "unit_price": "-50",
                "date": "2026-11-26",
            },
        ]
    )

    valid_orders, invalid_orders = validate_orders(orders)

    assert valid_orders["order_item_id"].tolist() == [1]
    assert valid_orders["quantity"].iloc[0] == 2
    assert valid_orders["unit_price"].iloc[0] == Decimal("100.50")
    assert is_numeric_dtype(valid_orders["quantity"])
    assert isinstance(valid_orders["unit_price"].iloc[0], Decimal)

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
            "order_id": [1001, 1001, 1002, 1003],
            "product_id": 5,
            "quantity": 2,
            "unit_price": 100,
            "date": "2026-11-26",
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


def test_validate_orders_integer_order_id():
    orders = pd.DataFrame(
        {
            "order_item_id": range(1, 8),
            "order_id": [101, "101", 102.0, None, "unknown", 102.5, float("inf")],
            "product_id": 5,
            "quantity": 2,
            "unit_price": 100,
            "date": "2026-11-26",
        }
    )

    valid_orders, invalid_orders = validate_orders(orders)

    assert valid_orders["order_item_id"].tolist() == [1, 2, 3]
    assert invalid_orders["order_item_id"].tolist() == [4, 5, 6, 7]
    assert invalid_orders["validation_errors"].tolist() == [
        ["missing_order_id"],
        ["invalid_order_id"],
        ["invalid_order_id"],
        ["invalid_order_id"],
    ]
    assert valid_orders["order_id"].tolist() == [101, 101, 102]
    assert valid_orders["order_id"].dtype == "Int64"


def test_validate_orders_order_id_boundaries():
    orders = pd.DataFrame(
        {
            "order_item_id": [1, 2, 3, 4],
            "order_id": [
                0,
                -5,
                "9223372036854775807",
                "9223372036854775808",
            ],
            "product_id": 5,
            "quantity": 2,
            "unit_price": 100,
            "date": "2026-11-26",
        }
    )

    valid_orders, invalid_orders = validate_orders(orders)

    assert valid_orders["order_item_id"].tolist() == [3]
    assert invalid_orders["order_item_id"].tolist() == [1, 2, 4]

    assert invalid_orders["validation_errors"].tolist() == [
        ["invalid_order_id"],
        ["invalid_order_id"],
        ["invalid_order_id"],
    ]

    assert valid_orders["order_id"].iloc[0] == 9223372036854775807


def test_validate_orders_integer_quantity():
    orders = pd.DataFrame(
        {
            "order_item_id": range(1, 8),
            "order_id": range(1001, 1008),
            "product_id": 5,
            "quantity": pd.Series(
                [3, 3.0, 2.5, 0, -5, float("nan"), float("inf")],
                dtype=object,
            ),
            "unit_price": 100,
            "date": "2026-11-24",
        }
    )

    valid_orders, invalid_orders = validate_orders(orders)

    assert valid_orders["order_item_id"].tolist() == [1, 2]
    assert invalid_orders["order_item_id"].tolist() == [3, 4, 5, 6, 7]
    assert invalid_orders["validation_errors"].tolist() == [["invalid_quantity"]] * 5


@pytest.mark.parametrize(
    "value, expected",
    [
        # Обычные целые числа
        (42, 42),
        ("42", 42),
        (42.0, 42),
        (np.int64(100), 100),
        (np.uint64(100), 100),
        # Строковые значения
        ("00042", 42),
        ("  123  ", 123),
        ("+42", 42),
        ("-5", -5),
        ("0", 0),
        ("", None),
        ("unknown", None),
        ("12.5", None),
        # Некорректные значения
        (42.5, None),
        (True, None),
        (False, None),
        (np.bool_(True), None),
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        # Границы PostgreSQL BIGINT
        (-(2**63), -(2**63)),
        (2**63 - 1, 2**63 - 1),
        (-(2**63) - 1, None),
        (2**63, None),
        # Граничные значения в строках
        (str(-(2**63)), -(2**63)),
        (str(2**63 - 1), 2**63 - 1),
        (str(-(2**63) - 1), None),
        (str(2**63), None),
        # Очень большие числа
        ("9" * 100, None),
        (np.uint64(2**64 - 1), None),
    ],
)
def test_parse_bigint(value, expected):
    assert parse_bigint(value) == expected


def test_validate_orders_normalized_ids():
    orders = pd.DataFrame(
        {
            "order_item_id": [101, "101", 102, 0, 103],
            "order_id": [1001, 1001, 1002, 1003, 1004],
            "product_id": [5, "5", "5", 5, -5],
            "quantity": 2,
            "unit_price": 100,
            "date": "2026-11-24",
        }
    )

    valid_orders, invalid_orders = validate_orders(orders)

    assert valid_orders["order_item_id"].tolist() == [102]
    assert valid_orders["product_id"].tolist() == [5]

    assert invalid_orders["validation_errors"].tolist() == [
        ["duplicate_order_item_id"],
        ["duplicate_order_item_id"],
        ["invalid_order_item_id"],
        ["invalid_product_id"],
    ]

    assert valid_orders["order_item_id"].dtype == "Int64"
    assert valid_orders["product_id"].dtype == "Int64"


def test_transform_data_normalized_product_ids():
    orders = pd.DataFrame(
        {
            "order_item_id": [1, 2],
            "order_id": [1001, 1002],
            "product_id": ["5", 7],
            "quantity": [2, 3],
            "unit_price": [100, 200],
            "date": ["2026-11-24", "2026-11-24"],
        }
    )

    products = pd.DataFrame(
        {
            "id": [5, "7", -1, "9223372036854775808"],
            "title": ["Keyboard", "Mouse", "Invalid", "Too Large"],
            "category": ["electronics"] * 4,
        }
    )

    processed, invalid_orders, invalid_products = transform_data(orders, products)

    assert processed["order_item_id"].tolist() == [1, 2]
    assert processed["product_id"].tolist() == [5, 7]
    assert processed["product_name"].tolist() == ["Keyboard", "Mouse"]
    assert processed["revenue"].tolist() == [200, 600]

    assert invalid_orders.empty
    assert invalid_products["validation_errors"].tolist() == [
        ["invalid_id"],
        ["invalid_id"],
    ]


def test_validate_orders_quantity_boundaries():
    orders = pd.DataFrame(
        {
            "order_item_id": [1, 2, 3, 4, 5],
            "order_id": 1001,
            "product_id": 5,
            "quantity": [0, 1, 2**31 - 1, 2**31, 2.5],
            "unit_price": 100,
            "date": "2026-11-24",
        }
    )

    valid_orders, invalid_orders = validate_orders(orders)

    assert valid_orders["order_item_id"].tolist() == [2, 3]
    assert valid_orders["quantity"].tolist() == [1, 2147483647]
    assert valid_orders["quantity"].dtype == "Int64"

    assert invalid_orders["order_item_id"].tolist() == [1, 4, 5]
    assert invalid_orders["validation_errors"].tolist() == [
        ["invalid_quantity"],
        ["invalid_quantity"],
        ["invalid_quantity"],
    ]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("100.50", Decimal("100.50")),
        ("0", Decimal("0.00")),
        ("1.230", Decimal("1.23")),
        ("9999999999.99", Decimal("9999999999.99")),
        ("1.234", None),
        ("-5", None),
        ("NaN", None),
        ("Infinity", None),
        ("abc", None),
        (None, None),
        (True, None),
        ("10000000000.00", None),
    ],
)
def test_parse_unit_price(raw, expected):
    assert parse_unit_price(raw) == expected


def test_transform_decimal_prices():
    orders = pd.DataFrame(
        {
            "order_item_id": [1, 2],
            "order_id": [10, 11],
            "product_id": [5, 5],
            "quantity": [2, 1],
            "unit_price": ["0.10", "1.234"],
            "date": "2026-11-26",
        }
    )

    products = pd.DataFrame(
        {
            "id": [5],
            "title": ["Keyboard"],
            "category": ["Electronics"],
        }
    )

    processed, invalid_orders, invalid_products = transform_data(orders, products)

    assert processed["revenue"].iloc[0] == Decimal("0.20")
    assert isinstance(processed["revenue"].iloc[0], Decimal)

    assert invalid_orders["unit_price"].iloc[0] == "1.234"
    assert invalid_orders["validation_errors"].iloc[0] == ["invalid_unit_price"]

    assert invalid_products.empty


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-11-26", date(2026, 11, 26)),
        (" 2026-11-26 ", date(2026, 11, 26)),
        ("2024-02-29", date(2024, 2, 29)),
        (date(2026, 11, 26), date(2026, 11, 26)),
        ("2026-02-29", None),
        ("2026-13-01", None),
        ("2026-11-31", None),
        ("2026-1-2", None),
        ("26.11.2026", None),
        ("2026-11-26 15:30:00", None),
        ("abc", None),
        ("", None),
        (None, None),
        (20261126, None),
    ],
)
def test_parse_order_date(raw, expected):
    assert parse_order_date(raw) == expected


def test_validate_orders_dates():
    orders = pd.DataFrame(
        {
            "order_item_id": [1, 2, 3],
            "order_id": [101, 102, 103],
            "product_id": [5, 5, 5],
            "quantity": [2, 3, 1],
            "unit_price": ["100.50", "200.00", "50.00"],
            "date": ["2026-11-26", "2026-02-29", None],
        }
    )

    valid, invalid = validate_orders(orders)

    assert len(valid) == 1
    assert valid["date"].iloc[0] == date(2026, 11, 26)
    assert type(valid["date"].iloc[0]) is date

    assert invalid["order_item_id"].tolist() == [2, 3]
    assert invalid["validation_errors"].tolist() == [
        ["invalid_date"],
        ["invalid_date"],
    ]

    # Исходные значения сохраняются
    assert invalid["date"].iloc[0] == "2026-02-29"
    assert pd.isna(invalid["date"].iloc[1])


def test_validate_orders_revenue_boundaries():
    orders = pd.DataFrame(
        {
            "order_item_id": [1, 2],
            "order_id": [101, 102],
            "product_id": [5, 5],
            "quantity": [100, 101],
            "unit_price": [
                "9999999999.99",
                "9999999999.99",
            ],
            "date": ["2026-11-26", "2026-11-26"],
        }
    )

    valid, invalid = validate_orders(orders)

    assert len(valid) == 1
    assert valid["order_item_id"].iloc[0] == 1

    assert len(invalid) == 1
    assert invalid["order_item_id"].iloc[0] == 2
    assert invalid["validation_errors"].iloc[0] == ["invalid_revenue"]

    # Невалидная строка сохраняет исходные значения
    assert invalid["quantity"].iloc[0] == 101
    assert invalid["unit_price"].iloc[0] == "9999999999.99"



def test_transform_data_unmatched_preserves_raw_values():
    orders = pd.DataFrame({
        "order_item_id": ["000501"],
        "order_id": ["00101"],
        "product_id": ["999"],
        "quantity": ["2"],
        "unit_price": ["100.50"],
        "date": ["2026-11-26"],
    })

    products = pd.DataFrame({
        "id": ["5"],
        "title": ["Keyboard"],
        "category": ["Electronics"],
    })

    processed, invalid_orders, invalid_products = transform_data(
        orders, products
    )

    assert processed.empty
    assert invalid_products.empty

    assert len(invalid_orders) == 1

    raw_order = invalid_orders.iloc[0]

    assert raw_order["order_item_id"] == "000501"
    assert raw_order["order_id"] == "00101"
    assert raw_order["product_id"] == "999"
    assert raw_order["quantity"] == "2"
    assert raw_order["unit_price"] == "100.50"
    assert raw_order["date"] == "2026-11-26"

    assert raw_order["validation_errors"] == [
        "unmatched_product_id"
    ]
