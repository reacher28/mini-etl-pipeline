from datetime import date
from decimal import Decimal

import pandas as pd
import requests

from extract import extract_orders, extract_products
from transform import transform_data

API_URL = "https://example.test/products"


def mock_products_api(monkeypatch, payload):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def fake_get(api_url, timeout=10):
        assert api_url == API_URL
        assert timeout == 10
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)


def test_extract_transform_pipeline_success_with_large_ids(tmp_path, monkeypatch):
    csv_path = tmp_path / "orders.csv"

    csv_path.write_text(
        "order_item_id,order_id,product_id,quantity,unit_price,date\n"
        "9007199254740993,9223372036854775807,5,2,100.50,2026-11-24\n"
        "9007199254740994,9223372036854775806,8,1,25.00,2026-11-25\n",
        encoding="utf-8",
    )

    mock_products_api(
        monkeypatch,
        {
            "products": [
                {"id": 5, "title": "Keyboard", "category": "Electronics"},
                {"id": 8, "title": "Mouse", "category": "Electronics"},
            ]
        },
    )

    orders = extract_orders(csv_path)
    products = extract_products(API_URL)

    processed, invalid_orders, invalid_products = transform_data(orders, products)

    assert invalid_orders.empty
    assert invalid_products.empty

    assert processed["order_item_id"].tolist() == [
        9007199254740993,
        9007199254740994,
    ]

    assert processed["order_id"].tolist() == [
        9223372036854775807,
        9223372036854775806,
    ]

    assert processed["order_item_id"].dtype == "Int64"
    assert processed["order_id"].dtype == "Int64"
    assert processed["product_id"].dtype == "Int64"
    assert processed["quantity"].dtype == "Int64"

    assert processed["unit_price"].tolist() == [
        Decimal("100.50"),
        Decimal("25.00"),
    ]

    assert processed["revenue"].tolist() == [
        Decimal("201.00"),
        Decimal("25.00"),
    ]

    assert processed["date"].tolist() == [
        date(2026, 11, 24),
        date(2026, 11, 25),
    ]

    assert processed["product_name"].tolist() == [
        "Keyboard",
        "Mouse",
    ]


def test_extract_transform_pipeline_mixed_data(tmp_path, monkeypatch):
    csv_path = tmp_path / "orders.csv"

    csv_path.write_text(
        "order_item_id,order_id,product_id,quantity,unit_price,date\n"
        "501,101,5,2,100.50,2026-11-24\n"
        "502,102,999,1,50.00,2026-11-24\n"
        "503,103,5,0,10.00,2026-11-24\n"
        "504,104,8,1,25.00,2026-11-24\n",
        encoding="utf-8",
    )

    mock_products_api(
        monkeypatch,
        {
            "products": [
                {"id": 5, "title": "Keyboard", "category": "Electronics"},
                {"id": 8, "title": "Mouse"},
            ]
        },
    )

    orders = extract_orders(csv_path)
    products = extract_products(API_URL)

    processed, invalid_orders, invalid_products = transform_data(orders, products)

    assert processed["order_item_id"].tolist() == [501]
    assert processed["revenue"].tolist() == [Decimal("201.00")]

    errors_by_item = invalid_orders.set_index("order_item_id")[
        "validation_errors"
    ].to_dict()

    assert errors_by_item == {
        "503": ["invalid_quantity"],
        "502": ["unmatched_product_id"],
        "504": ["unmatched_product_id"],
    }

    raw_by_item = invalid_orders.set_index("order_item_id")

    assert raw_by_item.at["502", "product_id"] == "999"
    assert raw_by_item.at["502", "unit_price"] == "50.00"

    assert raw_by_item.at["504", "product_id"] == "8"
    assert raw_by_item.at["504", "unit_price"] == "25.00"

    assert invalid_products["id"].tolist() == [8]

    assert invalid_products["validation_errors"].tolist() == [["invalid_category"]]

    assert pd.isna(invalid_products["category"].iloc[0])

    assert len(processed) + len(invalid_orders) == len(orders)
