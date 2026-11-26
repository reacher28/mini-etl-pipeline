from extract import extract_orders, extract_products
import pandas as pd
import requests
import pytest


def test_extract_orders(tmp_path):
    expected_orders = pd.DataFrame(
        {
            "order_item_id": [101, 102, 103],
            "order_id": [1001, 1001, 1002],
            "product_id": [1, 2, 3],
            "quantity": [2, 1, 4],
            "unit_price": [150.0, 299.5, 75.0],
            "date": ["2026-11-20", "2026-11-20", "2026-11-21"],
        }
    )
    filepath = tmp_path / "orders.csv"
    expected_orders.to_csv(filepath, index=False)
    expected_orders["unit_price"] = expected_orders["unit_price"].astype("string")
    result = extract_orders(filepath)

    id_columns = ["order_item_id", "order_id", "product_id"]
    expected_orders[id_columns] = expected_orders[id_columns].astype("string")

    pd.testing.assert_frame_equal(expected_orders, result)


def test_extract_products(monkeypatch):
    products = [
        {"id": 1, "title": "Keyboard", "category": "electronics"},
        {"id": 2, "title": "Mouse", "category": "electronics"},
    ]

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"products": products}

    def fake_get(api_url, timeout=10):
        assert api_url == "https://example.test/products"
        assert timeout == 10
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    result = extract_products("https://example.test/products")

    expected = pd.DataFrame(products)
    pd.testing.assert_frame_equal(result, expected)


def test_extract_products_retry(monkeypatch):
    products = [
        {"id": 1, "title": "Keyboard", "category": "electronics"},
        {"id": 2, "title": "Mouse", "category": "electronics"},
    ]
    call_count = 0

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"products": products}

    def fake_get(api_url, timeout=10):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise requests.exceptions.Timeout
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    result = extract_products("https://example.test/products")

    pd.testing.assert_frame_equal(result, pd.DataFrame(products))
    assert call_count == 3


def test_extract_products_retry_exhausted(monkeypatch):
    call_count = 0

    def fake_get(api_url, timeout=10):
        nonlocal call_count
        call_count += 1
        raise requests.exceptions.Timeout

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(requests.exceptions.Timeout):
        extract_products("https://example.test/products")

    assert call_count == 3


def test_extract_products_http_error(monkeypatch):
    call_count = 0

    class FakeResponse:
        def raise_for_status(self):
            raise requests.exceptions.HTTPError("404 Not Found")

    def fake_get(api_url, timeout=10):
        nonlocal call_count
        call_count += 1
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(requests.exceptions.HTTPError, match="404 Not Found"):
        extract_products("https://example.test/products")

    assert call_count == 1


def test_extract_orders_preserves_bigint(tmp_path):
    filepath = tmp_path / "orders.csv"

    filepath.write_text(
        "order_item_id,order_id,product_id,quantity,unit_price,date\n"
        "1,9223372036854775807,5,2,100,2026-11-24\n"
        "2,9007199254740993,,3,200,2026-11-24\n",
        encoding="utf-8",
    )

    orders = extract_orders(str(filepath))

    assert orders["order_id"].tolist() == [
        "9223372036854775807",
        "9007199254740993",
    ]

    assert orders["order_id"].dtype == "string"
    assert orders["product_id"].dtype == "string"
    assert pd.isna(orders["product_id"].iloc[1])


@pytest.mark.parametrize(
    "payload, error_message",
    [
        ([], "JSON object"),
        ({}, "products"),
        ({"products": {"id": 1}}, "must be a list"),
        ({"products": [123]}, "must be an object"),
    ],
)
def test_extract_products_invalid_response(monkeypatch, payload, error_message):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def fake_get(api_url, timeout=10):
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(ValueError, match=error_message):
        extract_products("https://example.test/products")


def test_extract_products_empty_list(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"products": []}

    def fake_get(api_url, timeout=10):
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    result = extract_products("https://example.test/products")

    expected = pd.DataFrame(columns=["id", "title", "category"])

    pd.testing.assert_frame_equal(result, expected)


def test_extract_products_missing_category(monkeypatch):
    products = [{"id": 1, "title": "Keyboard"}]

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"products": products}

    def fake_get(api_url, timeout=10):
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    result = extract_products("https://example.test/products")

    assert result["id"].iloc[0] == 1
    assert result["title"].iloc[0] == "Keyboard"
    assert pd.isna(result["category"].iloc[0])
