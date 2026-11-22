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
    result = extract_orders(filepath)
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
