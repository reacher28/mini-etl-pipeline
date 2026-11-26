from functools import wraps

import pandas as pd
import requests


def retry(n):
    if n <= 0:
        raise ValueError

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    result = func(*args, **kwargs)
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ):
                    if i == n - 1:
                        raise
                    print("Перезапуск функции")
                else:
                    return result

        return wrapper

    return decorator


def extract_orders(filepath: str) -> pd.DataFrame:
    return pd.read_csv(
        filepath,
        dtype={
            "order_item_id": "string",
            "order_id": "string",
            "product_id": "string",
            "unit_price": "string",
        },
    )


@retry(3)
def extract_products(api_url: str) -> pd.DataFrame:
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError("API response must be a JSON object")

    if "products" not in data:
        raise ValueError("API response must contain 'products'")

    products = data["products"]

    if not isinstance(products, list):
        raise ValueError("'products' must be a list")

    if not all(isinstance(product, dict) for product in products):
        raise ValueError("Each item in 'products' must be an object")

    products_df = pd.DataFrame(products)

    for column in ("id", "title", "category"):
        if column not in products_df.columns:
            products_df[column] = pd.Series(
                pd.NA,
                index=products_df.index,
                dtype="object",
            )

    return products_df
