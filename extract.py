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
    return pd.read_csv(filepath)


@retry(3)
def extract_products(api_url: str) -> pd.DataFrame:
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data["products"])
