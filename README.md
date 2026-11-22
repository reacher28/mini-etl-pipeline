
# Mini ETL Pipeline

A Python ETL project for processing e-commerce orders and product data.

**Status:** In development. The Extract and Transform stages are implemented; loading into PostgreSQL is planned.

## Data Sources

- **Orders:** CSV file with `order_item_id`, `order_id`, `product_id`, `quantity`, `unit_price`, and `date`.
- **Products:** REST API (DummyJSON) with product IDs, titles, and categories.

The pipeline uses the historical `unit_price` from orders rather than the current product price from the API.

## Implemented Features

### Extract

- Read orders from a CSV file using Pandas.
- Fetch product data from a REST API using Requests.
- Retry requests on connection errors and timeouts.

### Transform

- Validate required IDs, quantities, and unit prices.
- Detect duplicate order-item IDs and product IDs.
- Convert numeric strings and flag invalid numeric values.
- Trim leading and trailing whitespace from valid product titles and categories.
- Join orders with product data and calculate `revenue = quantity * unit_price`.
- Detect orders without a matching valid product.
- Separate valid and invalid records while retaining validation error reasons.
- Preserve original values for records rejected during source-data validation.

The `transform_data(orders, products)` function returns three Pandas DataFrames:

1. `processed_orders` — valid, enriched order items.
2. `invalid_orders` — invalid or unmatched order items with `validation_errors`.
3. `invalid_products` — invalid products with `validation_errors`.

### Testing

- Automated tests with pytest for validation, duplicate detection, normalization, and the full Transform stage.

## Project Structure

```text
mini-etl-pipeline/
├── data/
│   └── orders.csv          # Source orders dataset
├── tests/
│   └── test_transform.py   # Validation and transformation tests
├── .gitignore              # Git ignore rules
├── extract.py              # Extract data from CSV and REST API
├── transform.py            # Validate, clean, and transform data
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup and Testing

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows (Command Prompt):

```bat
.venv\Scripts\activate.bat
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests from the project root:

```bash
python -m pytest -v
```

**Note:** A single-command, end-to-end ETL run is not available yet; the orchestration and Load stages are still in development.

## Roadmap

- [x] Extract orders and products.
- [x] Validate, normalize, and transform data.
- [x] Add automated tests.
- [ ] Design the PostgreSQL schema and implement data loading with UPSERT.
- [ ] Add end-to-end ETL orchestration and logging.
- [ ] Add Docker Compose for reproducible local execution.
- [ ] Set up code quality checks and CI.
