CREATE TABLE IF NOT EXISTS processed_orders (
    order_item_id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    date DATE NOT NULL,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    revenue NUMERIC(14,2) NOT NULL CHECK (revenue >= 0)
);


CREATE TABLE IF NOT EXISTS invalid_orders (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    raw_data JSONB NOT NULL
        CHECK (jsonb_typeof(raw_data) = 'object'),
    validation_errors JSONB NOT NULL
        CHECK (
            CASE
                WHEN jsonb_typeof(validation_errors) = 'array'
                THEN jsonb_array_length(validation_errors) > 0
                ELSE FALSE
            END
        )
);

CREATE TABLE IF NOT EXISTS invalid_products (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    raw_data JSONB NOT NULL
        CHECK (jsonb_typeof(raw_data) = 'object'),
    validation_errors JSONB NOT NULL
        CHECK (
            CASE
                WHEN jsonb_typeof(validation_errors) = 'array'
                THEN jsonb_array_length(validation_errors) > 0
                ELSE FALSE
            END
        )
);