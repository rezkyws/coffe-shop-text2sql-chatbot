-- DimDate: dimensi tanggal harian
CREATE TABLE dim_date (
    date_id         INTEGER PRIMARY KEY,           -- contoh: 20190401
    full_date       DATE        NOT NULL,          -- 2019-04-01
    week_id         INTEGER,                       -- ngambil dari Week_ID
    week_desc       VARCHAR(50),
    month_number    SMALLINT,                      -- 1..12 (dari Month_ID)
    month_name      VARCHAR(20),
    quarter_number  SMALLINT,                      -- 1..4 (dari Quarter_ID)
    quarter_name    VARCHAR(20),
    year_number     INTEGER                        -- ngambil dari Year_ID
);

-- DimTime: dimensi jam (bukan per detik)
CREATE TABLE dim_time (
    time_id         INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hour_of_day     SMALLINT    NOT NULL,          -- 0..23
    minute_interval SMALLINT    NOT NULL,          -- misal 0, 30 (kalau pakai interval 30 menit)
    day_part        VARCHAR(30),                   -- Morning, Afternoon, Evening, dsb.
    is_morning_rush BOOLEAN     NOT NULL DEFAULT FALSE,  -- jam 7-10
    is_lunch_break  BOOLEAN     NOT NULL DEFAULT FALSE,  -- jam 11-14
    is_after_work   BOOLEAN     NOT NULL DEFAULT FALSE   -- jam 17-20
);

-- DimStore: dimensi outlet / toko
CREATE TABLE dim_store (
    sales_outlet_id     INTEGER PRIMARY KEY,
    sales_outlet_type   VARCHAR(50),
    store_square_feet   INTEGER,
    store_address       VARCHAR(200),
    store_city          VARCHAR(100),
    store_state_province VARCHAR(100),
    store_telephone     VARCHAR(50),
    store_postal_code   VARCHAR(20),
    store_longitude     NUMERIC(10,6),
    store_latitude      NUMERIC(10,6),
    manager             VARCHAR(100),
    neighborhood        VARCHAR(100)               -- fixed typo dari 'Neighorhood'
);

-- DimProduct: dimensi produk (minuman, pastry, dll)
CREATE TABLE dim_product (
    product_id              INTEGER PRIMARY KEY,
    product_group           VARCHAR(50),
    product_category        VARCHAR(50),
    product_type            VARCHAR(50),
    product_name            VARCHAR(100),          -- dari kolom 'product'
    product_description     VARCHAR(500),
    unit_of_measure         VARCHAR(20),
    current_wholesale_price NUMERIC(12,2),
    current_retail_price    NUMERIC(12,2),
    tax_exempt              BOOLEAN,               -- mapping dari 'tax_exempt_yn'
    promo                   BOOLEAN,               -- mapping dari 'promo_yn'
    new_product             BOOLEAN                -- mapping dari 'new_product_yn'
);

-- DimCustomer: dimensi customer + generasi
CREATE TABLE dim_customer (
    customer_id             INTEGER PRIMARY KEY,
    home_store              INTEGER,               -- bisa FK ke dim_store
    customer_first_name     VARCHAR(100),
    customer_email          VARCHAR(200),
    customer_since          DATE,
    loyalty_card_number     VARCHAR(50),
    birthdate               DATE,
    gender                  VARCHAR(20),
    birth_year              INTEGER,
    generation              VARCHAR(50),

    CONSTRAINT fk_dim_customer_home_store
        FOREIGN KEY (home_store) REFERENCES dim_store (sales_outlet_id)
);

-- DimStaff: dimensi staff
CREATE TABLE dim_staff (
    staff_id    INTEGER PRIMARY KEY,
    first_name  VARCHAR(100) NOT NULL,
    last_name   VARCHAR(100) NOT NULL,
    position    VARCHAR(100),
    start_date  DATE,
    location    VARCHAR(20)               -- bisa simpan 'HQ', 'WH', 'FL', atau id store dalam bentuk teks
);

-- FactSales: fakta penjualan per line item
CREATE TABLE fact_sales (
    sales_fact_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    date_id            INTEGER     NOT NULL,
    time_id            INTEGER     NOT NULL,
    sales_outlet_id    INTEGER     NOT NULL,
    staff_id           INTEGER,
    customer_id        INTEGER     NOT NULL,
    product_id         INTEGER     NOT NULL,

    -- keys dan info transaksi
    receipt_key        VARCHAR(100) NOT NULL,    -- gabungan (store, date, transaction_id, order) dari ETL
    transaction_id     INTEGER      NOT NULL,
    order_number       INTEGER      NOT NULL,    -- dari kolom 'order'
    line_item_id       INTEGER      NOT NULL,

    -- atribut transaksi
    instore            BOOLEAN      NOT NULL,    -- mapping dari 'instore_yn' (Y/N)
    promo_item         BOOLEAN      NOT NULL,    -- mapping dari 'promo_item_yn' (Y/N)

    -- measures
    quantity           INTEGER      NOT NULL,
    unit_price         NUMERIC(12,2) NOT NULL,
    line_item_amount   NUMERIC(14,2) NOT NULL,

    -- FK constraints
    CONSTRAINT fk_fact_sales_date
        FOREIGN KEY (date_id)   REFERENCES dim_date (date_id),
    CONSTRAINT fk_fact_sales_time
        FOREIGN KEY (time_id)   REFERENCES dim_time (time_id),
    CONSTRAINT fk_fact_sales_store
        FOREIGN KEY (sales_outlet_id)  REFERENCES dim_store (sales_outlet_id),
    CONSTRAINT fk_fact_sales_staff
        FOREIGN KEY (staff_id)  REFERENCES dim_staff (staff_id),
    CONSTRAINT fk_fact_sales_customer
        FOREIGN KEY (customer_id) REFERENCES dim_customer (customer_id),
    CONSTRAINT fk_fact_sales_product
        FOREIGN KEY (product_id) REFERENCES dim_product (product_id)
);

-- FactPastryInventory: stok & waste per produk–store–tanggal
CREATE TABLE fact_pastry_inventory (
    sales_outlet_id INTEGER     NOT NULL,
    product_id      INTEGER     NOT NULL,
    date_id         INTEGER     NOT NULL,

    start_of_day    INTEGER     NOT NULL,
    quantity_sold   INTEGER     NOT NULL,
    waste           INTEGER     NOT NULL,

    PRIMARY KEY (sales_outlet_id, product_id, date_id),

    CONSTRAINT fk_fact_pastry_store
        FOREIGN KEY (sales_outlet_id) REFERENCES dim_store (sales_outlet_id),
    CONSTRAINT fk_fact_pastry_product
        FOREIGN KEY (product_id) REFERENCES dim_product (product_id),
    CONSTRAINT fk_fact_pastry_date
        FOREIGN KEY (date_id) REFERENCES dim_date (date_id)
);

-- FactSalesTarget: target penjualan per store per bulan
CREATE TABLE fact_sales_target (
    sales_outlet_id INTEGER     NOT NULL,
    date_id_month   INTEGER     NOT NULL,   -- refer ke dim_date.date_id (hari pertama bulan tsb)

    beans_goal          INTEGER NOT NULL,
    beverage_goal       INTEGER NOT NULL,
    food_goal           INTEGER NOT NULL,
    merchandise_goal    INTEGER NOT NULL,   -- rename dari 'merchandise _goal'
    total_goal          INTEGER NOT NULL,

    PRIMARY KEY (sales_outlet_id, date_id_month),

    CONSTRAINT fk_fact_target_store
        FOREIGN KEY (sales_outlet_id)      REFERENCES dim_store (sales_outlet_id),
    CONSTRAINT fk_fact_target_date_month
        FOREIGN KEY (date_id_month) REFERENCES dim_date (date_id)
);

-- index-index untuk foreign key di fact tables
CREATE INDEX idx_fact_sales_date      ON fact_sales (date_id);
CREATE INDEX idx_fact_sales_time      ON fact_sales (time_id);
CREATE INDEX idx_fact_sales_store     ON fact_sales (sales_outlet_id);
CREATE INDEX idx_fact_sales_customer  ON fact_sales (customer_id);
CREATE INDEX idx_fact_sales_product   ON fact_sales (product_id);
CREATE INDEX idx_fact_sales_staff     ON fact_sales (staff_id);
CREATE INDEX idx_fact_sales_receipt   ON fact_sales (receipt_key);

CREATE INDEX idx_fact_pastry_date     ON fact_pastry_inventory (date_id);
CREATE INDEX idx_fact_pastry_store    ON fact_pastry_inventory (sales_outlet_id);
CREATE INDEX idx_fact_pastry_product  ON fact_pastry_inventory (product_id);

CREATE INDEX idx_fact_target_date     ON fact_sales_target (date_id_month);
CREATE INDEX idx_fact_target_store    ON fact_sales_target (sales_outlet_id);
