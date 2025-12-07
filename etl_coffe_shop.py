import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = "./data"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432),
    "dbname": os.getenv("DB_NAME", "coffee_shop"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "secret"),
}

DB_CONFIG_VECTOR = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432),
    "dbname": os.getenv("DB_NAME_USER", "user_coffee_shop"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "secret"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_vector_connection():
    return psycopg2.connect(**DB_CONFIG_VECTOR)


def df_to_records(df, columns):
    df = df.copy()
    df = df[columns]

    # ganti NaN dengan None agar dimengerti SQL sebagai NULL
    df = df.where(pd.notnull(df), None)

    return [tuple(x) for x in df.values.tolist()]


def bulk_insert(conn, table_name, df, columns):
    if df.empty:
        print(f"[INFO] {table_name}: DataFrame kosong, skip insert.")
        return

    records = df_to_records(df, columns)
    cols_sql = ", ".join(columns)
    sql = f"INSERT INTO {table_name} ({cols_sql}) VALUES %s"

    with conn.cursor() as cur:
        execute_values(cur, sql, records)
    print(f"[OK] Insert {len(df)} baris ke {table_name}")


def load_raw_data():
    raw = {}
    raw["dates"] = pd.read_csv(os.path.join(DATA_DIR, "dates.csv"))
    raw["sales_outlet"] = pd.read_csv(os.path.join(DATA_DIR, "sales_outlet.csv"))
    raw["product"] = pd.read_csv(os.path.join(DATA_DIR, "product.csv"))
    raw["customer"] = pd.read_csv(os.path.join(DATA_DIR, "customer.csv"))
    raw["generations"] = pd.read_csv(os.path.join(DATA_DIR, "generations.csv"))
    raw["staff"] = pd.read_csv(os.path.join(DATA_DIR, "staff.csv"))
    raw["sales_receipts"] = pd.read_csv(os.path.join(DATA_DIR, "201904_sales_reciepts.csv"))
    raw["pastry_inventory"] = pd.read_csv(os.path.join(DATA_DIR, "pastry_inventory.csv"))
    raw["sales_targets"] = pd.read_csv(os.path.join(DATA_DIR, "sales_targets.csv"))
    return raw


def build_dim_date(df_dates: pd.DataFrame) -> pd.DataFrame:
    df = df_dates.copy()

    # transaction_date di Dates.csv: "4/1/2019" => date
    df["full_date"] = pd.to_datetime(df["transaction_date"], format="%m/%d/%Y").dt.date

    dim_date = df.rename(
        columns={
            "Date_ID": "date_id",
            "Week_ID": "week_id",
            "Week_Desc": "week_desc",
            "Month_ID": "month_number",
            "Month_Name": "month_name",
            "Quarter_ID": "quarter_number",
            "Quarter_Name": "quarter_name",
            "Year_ID": "year_number",
        }
    )[
        [
            "date_id",
            "full_date",
            "week_id",
            "week_desc",
            "month_number",
            "month_name",
            "quarter_number",
            "quarter_name",
            "year_number",
        ]
    ]

    return dim_date


def build_dim_store(df_store: pd.DataFrame) -> pd.DataFrame:
    df = df_store.copy()
    df = df.rename(columns={"Neighorhood": "neighborhood"})

    dim_store = df[
        [
            "sales_outlet_id",
            "sales_outlet_type",
            "store_square_feet",
            "store_address",
            "store_city",
            "store_state_province",
            "store_telephone",
            "store_postal_code",
            "store_longitude",
            "store_latitude",
            "manager",
            "neighborhood",
        ]
    ]
    return dim_store


def build_dim_product(df_product: pd.DataFrame) -> pd.DataFrame:
    df = df_product.copy()

    # remove simbol $ di current_retail_price
    df["current_retail_price"] = (
        df["current_retail_price"].astype(str).str.replace(r"[$]", "", regex=True)
    )
    df["current_retail_price"] = pd.to_numeric(df["current_retail_price"])

    # memastikan wholesale sudah numeric di dataset ini
    df["current_wholesale_price"] = pd.to_numeric(df["current_wholesale_price"])

    def yn_to_bool(s):
        if pd.isna(s):
            return None
        return str(s).upper() == "Y"

    df["tax_exempt"] = df["tax_exempt_yn"].apply(yn_to_bool)
    df["promo"] = df["promo_yn"].apply(yn_to_bool)
    df["new_product"] = df["new_product_yn"].apply(yn_to_bool)

    dim_product = df.rename(
        columns={
            "product": "product_name",
        }
    )[
        [
            "product_id",
            "product_group",
            "product_category",
            "product_type",
            "product_name",
            "product_description",
            "unit_of_measure",
            "current_wholesale_price",
            "current_retail_price",
            "tax_exempt",
            "promo",
            "new_product",
        ]
    ]

    return dim_product


def build_dim_customer(df_customer: pd.DataFrame, df_gen: pd.DataFrame) -> pd.DataFrame:
    df = df_customer.copy()
    df_gen = df_gen.copy()

    # join generasi berdasarkan birth_year
    df = df.merge(df_gen, on="birth_year", how="left")

    # rename penamaan kolom yang ga lazim
    df = df.rename(columns={"customer_first-name": "customer_first_name"})

    # parse tanggal
    # format di CSV contoh: "2019-03-10" / dsb, jadi pakai to_datetime generic
    df["customer_since"] = pd.to_datetime(df["customer_since"]).dt.date
    df["birthdate"] = pd.to_datetime(df["birthdate"]).dt.date

    dim_customer = df[
        [
            "customer_id",
            "home_store",
            "customer_first_name",
            "customer_email",
            "customer_since",
            "loyalty_card_number",
            "birthdate",
            "gender",
            "birth_year",
            "generation",
        ]
    ]

    # tambah baris khusus 'Unknown' (id=0). Buat terpisah, nanti diinsert manual
    return dim_customer


def build_dim_staff(df_staff: pd.DataFrame) -> pd.DataFrame:
    df = df_staff.copy()

    # drop kolom Unnamed
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    # parse tanggal
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date

    dim_staff = df[
        [
            "staff_id",
            "first_name",
            "last_name",
            "position",
            "start_date",
            "location",
        ]
    ]
    return dim_staff


def build_dim_time(df_sales: pd.DataFrame) -> pd.DataFrame:
    df = df_sales.copy()

    # parse time string "HH:MM:SS"
    times = pd.to_datetime(df["transaction_time"], format="%H:%M:%S")

    df["hour_of_day"] = times.dt.hour

    # bucket 30 menitan
    df["minute_interval"] = (times.dt.minute // 30) * 30

    dim_time = df[["hour_of_day", "minute_interval"]].drop_duplicates()

    # business flags
    def get_day_part(h):
        if 5 <= h < 11:
            return "Morning"
        elif 11 <= h < 15:
            return "Midday"
        elif 15 <= h < 18:
            return "Afternoon"
        elif 18 <= h < 22:
            return "Evening"
        else:
            return "Off hours"

    dim_time["day_part"] = dim_time["hour_of_day"].apply(get_day_part)
    dim_time["is_morning_rush"] = dim_time["hour_of_day"].between(7, 10, inclusive="left")
    dim_time["is_lunch_break"] = dim_time["hour_of_day"].between(11, 14, inclusive="left")
    dim_time["is_after_work"] = dim_time["hour_of_day"].between(17, 20, inclusive="left")

    return dim_time


def build_fact_sales(df_sales, dim_date, dim_time_map, valid_customer_ids):
    df = df_sales.copy()

    # make sure customer_id numeric, handle NaN jadi 0
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").fillna(0).astype(int)

    # make sure apakah customer_id ada di valid_ids. Jika tidak, ganti ke 0 (Unknown)
    # fixing customer_id=5000
    df.loc[~df["customer_id"].isin(valid_customer_ids), "customer_id"] = 0

    # map date_id dari dim_date berdasarkan full_date
    df_dates = dim_date[["date_id", "full_date"]].copy()

    # receipts punya transaction_date "2019-04-01"
    df["full_date"] = pd.to_datetime(df["transaction_date"]).dt.date
    df = df.merge(df_dates, on="full_date", how="left")

    # build hour/minute_interval sama seperti dim_time, lalu map ke time_id
    times = pd.to_datetime(df["transaction_time"], format="%H:%M:%S")
    df["hour_of_day"] = times.dt.hour
    df["minute_interval"] = (times.dt.minute // 30) * 30

    # mapping time_id dengan dim_time_map {(hour, minute_interval): time_id}
    df["time_id"] = df.apply(
        lambda r: dim_time_map[(int(r["hour_of_day"]), int(r["minute_interval"]))],
        axis=1,
    )

    def yn_to_bool(s):
        if pd.isna(s):
            return False
        return str(s).upper() == "Y"

    df["instore"] = df["instore_yn"].apply(yn_to_bool)
    df["promo_item"] = df["promo_item_yn"].apply(yn_to_bool)

    # receipt_key sederhana => store+date+transaction
    df["receipt_key"] = (
        df["sales_outlet_id"].astype(str)
        + "-"
        + df["full_date"].astype(str)
        + "-"
        + df["transaction_id"].astype(str)
    )

    fact_sales = df.rename(
        columns={
            "order": "order_number",
        }
    )[
        [
            "date_id",
            "time_id",
            "sales_outlet_id",
            "staff_id",
            "customer_id",
            "product_id",
            "receipt_key",
            "transaction_id",
            "order_number",
            "line_item_id",
            "instore",
            "promo_item",
            "quantity",
            "unit_price",
            "line_item_amount",
        ]
    ]

    return fact_sales


def build_fact_pastry_inventory(df_pastry, dim_date):
    df = df_pastry.copy()

    df_dates = dim_date[["date_id", "full_date"]].copy()
    df["full_date"] = pd.to_datetime(df["transaction_date"], format="%m/%d/%Y").dt.date

    df = df.merge(df_dates, on="full_date", how="left")

    fact_pastry = df[
        [
            "sales_outlet_id",
            "product_id",
            "date_id",
            "start_of_day",
            "quantity_sold",
            "waste",
        ]
    ]

    return fact_pastry


def build_fact_sales_target(df_targets, dim_date):
    df = df_targets.copy()

    # year_month format: "Apr-19"
    # treat sebagai hari pertama bulan tersebut
    df["month_start"] = pd.to_datetime(df["year_month"], format="%b-%y").dt.date

    df_dates = dim_date[["date_id", "full_date"]].copy()
    df = df.merge(df_dates, left_on="month_start", right_on="full_date", how="left")

    fact_targets = df.rename(
        columns={
            "date_id": "date_id_month",
            "merchandise _goal": "merchandise_goal",
        }
    )[
        [
            "sales_outlet_id",
            "date_id_month",
            "beans_goal",
            "beverage_goal",
            "food_goal",
            "merchandise_goal",
            "total_goal",
        ]
    ]

    return fact_targets


def load_initial_vector_data():
    """Load initial vector data from CSV and insert into query_vector_store table."""
    try:
        # Load the CSV data
        vector_df = pd.read_csv(os.path.join(DATA_DIR, "initial_vector_data.csv"))

        if vector_df.empty:
            print("[INFO] No vector data found in initial_vector_data.csv")
            return

        print(f"[INFO] Loading {len(vector_df)} vector data entries...")

        # Initialize the embedding model (same as in config)
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        # Connect to vector database
        vector_conn = get_vector_connection()
        vector_conn.autocommit = False

        try:
            with vector_conn.cursor() as cur:
                for index, row in vector_df.iterrows():
                    question = str(row["question"])
                    sql_query = str(row["query"])

                    # Generate embedding for the question
                    embedding = model.encode(question, normalize_embeddings=True)

                    # Check if question already exists
                    cur.execute(
                        "SELECT id FROM query_vector_store WHERE question = %s", (question,)
                    )
                    if cur.fetchone() is None:
                        # Insert into query_vector_store table
                        cur.execute(
                            """
                            INSERT INTO query_vector_store
                            (question, sql_query, embedding, metadata)
                            VALUES (%s, %s, %s, %s)
                        """,
                            (
                                question,
                                sql_query,
                                embedding.tolist(),  # Convert numpy array to list
                                "{}",  # JSON string for metadata
                            ),
                        )

                    if (index + 1) % 5 == 0:
                        print(f"[PROGRESS] Processed {index + 1}/{len(vector_df)} entries")

                vector_conn.commit()
                print(f"[SUCCESS] Inserted {len(vector_df)} vector entries into query_vector_store")

        except Exception as e:
            vector_conn.rollback()
            print(f"[ERROR] Failed to insert vector data: {e}")
            raise
        finally:
            vector_conn.close()

    except FileNotFoundError:
        print("[INFO] initial_vector_data.csv not found, skipping vector data loading")
    except Exception as e:
        print(f"[ERROR] Error loading vector data: {e}")
        raise


def main():
    raw = load_raw_data()

    dim_date = build_dim_date(raw["dates"])
    dim_store = build_dim_store(raw["sales_outlet"])
    dim_product = build_dim_product(raw["product"])
    dim_customer = build_dim_customer(raw["customer"], raw["generations"])
    dim_staff = build_dim_staff(raw["staff"])
    dim_time = build_dim_time(raw["sales_receipts"])

    # buat set ID yang valid dari dataframe customer
    valid_customer_ids = set(dim_customer["customer_id"].unique())
    # menambahkan 0 karena insert manual ID 0 nanti
    valid_customer_ids.add(0)

    conn = get_connection()
    conn.autocommit = False

    try:
        # dim_date
        bulk_insert(
            conn,
            "dim_date",
            dim_date,
            [
                "date_id",
                "full_date",
                "week_id",
                "week_desc",
                "month_number",
                "month_name",
                "quarter_number",
                "quarter_name",
                "year_number",
            ],
        )

        # dim_store
        bulk_insert(
            conn,
            "dim_store",
            dim_store,
            [
                "sales_outlet_id",
                "sales_outlet_type",
                "store_square_feet",
                "store_address",
                "store_city",
                "store_state_province",
                "store_telephone",
                "store_postal_code",
                "store_longitude",
                "store_latitude",
                "manager",
                "neighborhood",
            ],
        )

        # dim_product
        bulk_insert(
            conn,
            "dim_product",
            dim_product,
            [
                "product_id",
                "product_group",
                "product_category",
                "product_type",
                "product_name",
                "product_description",
                "unit_of_measure",
                "current_wholesale_price",
                "current_retail_price",
                "tax_exempt",
                "promo",
                "new_product",
            ],
        )

        # dim_customer (tanpa unknown dulu)
        bulk_insert(
            conn,
            "dim_customer",
            dim_customer,
            [
                "customer_id",
                "home_store",
                "customer_first_name",
                "customer_email",
                "customer_since",
                "loyalty_card_number",
                "birthdate",
                "gender",
                "birth_year",
                "generation",
            ],
        )

        # tambahkan customer_id=0 'Unknown'
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dim_customer
                    (customer_id, home_store, customer_first_name)
                VALUES
                    (0, NULL, 'Unknown')
                ON CONFLICT (customer_id) DO NOTHING;
                """
            )
        print("[OK] Insert Unknown customer (id=0)")

        # dim_staff
        bulk_insert(
            conn,
            "dim_staff",
            dim_staff,
            [
                "staff_id",
                "first_name",
                "last_name",
                "position",
                "start_date",
                "location",
            ],
        )

        # dim_time: biarkan time_id di-generate oleh postgres (identity)
        # jadi disini insert tanpa kolom time_id
        bulk_insert(
            conn,
            "dim_time",
            dim_time,
            [
                "hour_of_day",
                "minute_interval",
                "day_part",
                "is_morning_rush",
                "is_lunch_break",
                "is_after_work",
            ],
        )

        # setelah insert, ambil mapping (hour_of_day, minute_interval) => time_id
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT time_id, hour_of_day, minute_interval
                FROM dim_time
                """
            )
            rows = cur.fetchall()
        dim_time_map = {(r[1], r[2]): r[0] for r in rows}  # (hour, minute_interval) => time_id

        fact_sales = build_fact_sales(
            raw["sales_receipts"], dim_date, dim_time_map, valid_customer_ids
        )
        fact_pastry = build_fact_pastry_inventory(raw["pastry_inventory"], dim_date)
        fact_targets = build_fact_sales_target(raw["sales_targets"], dim_date)

        bulk_insert(
            conn,
            "fact_sales",
            fact_sales,
            [
                "date_id",
                "time_id",
                "sales_outlet_id",
                "staff_id",
                "customer_id",
                "product_id",
                "receipt_key",
                "transaction_id",
                "order_number",
                "line_item_id",
                "instore",
                "promo_item",
                "quantity",
                "unit_price",
                "line_item_amount",
            ],
        )

        bulk_insert(
            conn,
            "fact_pastry_inventory",
            fact_pastry,
            [
                "sales_outlet_id",
                "product_id",
                "date_id",
                "start_of_day",
                "quantity_sold",
                "waste",
            ],
        )

        bulk_insert(
            conn,
            "fact_sales_target",
            fact_targets,
            [
                "sales_outlet_id",
                "date_id_month",
                "beans_goal",
                "beverage_goal",
                "food_goal",
                "merchandise_goal",
                "total_goal",
            ],
        )

        conn.commit()
        print("[SUCCESS] ETL selesai dan commit ke database.")

        # Load vector data after main ETL is complete
        load_initial_vector_data()

    except Exception as e:
        conn.rollback()
        print("[ERROR] Terjadi error, rollback transaksi.")
        print(e)
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
