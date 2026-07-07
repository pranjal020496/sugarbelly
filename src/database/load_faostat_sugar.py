import pandas as pd
import psycopg2
from pathlib import Path
from io import StringIO


CLEAN_FILE = Path("data/interim/faostat_sugar_supply_clean.csv")
DATABASE_NAME = "sugarbelly"
TABLE_NAME = "faostat_sugar_supply"


COLUMNS = [
    "iso3",
    "country",
    "year",
    "sugar_supply_kg_per_capita",
    "sugar_supply_kcal_per_capita_day",
]


def load_clean_data() -> pd.DataFrame:
    """
    Load cleaned FAOSTAT sugar supply CSV.
    """

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Clean file not found: {CLEAN_FILE}. "
            "Run src/ingestion/fetch_faostat_sugar.py first."
        )

    df = pd.read_csv(CLEAN_FILE)

    missing_columns = [col for col in COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[COLUMNS].copy()

    before_filter = len(df)

    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    df = df[df["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["sugar_supply_kg_per_capita"] = pd.to_numeric(
        df["sugar_supply_kg_per_capita"],
        errors="coerce",
    )
    df["sugar_supply_kcal_per_capita_day"] = pd.to_numeric(
        df["sugar_supply_kcal_per_capita_day"],
        errors="coerce",
    )

    df = df.dropna(subset=["iso3", "country", "year"])
    df["year"] = df["year"].astype(int)

    after_filter = len(df)

    print("Loaded cleaned FAOSTAT sugar CSV.")
    print(f"Rows before filter: {before_filter}")
    print(f"Rows after filter: {after_filter}")
    print(f"Removed rows: {before_filter - after_filter}")
    print(f"Countries: {df['iso3'].nunique()}")
    print(f"Year range: {df['year'].min()} to {df['year'].max()}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def load_to_postgres(df: pd.DataFrame) -> None:
    """
    Load cleaned FAOSTAT sugar data into PostgreSQL using COPY.
    """

    connection = psycopg2.connect(dbname=DATABASE_NAME)

    try:
        with connection:
            with connection.cursor() as cursor:
                print(f"Clearing existing rows from {TABLE_NAME}...")
                cursor.execute(f"TRUNCATE TABLE {TABLE_NAME};")

                print("Preparing CSV buffer...")
                buffer = StringIO()

                df.to_csv(
                    buffer,
                    index=False,
                    header=True,
                    na_rep="",
                )

                buffer.seek(0)

                print("Loading rows into PostgreSQL using COPY...")

                copy_sql = f"""
                    COPY {TABLE_NAME} (
                        iso3,
                        country,
                        year,
                        sugar_supply_kg_per_capita,
                        sugar_supply_kcal_per_capita_day
                    )
                    FROM STDIN
                    WITH CSV HEADER
                """

                cursor.copy_expert(copy_sql, buffer)

        print("Loaded FAOSTAT sugar data into PostgreSQL.")
        print(f"Rows loaded: {len(df)}")

    finally:
        connection.close()


def verify_load() -> None:
    """
    Verify sugar data load.
    """

    connection = psycopg2.connect(dbname=DATABASE_NAME)

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT iso3) AS countries,
                    MIN(year) AS min_year,
                    MAX(year) AS max_year
                FROM faostat_sugar_supply;
                """
            )

            row = cursor.fetchone()

            print("\nVerification query result:")
            print(
                f"Rows: {row[0]}, "
                f"Countries: {row[1]}, "
                f"Year range: {row[2]} to {row[3]}"
            )

    finally:
        connection.close()


if __name__ == "__main__":
    sugar_df = load_clean_data()
    load_to_postgres(sugar_df)
    verify_load()