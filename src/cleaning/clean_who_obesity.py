import pandas as pd
from pathlib import Path


RAW_FILE = Path("data/raw/who/who_obesity_age_standardized_raw.csv")
OUTPUT_FILE = Path("data/interim/who_obesity_clean.csv")


def load_raw_data() -> pd.DataFrame:
    """
    Load raw WHO obesity data.
    """

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw file not found: {RAW_FILE}. "
            "Run src/ingestion/fetch_who_obesity.py first."
        )

    df = pd.read_csv(RAW_FILE)
    return df


def inspect_raw_data(df: pd.DataFrame) -> None:
    """
    Print useful information before cleaning.
    """

    print("\nRaw data shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDim1 values, usually sex categories:")
    print(df["Dim1"].value_counts(dropna=False))

    print("\nTime range:")
    print(df["TimeDim"].min(), "to", df["TimeDim"].max())

    print("\nNumber of countries/areas:")
    print(df["SpatialDim"].nunique())


def clean_obesity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean WHO obesity data into a country-year-sex table.
    """

    required_columns = [
        "SpatialDim",
        "TimeDim",
        "Dim1",
        "NumericValue",
        "Low",
        "High",
        "ParentLocation",
        "ParentLocationCode",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    clean_df = df[required_columns].copy()

    clean_df = clean_df.rename(
        columns={
            "SpatialDim": "iso3",
            "TimeDim": "year",
            "Dim1": "sex_code",
            "NumericValue": "obesity_pct",
            "Low": "obesity_pct_low",
            "High": "obesity_pct_high",
            "ParentLocation": "who_region",
            "ParentLocationCode": "who_region_code",
        }
    )

    sex_map = {
        "SEX_BTSX": "Both sexes",
        "SEX_MLE": "Male",
        "SEX_FMLE": "Female",
    }

    clean_df["sex"] = clean_df["sex_code"].map(sex_map)

    # Keep all sex categories for now, but the ML model will mainly use Both sexes.
    clean_df["sex"] = clean_df["sex"].fillna(clean_df["sex_code"])

    numeric_columns = [
        "year",
        "obesity_pct",
        "obesity_pct_low",
        "obesity_pct_high",
    ]

    for col in numeric_columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df["iso3"] = clean_df["iso3"].astype(str).str.upper().str.strip()

    clean_df = clean_df[
        [
            "iso3",
            "year",
            "sex",
            "sex_code",
            "obesity_pct",
            "obesity_pct_low",
            "obesity_pct_high",
            "who_region",
            "who_region_code",
        ]
    ]

    clean_df = clean_df.dropna(subset=["iso3", "year", "obesity_pct"])

    clean_df = clean_df.drop_duplicates(
        subset=["iso3", "year", "sex_code"],
        keep="first",
    )

    clean_df = clean_df.sort_values(["iso3", "sex", "year"])

    return clean_df


def save_clean_data(df: pd.DataFrame) -> None:
    """
    Save cleaned obesity data.
    """

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved cleaned data to: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    raw_df = load_raw_data()

    inspect_raw_data(raw_df)

    clean_df = clean_obesity_data(raw_df)

    save_clean_data(clean_df)

    print("\nCleaned data preview:")
    print(clean_df.head())

    print("\nSex categories after cleaning:")
    print(clean_df["sex"].value_counts())

    print("\nCleaned year range:")
    print(clean_df["year"].min(), "to", clean_df["year"].max())