import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from io import BytesIO

import pandas as pd
import pycountry
import requests


DATASETS_XML_URL = "https://bulks-faostat.fao.org/production/datasets_E.xml"

RAW_DIR = Path("data/raw/faostat")
INTERIM_DIR = Path("data/interim")

RAW_ZIP_FILE = RAW_DIR / "faostat_food_balances.zip"
OUTPUT_FILE = INTERIM_DIR / "faostat_sugar_supply_clean.csv"


def download_file(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    print(f"Request URL: {response.url}")
    print(f"Status code: {response.status_code}")
    response.raise_for_status()
    return response.content


def parse_faostat_datasets(xml_content: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_content)

    rows = []

    for dataset in root:
        row = {}
        for child in dataset:
            tag = child.tag.split("}")[-1]
            row[tag] = child.text
        if row:
            rows.append(row)

    return pd.DataFrame(rows)


def find_food_balance_dataset(datasets_df: pd.DataFrame) -> dict:
    print("\nAvailable metadata columns:")
    print(datasets_df.columns.tolist())

    text = datasets_df.fillna("").astype(str).apply(
        lambda row: " ".join(row.values),
        axis=1,
    )

    candidates = datasets_df[
        text.str.contains("Food Balances", case=False, na=False)
        | text.str.contains("Food Balance Sheets", case=False, na=False)
    ].copy()

    if candidates.empty:
        raise ValueError("Could not find a FAOSTAT Food Balances dataset in metadata.")

    print("\nFood balance dataset candidates:")
    display_cols = [
        col for col in candidates.columns
        if col.lower() in {
            "datasetcode",
            "datasetname",
            "domaincode",
            "domainname",
            "filelocation",
            "dateupdate",
            "filerows",
            "filesize",
        }
    ]

    print(candidates[display_cols].head(20).to_string(index=False))

    # Prefer the modern Food Balances dataset if available.
    preferred_mask = text.loc[candidates.index].str.contains(
        "Food Balances \\(2010-\\)",
        case=False,
        na=False,
        regex=True,
    )

    if preferred_mask.any():
        chosen = candidates.loc[preferred_mask].iloc[0].to_dict()
    else:
        chosen = candidates.iloc[0].to_dict()

    print("\nChosen FAOSTAT dataset:")
    for key, value in chosen.items():
        if key.lower() in {"datasetcode", "datasetname", "domaincode", "domainname", "filelocation"}:
            print(f"{key}: {value}")

    if "FileLocation" not in chosen or not chosen["FileLocation"]:
        raise ValueError("Chosen dataset does not contain FileLocation.")

    return chosen


def m49_to_iso3(value) -> str | None:
    """
    Convert FAOSTAT M49 numeric country code to ISO3.
    Example: 276 -> DEU, 356 -> IND, 840 -> USA.
    """

    if pd.isna(value):
        return None

    code = str(value).replace("'", "").strip()

    if code.endswith(".0"):
        code = code[:-2]

    code = code.zfill(3)

    country = pycountry.countries.get(numeric=code)

    if country is None:
        return None

    return country.alpha_3


def find_column(columns: list[str], possible_names: list[str]) -> str:
    normalized = {col.lower().strip(): col for col in columns}

    for name in possible_names:
        key = name.lower().strip()
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not find any of these columns: {possible_names}. "
        f"Available columns: {columns}"
    )


def clean_food_balance_zip(zip_path: Path) -> pd.DataFrame:
    filtered_chunks = []

    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]

        if not csv_files:
            raise ValueError("No CSV file found inside FAOSTAT zip.")

        csv_name = csv_files[0]
        print(f"\nReading CSV inside zip: {csv_name}")

        with zf.open(csv_name) as file:
            header_df = pd.read_csv(file, nrows=0)

        columns = header_df.columns.tolist()
        print("\nCSV columns:")
        print(columns)

        area_code_col = find_column(columns, ["Area Code (M49)", "Area Code"])
        area_col = find_column(columns, ["Area"])
        item_col = find_column(columns, ["Item"])
        element_col = find_column(columns, ["Element"])
        year_col = find_column(columns, ["Year"])
        unit_col = find_column(columns, ["Unit"])
        value_col = find_column(columns, ["Value"])

        with zf.open(csv_name) as file:
            reader = pd.read_csv(file, chunksize=200_000, low_memory=False)

            for i, chunk in enumerate(reader, start=1):
                chunk = chunk[
                    [
                        area_code_col,
                        area_col,
                        item_col,
                        element_col,
                        year_col,
                        unit_col,
                        value_col,
                    ]
                ].copy()

                sugar_mask = chunk[item_col].astype(str).str.contains(
                    "Sugar & Sweeteners",
                    case=False,
                    na=False,
                    regex=False,
                )

                element_mask = chunk[element_col].astype(str).isin(
                    [
                        "Food supply quantity (kg/capita/yr)",
                        "Food supply (kcal/capita/day)",
                    ]
                )

                filtered = chunk[sugar_mask & element_mask].copy()

                if not filtered.empty:
                    filtered_chunks.append(filtered)

                print(f"Processed chunk {i}, kept rows: {len(filtered)}")

    if not filtered_chunks:
        raise ValueError("No sugar/sweeteners rows found in FAOSTAT file.")

    df = pd.concat(filtered_chunks, ignore_index=True)

    df = df.rename(
        columns={
            area_code_col: "area_code_m49",
            area_col: "country",
            item_col: "item",
            element_col: "element",
            year_col: "year",
            unit_col: "unit",
            value_col: "value",
        }
    )

    df["iso3"] = df["area_code_m49"].apply(m49_to_iso3)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["iso3", "year", "value"])

    pivot = df.pivot_table(
        index=["iso3", "country", "year"],
        columns="element",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivot = pivot.rename(
        columns={
            "Food supply quantity (kg/capita/yr)": "sugar_supply_kg_per_capita",
            "Food supply (kcal/capita/day)": "sugar_supply_kcal_per_capita_day",
        }
    )
    pivot.columns.name = None

    expected_cols = [
        "iso3",
        "country",
        "year",
        "sugar_supply_kg_per_capita",
        "sugar_supply_kcal_per_capita_day",
    ]

    for col in expected_cols:
        if col not in pivot.columns:
            pivot[col] = pd.NA

    pivot = pivot[expected_cols].copy()
    pivot = pivot.sort_values(["iso3", "year"])

    return pivot


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching FAOSTAT bulk dataset metadata...")
    xml_content = download_file(DATASETS_XML_URL)

    datasets = parse_faostat_datasets(xml_content)
    chosen_dataset = find_food_balance_dataset(datasets)

    food_balance_url = chosen_dataset["FileLocation"]

    print("\nDownloading FAOSTAT Food Balances zip...")
    zip_content = download_file(food_balance_url)

    RAW_ZIP_FILE.write_bytes(zip_content)
    print(f"Saved raw FAOSTAT zip to: {RAW_ZIP_FILE}")

    print("\nCleaning sugar availability data...")
    sugar_df = clean_food_balance_zip(RAW_ZIP_FILE)

    sugar_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved cleaned sugar data to: {OUTPUT_FILE}")
    print(f"Rows: {len(sugar_df)}")
    print(f"Countries: {sugar_df['iso3'].nunique()}")
    print(f"Year range: {int(sugar_df['year'].min())} to {int(sugar_df['year'].max())}")

    print("\nPreview:")
    print(sugar_df.head(20))