import requests
import pandas as pd
from pathlib import Path


BASE_URL = "https://ghoapi.azureedge.net/api"
OBESITY_INDICATOR_CODE = "NCD_BMI_30A"


def fetch_who_indicator_data(indicator_code: str) -> pd.DataFrame:
    """
    Fetch WHO GHO data for a specific indicator code.

    NCD_BMI_30A:
    Adult obesity prevalence, BMI >= 30,
    age-standardized estimate (%).
    """

    url = f"{BASE_URL}/{indicator_code}"
    all_rows = []

    while url:
        response = requests.get(
            url,
            timeout=60,
            headers={"Accept": "application/json"},
        )

        print(f"Request URL: {response.url}")
        print(f"Status code: {response.status_code}")

        response.raise_for_status()
        data = response.json()

        if "value" not in data:
            raise ValueError("Unexpected WHO API response format. 'value' key not found.")

        all_rows.extend(data["value"])

        # Some OData APIs paginate results using @odata.nextLink.
        url = data.get("@odata.nextLink")

    return pd.DataFrame(all_rows)


def save_raw_data(df: pd.DataFrame) -> None:
    """
    Save raw WHO obesity data locally.
    """

    output_dir = Path("data/raw/who")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "who_obesity_age_standardized_raw.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved data to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    print("Fetching WHO adult obesity data...")
    obesity_df = fetch_who_indicator_data(OBESITY_INDICATOR_CODE)

    save_raw_data(obesity_df)

    print("\nPreview:")
    print(obesity_df.head())

    print("\nColumns:")
    print(obesity_df.columns.tolist())

    print("\nUnique spatial dimensions preview:")
    if "SpatialDim" in obesity_df.columns:
        print(obesity_df["SpatialDim"].dropna().unique()[:20])

    print("\nUnique years preview:")
    if "TimeDim" in obesity_df.columns:
        print(sorted(obesity_df["TimeDim"].dropna().unique())[:20])
        print(sorted(obesity_df["TimeDim"].dropna().unique())[-20:])