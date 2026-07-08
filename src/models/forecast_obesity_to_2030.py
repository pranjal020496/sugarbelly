from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine


DATABASE_URL = "postgresql+psycopg2:///sugarbelly"

MODEL_PATH = Path("models/obesity_forecast_model.joblib")
OUTPUT_PATH = Path("reports/obesity_forecasts_2030.csv")

FORECAST_START_YEAR = 2024
FORECAST_END_YEAR = 2030


SCENARIOS = {
    "Reduced sugar scenario": -0.01,
    "Constant sugar scenario": 0.00,
    "Increased sugar scenario": 0.01,
}


FEATURE_COLUMNS = [
    "year",
    "obesity_pct",
    "obesity_lag_1",
    "obesity_lag_2",
    "obesity_lag_3",
    "obesity_change_1yr",
    "obesity_change_3yr",
    "sugar_supply_kg_per_capita",
    "sugar_supply_kcal_per_capita_day",
    "sugar_lag_1",
    "sugar_lag_3",
    "sugar_change_1yr",
    "sugar_change_3yr",
    "who_region",
]


def load_model():
    """
    Load the trained machine learning model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run src/models/train_obesity_forecast.py first."
        )

    return joblib.load(MODEL_PATH)


def load_historical_data() -> pd.DataFrame:
    """
    Load historical joined sugar-obesity data from PostgreSQL.
    """

    engine = create_engine(DATABASE_URL)

    query = """
    SELECT
        iso3,
        country,
        year,
        obesity_pct,
        sugar_supply_kg_per_capita,
        sugar_supply_kcal_per_capita_day,
        who_region,
        who_region_code
    FROM v_sugar_obesity_country_year
    ORDER BY iso3, year;
    """

    df = pd.read_sql(query, engine)

    numeric_columns = [
        "year",
        "obesity_pct",
        "sugar_supply_kg_per_capita",
        "sugar_supply_kcal_per_capita_day",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "iso3",
            "country",
            "year",
            "obesity_pct",
            "sugar_supply_kg_per_capita",
            "sugar_supply_kcal_per_capita_day",
            "who_region",
        ]
    ).copy()

    df["year"] = df["year"].astype(int)

    print("Loaded historical joined data.")
    print(f"Rows: {len(df)}")
    print(f"Countries: {df['iso3'].nunique()}")
    print(f"Year range: {df['year'].min()} to {df['year'].max()}")

    return df


def build_feature_row(country_history: pd.DataFrame, input_year: int) -> dict | None:
    """
    Build one ML input row.

    The model uses the input year to predict obesity in input_year + 1.
    """

    by_year = country_history.set_index("year")

    required_years = [
        input_year,
        input_year - 1,
        input_year - 2,
        input_year - 3,
    ]

    for year in required_years:
        if year not in by_year.index:
            return None

    current = by_year.loc[input_year]
    lag_1 = by_year.loc[input_year - 1]
    lag_2 = by_year.loc[input_year - 2]
    lag_3 = by_year.loc[input_year - 3]

    return {
        "year": input_year,
        "obesity_pct": current["obesity_pct"],
        "obesity_lag_1": lag_1["obesity_pct"],
        "obesity_lag_2": lag_2["obesity_pct"],
        "obesity_lag_3": lag_3["obesity_pct"],
        "obesity_change_1yr": current["obesity_pct"] - lag_1["obesity_pct"],
        "obesity_change_3yr": current["obesity_pct"] - lag_3["obesity_pct"],
        "sugar_supply_kg_per_capita": current["sugar_supply_kg_per_capita"],
        "sugar_supply_kcal_per_capita_day": current[
            "sugar_supply_kcal_per_capita_day"
        ],
        "sugar_lag_1": lag_1["sugar_supply_kg_per_capita"],
        "sugar_lag_3": lag_3["sugar_supply_kg_per_capita"],
        "sugar_change_1yr": current["sugar_supply_kg_per_capita"]
        - lag_1["sugar_supply_kg_per_capita"],
        "sugar_change_3yr": current["sugar_supply_kg_per_capita"]
        - lag_3["sugar_supply_kg_per_capita"],
        "who_region": current["who_region"],
    }


def apply_sugar_scenario(
    latest_value: float,
    annual_change_rate: float,
    years_after_latest: int,
) -> float:
    """
    Apply a sugar scenario.

    Example:
    - annual_change_rate = -0.01 means sugar falls by 1% per year.
    - annual_change_rate = 0.00 means sugar stays constant.
    - annual_change_rate = 0.01 means sugar rises by 1% per year.
    """

    return latest_value * ((1 + annual_change_rate) ** years_after_latest)


def forecast_country_scenario(
    model,
    country_history: pd.DataFrame,
    forecast_end_year: int,
    scenario_name: str,
    annual_sugar_change_rate: float,
) -> list[dict]:
    """
    Forecast one country under one sugar scenario.
    """

    country_history = country_history.sort_values("year").copy()

    latest_actual_year = int(country_history["year"].max())
    forecast_start_year = latest_actual_year + 1

    if forecast_start_year > forecast_end_year:
        return []

    latest_row = country_history.iloc[-1]

    iso3 = latest_row["iso3"]
    country = latest_row["country"]
    who_region = latest_row["who_region"]
    who_region_code = latest_row["who_region_code"]

    latest_sugar_kg = float(latest_row["sugar_supply_kg_per_capita"])
    latest_sugar_kcal = float(latest_row["sugar_supply_kcal_per_capita_day"])

    forecasts = []
    working_history = country_history.copy()

    for forecast_year in range(forecast_start_year, forecast_end_year + 1):
        input_year = forecast_year - 1

        feature_row = build_feature_row(working_history, input_year)

        if feature_row is None:
            break

        X_forecast = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)

        prediction = float(model.predict(X_forecast)[0])
        prediction = max(0.0, min(100.0, prediction))

        years_after_latest = forecast_year - latest_actual_year

        scenario_sugar_kg = apply_sugar_scenario(
            latest_value=latest_sugar_kg,
            annual_change_rate=annual_sugar_change_rate,
            years_after_latest=years_after_latest,
        )

        scenario_sugar_kcal = apply_sugar_scenario(
            latest_value=latest_sugar_kcal,
            annual_change_rate=annual_sugar_change_rate,
            years_after_latest=years_after_latest,
        )

        forecasts.append(
            {
                "iso3": iso3,
                "country": country,
                "who_region": who_region,
                "who_region_code": who_region_code,
                "scenario": scenario_name,
                "annual_sugar_change_rate": annual_sugar_change_rate,
                "forecast_year": forecast_year,
                "forecast_obesity_pct": prediction,
                "input_year": input_year,
                "sugar_supply_kg_per_capita_assumption": scenario_sugar_kg,
                "sugar_supply_kcal_per_capita_day_assumption": scenario_sugar_kcal,
            }
        )

        new_row = {
            "iso3": iso3,
            "country": country,
            "year": forecast_year,
            "obesity_pct": prediction,
            "sugar_supply_kg_per_capita": scenario_sugar_kg,
            "sugar_supply_kcal_per_capita_day": scenario_sugar_kcal,
            "who_region": who_region,
            "who_region_code": who_region_code,
        }

        working_history = pd.concat(
            [working_history, pd.DataFrame([new_row])],
            ignore_index=True,
        )

    return forecasts


def run_forecasts() -> pd.DataFrame:
    """
    Run forecasts for all countries and all sugar scenarios.
    """

    model = load_model()
    historical_df = load_historical_data()

    all_forecasts = []

    for iso3, country_history in historical_df.groupby("iso3"):
        for scenario_name, annual_sugar_change_rate in SCENARIOS.items():
            scenario_forecasts = forecast_country_scenario(
                model=model,
                country_history=country_history,
                forecast_end_year=FORECAST_END_YEAR,
                scenario_name=scenario_name,
                annual_sugar_change_rate=annual_sugar_change_rate,
            )

            all_forecasts.extend(scenario_forecasts)

    forecast_df = pd.DataFrame(all_forecasts)

    if forecast_df.empty:
        raise ValueError("No forecasts were created. Check historical data coverage.")

    forecast_df = forecast_df[
        forecast_df["forecast_year"] >= FORECAST_START_YEAR
    ].copy()

    forecast_df = forecast_df.sort_values(
        ["iso3", "scenario", "forecast_year"]
    )

    print("\nCreated scenario forecasts.")
    print(f"Rows: {len(forecast_df)}")
    print(f"Countries: {forecast_df['iso3'].nunique()}")
    print(f"Scenarios: {forecast_df['scenario'].nunique()}")
    print(
        f"Forecast year range: "
        f"{forecast_df['forecast_year'].min()} to {forecast_df['forecast_year'].max()}"
    )

    return forecast_df


def save_forecasts(forecast_df: pd.DataFrame) -> None:
    """
    Save scenario forecasts to CSV.
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved forecasts to: {OUTPUT_PATH}")

    print("\nScenario counts:")
    print(forecast_df["scenario"].value_counts())

    print("\nPreview:")
    print(
        forecast_df[
            [
                "iso3",
                "country",
                "scenario",
                "forecast_year",
                "forecast_obesity_pct",
                "sugar_supply_kg_per_capita_assumption",
            ]
        ].head(30)
    )


if __name__ == "__main__":
    forecasts = run_forecasts()
    save_forecasts(forecasts)