from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine


DATABASE_URL = "postgresql+psycopg2:///sugarbelly"

MODEL_PATH = Path("models/sugar_sensitivity_model.joblib")
OUTPUT_PATH = Path("reports/sugar_sensitivity_forecasts_2030.csv")

FORECAST_START_YEAR = 2024
FORECAST_END_YEAR = 2030
HORIZON_YEARS = 3


SCENARIOS = {
    "Sugar -10% by 2030": -0.10,
    "Sugar -5% by 2030": -0.05,
    "Sugar -2% by 2030": -0.02,
    "Constant sugar": 0.00,
    "Sugar +2% by 2030": 0.02,
    "Sugar +5% by 2030": 0.05,
    "Sugar +10% by 2030": 0.10,
}


FEATURE_COLUMNS = [
    "year",
    "obesity_pct",
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


def load_model_bundle():
    """
    Load the trained sugar sensitivity model bundle.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run src/models/train_sugar_sensitivity_model.py first."
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(
            "Model file is not a valid sugar sensitivity model bundle."
        )

    return bundle


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


def apply_sugar_scenario(
    latest_value: float,
    total_change_by_2030: float,
    forecast_year: int,
    latest_actual_year: int,
    forecast_end_year: int,
) -> float:
    """
    Apply a total sugar availability change by 2030.

    Example:
    - -0.10 means sugar availability is 10% lower by 2030.
    -  0.00 means sugar availability stays constant.
    -  0.10 means sugar availability is 10% higher by 2030.
    """

    total_years = forecast_end_year - latest_actual_year

    if total_years <= 0:
        return latest_value

    progress = (forecast_year - latest_actual_year) / total_years
    progress = max(0.0, min(1.0, progress))

    return latest_value * (1 + total_change_by_2030 * progress)


def build_feature_row(country_history: pd.DataFrame, input_year: int) -> dict | None:
    """
    Build one ML input row for the sugar sensitivity model.

    The model predicts 3-year obesity change from the current country-year state.
    """

    by_year = country_history.set_index("year")

    required_years = [
        input_year,
        input_year - 1,
        input_year - 3,
    ]

    for year in required_years:
        if year not in by_year.index:
            return None

    current = by_year.loc[input_year]
    lag_1 = by_year.loc[input_year - 1]
    lag_3 = by_year.loc[input_year - 3]

    return {
        "year": input_year,
        "obesity_pct": current["obesity_pct"],
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


def forecast_country_scenario(
    model,
    selected_model_name: str,
    country_history: pd.DataFrame,
    forecast_end_year: int,
    scenario_name: str,
    total_sugar_change_by_2030: float,
) -> list[dict]:
    """
    Forecast one country under one sugar availability scenario.

    The model predicts a 3-year obesity change.
    For yearly forecasts, the predicted 3-year change is annualized.
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

        predicted_obesity_change_3yr = float(model.predict(X_forecast)[0])
        predicted_annualized_change = predicted_obesity_change_3yr / HORIZON_YEARS

        current_obesity = float(feature_row["obesity_pct"])
        forecast_obesity_pct = current_obesity + predicted_annualized_change
        forecast_obesity_pct = max(0.0, min(100.0, forecast_obesity_pct))

        scenario_sugar_kg = apply_sugar_scenario(
            latest_value=latest_sugar_kg,
            total_change_by_2030=total_sugar_change_by_2030,
            forecast_year=forecast_year,
            latest_actual_year=latest_actual_year,
            forecast_end_year=forecast_end_year,
        )

        scenario_sugar_kcal = apply_sugar_scenario(
            latest_value=latest_sugar_kcal,
            total_change_by_2030=total_sugar_change_by_2030,
            forecast_year=forecast_year,
            latest_actual_year=latest_actual_year,
            forecast_end_year=forecast_end_year,
        )

        forecasts.append(
            {
                "iso3": iso3,
                "country": country,
                "who_region": who_region,
                "who_region_code": who_region_code,
                "scenario": scenario_name,
                "total_sugar_change_by_2030": total_sugar_change_by_2030,
                "forecast_year": forecast_year,
                "forecast_obesity_pct": forecast_obesity_pct,
                "input_year": input_year,
                "predicted_obesity_change_3yr": predicted_obesity_change_3yr,
                "predicted_annualized_change": predicted_annualized_change,
                "sugar_supply_kg_per_capita_assumption": scenario_sugar_kg,
                "sugar_supply_kcal_per_capita_day_assumption": scenario_sugar_kcal,
                "model_name": selected_model_name,
                "model_type": "sugar_sensitivity_model",
            }
        )

        new_row = {
            "iso3": iso3,
            "country": country,
            "year": forecast_year,
            "obesity_pct": forecast_obesity_pct,
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
    Run sugar sensitivity forecasts for all countries and scenarios.
    """

    model_bundle = load_model_bundle()
    model = model_bundle["model"]
    selected_model_name = model_bundle.get(
        "selected_model_name",
        "sugar_sensitivity_model",
    )

    historical_df = load_historical_data()

    all_forecasts = []

    for iso3, country_history in historical_df.groupby("iso3"):
        for scenario_name, total_sugar_change_by_2030 in SCENARIOS.items():
            scenario_forecasts = forecast_country_scenario(
                model=model,
                selected_model_name=selected_model_name,
                country_history=country_history,
                forecast_end_year=FORECAST_END_YEAR,
                scenario_name=scenario_name,
                total_sugar_change_by_2030=total_sugar_change_by_2030,
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

    print("\nCreated sugar sensitivity forecasts.")
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
    Save sugar sensitivity forecasts to CSV.
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
                "total_sugar_change_by_2030",
                "forecast_year",
                "forecast_obesity_pct",
                "predicted_obesity_change_3yr",
                "sugar_supply_kg_per_capita_assumption",
            ]
        ].head(30)
    )


if __name__ == "__main__":
    forecasts = run_forecasts()
    save_forecasts(forecasts)