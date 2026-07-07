from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine


DATABASE_URL = "postgresql+psycopg2:///sugarbelly"

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_PATH = MODEL_DIR / "obesity_forecast_model.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.csv"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"


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

TARGET_COLUMN = "target_obesity_next_year"


def load_ml_data() -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT *
    FROM v_ml_obesity_features
    WHERE target_obesity_next_year IS NOT NULL
      AND obesity_lag_1 IS NOT NULL
      AND obesity_lag_2 IS NOT NULL
      AND obesity_lag_3 IS NOT NULL
      AND sugar_lag_1 IS NOT NULL
      AND sugar_lag_3 IS NOT NULL;
    """

    df = pd.read_sql(query, engine)

    print("Loaded ML dataset.")
    print(f"Rows: {len(df)}")
    print(f"Countries: {df['iso3'].nunique()}")
    print(f"Year range: {df['year'].min()} to {df['year'].max()}")

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        col for col in FEATURE_COLUMNS
        if col != "who_region"
    ]

    categorical_features = ["who_region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor


def evaluate_model(name: str, model: Pipeline, X_test, y_test) -> dict:
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def train_models(df: pd.DataFrame):
    train_df = df[df["year"] <= 2020].copy()
    test_df = df[df["year"] > 2020].copy()

    print("\nTrain/test split:")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train years: {train_df['year'].min()} to {train_df['year'].max()}")
    print(f"Test years: {test_df['year'].min()} to {test_df['year'].max()}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    preprocessor = build_preprocessor()

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for name, regressor in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", regressor),
            ]
        )

        print(f"\nTraining model: {name}")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
        trained_models[name] = pipeline

        print(
            f"{name} | "
            f"MAE: {metrics['mae']:.3f} | "
            f"RMSE: {metrics['rmse']:.3f} | "
            f"R2: {metrics['r2']:.3f}"
        )

    metrics_df = pd.DataFrame(results).sort_values("mae")
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print(f"\nBest model by MAE: {best_model_name}")

    test_predictions = test_df[
        ["iso3", "country", "year", "who_region", TARGET_COLUMN]
    ].copy()

    test_predictions["prediction"] = best_model.predict(X_test)
    test_predictions["error"] = (
        test_predictions["prediction"] - test_predictions[TARGET_COLUMN]
    )

    return best_model_name, best_model, metrics_df, test_predictions


def save_outputs(best_model_name, best_model, metrics_df, test_predictions) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    metrics_df.to_csv(METRICS_PATH, index=False)
    test_predictions.to_csv(PREDICTIONS_PATH, index=False)

    print("\nSaved outputs:")
    print(f"Best model: {best_model_name}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Metrics path: {METRICS_PATH}")
    print(f"Predictions path: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    ml_df = load_ml_data()
    best_name, best_model, metrics, predictions = train_models(ml_df)
    save_outputs(best_name, best_model, metrics, predictions)