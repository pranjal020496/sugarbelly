from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine


DATABASE_URL = "postgresql+psycopg2:///sugarbelly"

MODEL_PATH = Path("models/sugar_sensitivity_model.joblib")
METRICS_PATH = Path("reports/sugar_sensitivity_metrics.csv")
PREDICTIONS_PATH = Path("reports/sugar_sensitivity_test_predictions.csv")

TRAIN_END_YEAR = 2018

TARGET_COLUMN = "target_obesity_change_3yr"

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

NUMERIC_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "who_region",
]


def make_one_hot_encoder():
    """
    Create OneHotEncoder in a way that works across scikit-learn versions.
    """

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_training_data() -> pd.DataFrame:
    """
    Load ML-ready sugar sensitivity features from PostgreSQL.
    """

    engine = create_engine(DATABASE_URL)

    query = """
    SELECT
        iso3,
        country,
        year,
        obesity_pct,
        obesity_change_1yr,
        obesity_change_3yr,
        sugar_supply_kg_per_capita,
        sugar_supply_kcal_per_capita_day,
        sugar_lag_1,
        sugar_lag_3,
        sugar_change_1yr,
        sugar_change_3yr,
        who_region,
        target_obesity_change_3yr
    FROM v_ml_sugar_sensitivity_features
    WHERE target_obesity_change_3yr IS NOT NULL
      AND obesity_change_1yr IS NOT NULL
      AND obesity_change_3yr IS NOT NULL
      AND sugar_lag_1 IS NOT NULL
      AND sugar_lag_3 IS NOT NULL
      AND sugar_change_1yr IS NOT NULL
      AND sugar_change_3yr IS NOT NULL
      AND who_region IS NOT NULL
    ORDER BY iso3, year;
    """

    df = pd.read_sql(query, engine)

    numeric_columns = NUMERIC_FEATURES + [TARGET_COLUMN]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    df["year"] = df["year"].astype(int)

    print("Loaded sugar sensitivity ML dataset.")
    print(f"Rows: {len(df)}")
    print(f"Countries: {df['iso3'].nunique()}")
    print(f"Year range: {df['year'].min()} to {df['year'].max()}")
    print(
        "Target definition: obesity percentage-point change over the next 3 years."
    )

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Return model metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def train_models(df: pd.DataFrame):
    """
    Train and evaluate sugar sensitivity models.

    The target is future 3-year obesity change.
    """

    train_df = df[df["year"] <= TRAIN_END_YEAR].copy()
    test_df = df[df["year"] > TRAIN_END_YEAR].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train/test split produced empty train or test set. "
            "Check data coverage and TRAIN_END_YEAR."
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print("\nTrain/test split:")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train years: {train_df['year'].min()} to {train_df['year'].max()}")
    print(f"Test years: {test_df['year'].min()} to {test_df['year'].max()}")

    preprocessor = build_preprocessor()

    candidate_models = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    }

    metrics_rows = []
    trained_models = {}
    prediction_frames = []

    baseline_pred = X_test["obesity_change_3yr"].fillna(y_train.mean())
    baseline_metrics = evaluate_predictions(y_test, baseline_pred)

    metrics_rows.append(
        {
            "model": "naive_previous_3yr_change",
            "model_family": "baseline",
            "mae": baseline_metrics["mae"],
            "rmse": baseline_metrics["rmse"],
            "r2": baseline_metrics["r2"],
            "selected_for_scenarios": False,
        }
    )

    print(
        "\nNaive baseline | "
        f"MAE: {baseline_metrics['mae']:.3f} | "
        f"RMSE: {baseline_metrics['rmse']:.3f} | "
        f"R2: {baseline_metrics['r2']:.3f}"
    )

    for model_name, estimator in candidate_models.items():
        print(f"\nTraining model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        model_metrics = evaluate_predictions(y_test, y_pred)

        print(
            f"{model_name} | "
            f"MAE: {model_metrics['mae']:.3f} | "
            f"RMSE: {model_metrics['rmse']:.3f} | "
            f"R2: {model_metrics['r2']:.3f}"
        )

        trained_models[model_name] = pipeline

        metrics_rows.append(
            {
                "model": model_name,
                "model_family": "ml",
                "mae": model_metrics["mae"],
                "rmse": model_metrics["rmse"],
                "r2": model_metrics["r2"],
                "selected_for_scenarios": False,
            }
        )

        pred_df = test_df[
            [
                "iso3",
                "country",
                "year",
                "obesity_pct",
                "sugar_supply_kg_per_capita",
                TARGET_COLUMN,
            ]
        ].copy()

        pred_df["model"] = model_name
        pred_df["predicted_obesity_change_3yr"] = y_pred
        pred_df["prediction_error"] = (
            pred_df["predicted_obesity_change_3yr"] - pred_df[TARGET_COLUMN]
        )

        prediction_frames.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows)

    scenario_candidate_metrics = metrics_df[
        metrics_df["model"].isin(["linear_regression", "ridge_regression"])
    ].copy()

    selected_model_name = scenario_candidate_metrics.sort_values("mae").iloc[0][
        "model"
    ]

    metrics_df.loc[
        metrics_df["model"] == selected_model_name,
        "selected_for_scenarios",
    ] = True

    selected_model = trained_models[selected_model_name]

    print("\nSelected sugar sensitivity model:")
    print(selected_model_name)
    print(
        "Selection rule: lowest MAE among interpretable regression models "
        "for stable scenario analysis."
    )

    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    return selected_model_name, selected_model, metrics_df, predictions_df


def save_outputs(
    selected_model_name: str,
    selected_model,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    """
    Save model and reports.
    """

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": selected_model,
        "selected_model_name": selected_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "horizon_years": 3,
        "model_purpose": "Sugar sensitivity model for obesity change forecasting",
    }

    joblib.dump(model_bundle, MODEL_PATH)

    metrics_df.to_csv(METRICS_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print("\nSaved outputs:")
    print(f"Selected model: {selected_model_name}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Metrics path: {METRICS_PATH}")
    print(f"Predictions path: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    training_df = load_training_data()
    (
        selected_name,
        selected_pipeline,
        model_metrics_df,
        test_predictions_df,
    ) = train_models(training_df)

    save_outputs(
        selected_model_name=selected_name,
        selected_model=selected_pipeline,
        metrics_df=model_metrics_df,
        predictions_df=test_predictions_df,
    )