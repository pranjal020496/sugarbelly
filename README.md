# Sugar Belly  
## Global Obesity Analytics & Sugar-Sensitivity Forecasting Platform

Sugar Belly is an end-to-end data analytics and machine learning application that explores global obesity trends, country-level sugar and sweeteners availability, and forward-looking obesity risk scenarios across countries and WHO regions.

The project combines data engineering, PostgreSQL analytics, SQL-based feature engineering, interactive dashboarding, machine learning model benchmarking, and sugar-sensitivity forecasting to 2030.

> **Important interpretation note:** This project analyzes associations and forecast scenarios, not causation. FAOSTAT sugar values represent national food supply availability, not exact individual-level consumption.

---

## Executive Summary

Obesity is a major global public-health challenge. At the same time, national food supply patterns provide useful context for understanding long-term population health trends.

Sugar Belly was built as a professional analytics product that integrates public-health and food-availability datasets into a structured intelligence platform. It enables users to explore global obesity prevalence, compare sugar availability patterns, evaluate regional differences, and forecast obesity outcomes under different sugar-availability scenarios.

The platform supports:

- Global and regional obesity comparison
- Sugar and sweeteners availability analysis
- Country-level trend exploration
- SQL-backed analytics views
- Machine learning model benchmarking
- Sugar-sensitivity scenario forecasting
- 2030 obesity risk comparison across countries and WHO regions
- Scenario impact analysis versus a constant-sugar baseline

This project demonstrates an end-to-end analytics workflow from raw public datasets to a PostgreSQL-backed dashboard and applied ML forecasting layer.

---

## Dashboard Preview

### Executive Overview

![Sugar Belly dashboard overview](assets/sugarbelly_dashboard_overview.png)

### Global Obesity and Sugar Availability Analytics

![Sugar Belly global overview](assets/sugarbelly_global_overview.png)

### Country-Level Trend Explorer

![Germany sugar and obesity trend](assets/germany_sugarvsobesity.png)

### Sugar-Sensitivity ML Scenario Forecast

![Sugar Belly sugar sensitivity forecast](assets/sugarbelly_ml_sugar_sensitivity_forecast.png)

### Scenario Impact vs Constant Sugar

![Sugar Belly scenario impact](assets/sugarbelly_scenario_impact.png)

### Rankings and Comparative Analytics

![Sugar Belly rankings](assets/rankings.png)

---

## Business Problem

Public-health and nutrition datasets are often published separately. This makes it difficult to analyze how obesity prevalence, food supply availability, country-level differences, and regional patterns evolve together over time.

Sugar Belly addresses this by creating a unified analytics layer that connects WHO obesity estimates with FAOSTAT sugar and sweeteners availability data.

The project is designed around the following analytical question:

> How can public-health and food-availability data be integrated into a decision-support platform that enables country comparison, regional insight generation, and forward-looking obesity scenario analysis?

---

## Data Sources

This project uses publicly available international datasets.

| Source | Dataset | Usage |
|---|---|---|
| WHO Global Health Observatory | Adult obesity prevalence, BMI >= 30, age-standardized estimate | Country-level obesity trend analysis |
| FAOSTAT Food Balance Sheets | Sugar and sweeteners food supply availability | Country-level sugar availability analysis |

Key fields used include:

- Country ISO3 code
- Country or area name
- Year
- Adult obesity prevalence
- Obesity uncertainty bounds
- WHO region
- Sugar and sweeteners supply in kg/capita/year
- Sugar and sweeteners supply in kcal/capita/day

---

## Solution Overview

Sugar Belly follows a modular analytics-product architecture.

```mermaid
flowchart LR
    A[WHO Obesity Data] --> B[Python Ingestion & Cleaning]
    C[FAOSTAT Food Balance Data] --> B
    B --> D[PostgreSQL Database]
    D --> E[SQL Analytics Views]
    E --> F[ML Feature Engineering]
    F --> G[Sugar-Sensitivity ML Model]
    G --> H[2030 Scenario Forecasts]
    E --> I[Streamlit Dashboard]
    H --> I
```

The system separates responsibilities clearly:

- **Python** handles ingestion, cleaning, database loading, model training, and forecasting
- **PostgreSQL** stores structured data and powers reusable analytics views
- **SQL views** create dashboard-ready and ML-ready datasets
- **Scikit-learn** supports model training, benchmarking, and forecasting
- **Streamlit and Plotly** deliver the interactive analytics interface

---

## Key Features

### 1. SQL-Backed Analytics Layer

The project uses PostgreSQL tables and SQL views to create a reusable analytics foundation.

Core database objects include:

- `who_obesity`
- `faostat_sugar_supply`
- `v_obesity_country_year`
- `v_sugar_obesity_country_year`
- `v_sugar_obesity_latest`
- `v_sugar_obesity_region_summary`
- `v_sugar_obesity_country_change`
- `v_ml_sugar_sensitivity_features`

This makes the application more scalable and professional than a CSV-only workflow.

---

### 2. Interactive Dashboard

The Streamlit dashboard provides:

- Executive KPI cards
- Global obesity choropleth map
- Sugar availability choropleth map
- Sugar vs obesity scatter plot with trendline
- Country-level trend explorer
- Regional comparison
- Obesity ranking
- Sugar availability ranking
- Largest obesity increase ranking
- Sugar-sensitivity forecast to 2030
- Scenario impact comparison versus constant sugar

---

### 3. Machine Learning Forecasting Layer

Sugar Belly uses a sugar-sensitivity machine learning model to estimate future obesity change under different sugar availability scenarios.

The model combines both datasets at country-year level:

- WHO obesity prevalence data
- FAOSTAT sugar and sweeteners availability data

The model does **not** switch when the user selects different scenarios. The trained model remains fixed. The dashboard changes the sugar availability input assumption, and the same model estimates the associated obesity forecast.

---

## Machine Learning Methodology

The machine learning model is designed to estimate how future obesity changes are associated with current obesity levels, sugar availability, sugar trends, year, and WHO region.

Instead of directly predicting next-year obesity prevalence, the model predicts future obesity movement.

The target variable is:

```text
target_obesity_change_3yr = obesity_pct in year + 3 - obesity_pct in current year
```

This means the model learns to estimate how much obesity prevalence changes over the following three years.

Input features include:

- Current obesity prevalence
- One-year obesity change
- Three-year obesity change
- Current sugar availability
- Sugar availability one year ago
- Sugar availability three years ago
- One-year sugar availability change
- Three-year sugar availability change
- Sugar availability in kcal/capita/day
- WHO region
- Year

The final forecast is calculated as:

```text
future obesity = current obesity + predicted obesity change
```

For yearly forecasts through 2030, the predicted three-year change is annualized.

This approach combines both obesity and sugar availability data into one forecast output while keeping the interpretation focused on association-based scenario analysis.

---

## Model Benchmarking

The sugar-sensitivity model is trained and evaluated using historical country-year observations.

Candidate models include:

| Model | Purpose |
|---|---|
| Naive Previous 3-Year Change | Baseline using recent historical obesity change |
| Linear Regression | Interpretable statistical ML model |
| Ridge Regression | Regularized linear model |
| Random Forest Regressor | Nonlinear tabular ML model |

The selected model is saved as:

```text
models/sugar_sensitivity_model.joblib
```

Model outputs are saved to:

```text
reports/sugar_sensitivity_metrics.csv
reports/sugar_sensitivity_test_predictions.csv
```

The model is selected for scenario analysis based on predictive performance and interpretability.

---

## Sugar Availability Scenario Forecasting

Because future sugar availability is unknown, the dashboard allows scenario-based analysis.

The scenario options are:

- Sugar -10% by 2030
- Sugar -5% by 2030
- Sugar -2% by 2030
- Constant sugar
- Sugar +2% by 2030
- Sugar +5% by 2030
- Sugar +10% by 2030

In the dashboard, users select:

```text
±2%, ±5%, or ±10% sugar availability change by 2030
```

For each selected range, the dashboard compares:

- Reduced sugar scenario
- Constant sugar scenario
- Increased sugar scenario

The forecast process is:

1. Start from the latest available country-year data
2. Apply the selected sugar availability scenario through 2030
3. Build ML features using current obesity, sugar availability, sugar trend, year, and WHO region
4. Predict expected three-year obesity change
5. Annualize the predicted change
6. Add the annualized change to current obesity prevalence
7. Repeat until 2030

The dashboard also shows a scenario impact chart that compares each scenario against the constant-sugar baseline.

---

## Technology Stack

| Layer | Tools |
|---|---|
| Data ingestion | Python, Requests, Pandas |
| Data cleaning | Python, Pandas |
| Database | PostgreSQL |
| SQL analytics | SQL views, joins, window functions, correlation |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Machine learning | Scikit-learn |
| Model persistence | Joblib |
| Version control | Git, GitHub |

---

## Project Structure

```text
sugarbelly/
├── app/
│   └── dashboard.py
├── assets/
│   ├── sugarbelly_logo.png
│   ├── sugarbelly_dashboard_overview.png
│   ├── sugarbelly_global_overview.png
│   ├── germany_sugarvsobesity.png
│   ├── sugarbelly_ml_sugar_sensitivity_forecast.png
│   ├── sugarbelly_scenario_impact.png
│   └── rankings.png
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   └── sugar_sensitivity_model.joblib
├── reports/
│   ├── sugar_sensitivity_metrics.csv
│   ├── sugar_sensitivity_test_predictions.csv
│   └── sugar_sensitivity_forecasts_2030.csv
├── sql/
│   ├── 01_create_tables.sql
│   ├── 02_basic_obesity_queries.sql
│   ├── 03_create_obesity_views.sql
│   ├── 04_create_sugar_table.sql
│   ├── 05_create_sugar_obesity_views.sql
│   ├── 06_create_ml_features.sql
│   └── 07_create_sugar_sensitivity_features.sql
├── src/
│   ├── cleaning/
│   ├── database/
│   ├── ingestion/
│   └── models/
│       ├── train_sugar_sensitivity_model.py
│       └── forecast_sugar_sensitivity_to_2030.py
├── requirements.txt
└── README.md
```

> Note: Trained `.joblib` model files may be excluded from GitHub by `.gitignore`. If the model file is not present locally, retrain it using the commands below.

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/pranjal020496/sugarbelly.git
cd sugarbelly
```

### 2. Create and activate environment

```bash
conda create -n sugarbelly python=3.11
conda activate sugarbelly
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create PostgreSQL database

Make sure PostgreSQL is running, then create the database:

```bash
createdb sugarbelly
```

### 5. Create tables and ingest data

Create WHO obesity table:

```bash
psql -d sugarbelly -f sql/01_create_tables.sql
```

Fetch, clean, and load WHO obesity data:

```bash
python src/ingestion/fetch_who_obesity.py
python src/cleaning/clean_who_obesity.py
python src/database/load_who_obesity.py
```

Create and load FAOSTAT sugar table:

```bash
python src/ingestion/fetch_faostat_sugar.py
psql -d sugarbelly -f sql/04_create_sugar_table.sql
python src/database/load_faostat_sugar.py
```

### 6. Create SQL analytics and ML feature views

```bash
psql -d sugarbelly -f sql/03_create_obesity_views.sql
psql -d sugarbelly -f sql/05_create_sugar_obesity_views.sql
psql -d sugarbelly -f sql/06_create_ml_features.sql
psql -d sugarbelly -f sql/07_create_sugar_sensitivity_features.sql
```

### 7. Train the sugar-sensitivity model

```bash
python src/models/train_sugar_sensitivity_model.py
```

### 8. Generate sugar-sensitivity forecasts to 2030

```bash
python src/models/forecast_sugar_sensitivity_to_2030.py
```

### 9. Run the dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard opens locally at:

```text
http://localhost:8501
```

---

## Key Outputs

| Output | Location |
|---|---|
| Cleaned WHO obesity data | `data/interim/who_obesity_clean.csv` |
| Cleaned FAOSTAT sugar data | `data/interim/faostat_sugar_supply_clean.csv` |
| Sugar-sensitivity model metrics | `reports/sugar_sensitivity_metrics.csv` |
| Sugar-sensitivity test predictions | `reports/sugar_sensitivity_test_predictions.csv` |
| Sugar-sensitivity 2030 forecasts | `reports/sugar_sensitivity_forecasts_2030.csv` |
| Streamlit dashboard | `app/dashboard.py` |

---

## Professional Relevance

This project demonstrates skills relevant to analytics consulting, data engineering, business intelligence, and applied machine learning roles.

It includes:

- End-to-end data pipeline development
- Public data ingestion and cleaning
- Relational database design
- SQL joins, views, window functions, and analytical transformations
- Dashboard development for executive-level insight delivery
- Machine learning feature engineering
- Model benchmarking against naive baselines
- Time-based train/test evaluation
- Forecasting and scenario analysis
- Sensitivity modelling for decision support
- GitHub-based project documentation and version control

---

## Limitations

This project should be interpreted carefully.

- FAOSTAT sugar values represent national food supply availability, not exact personal consumption.
- The analysis identifies associations, not causal relationships.
- Forecasts are scenario-based estimates, not guaranteed future outcomes.
- Sugar availability scenarios are analytical assumptions, not policy predictions.
- WHO obesity values are estimates and may be revised over time.
- Country-level observational data cannot prove individual-level behavioral effects.
- Additional variables such as income, urbanization, demographics, and physical activity could improve future model versions.

---

## Future Enhancements

Potential next steps include:

- Add GDP per capita, urbanization, and population structure features
- Add population-weighted regional summaries
- Add model explainability using feature importance or SHAP
- Add clustering to identify country risk segments
- Add automated data refresh pipeline
- Deploy the dashboard publicly
- Add Docker support for reproducible deployment
- Add a simplified CSV-backed deployment version for Streamlit Community Cloud

---

## Author

**Pranjal**  
Data Analytics, Machine Learning, and Public Health Intelligence Project

---