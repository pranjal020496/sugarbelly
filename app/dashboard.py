from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import create_engine


DATABASE_URL = "postgresql+psycopg2:///sugarbelly"

FORECAST_PATH = Path("reports/obesity_forecasts_2030.csv")
METRICS_PATH = Path("reports/model_metrics.csv")
LOGO_PATH = Path("assets/sugarbelly_logo.png")


st.set_page_config(
    page_title="Sugar Belly",
    page_icon="📊",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    header[data-testid="stHeader"] {
        display: none;
    }

    div[data-testid="stToolbar"] {
        display: none;
    }

    div[data-testid="stDecoration"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    .hero-text-card {
        padding: 1.8rem 2rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 32%),
            linear-gradient(135deg, #020617 0%, #0F172A 52%, #1E293B 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 42px rgba(2, 6, 23, 0.24);
        height: 390px;
    }

    .hero-eyebrow {
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .hero-text-card h1 {
        font-size: 2.6rem;
        margin-bottom: 0.75rem;
        color: #F8FAFC;
        font-weight: 850;
        letter-spacing: -0.04em;
        line-height: 1.05;
    }

    .hero-text-card p {
        font-size: 1.02rem;
        color: #CBD5E1;
        max-width: 880px;
        line-height: 1.75;
        margin-bottom: 1rem;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .hero-badge {
        padding: 0.42rem 0.68rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: #E2E8F0;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .metric-card {
        padding: 1.15rem;
        border-radius: 18px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .metric-label {
        font-size: 0.83rem;
        color: #64748B;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 1.4rem;
        margin-bottom: 0.7rem;
        letter-spacing: -0.01em;
    }

    .note {
        font-size: 0.92rem;
        color: #334155;
        background: #FFFFFF;
        border-left: 4px solid #2563EB;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        margin-top: 1rem;
        border: 1px solid #E2E8F0;
    }

    div[data-testid="stImage"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        padding: 1rem;
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
        height: 390px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 18px;
        object-fit: contain;
        max-height: 340px;
        width: 100%;
    }

    @media (max-width: 900px) {
        .hero-text-card {
            height: auto;
            min-height: 360px;
        }

        .hero-text-card h1 {
            font-size: 2.3rem;
        }

        div[data-testid="stImage"] {
            height: auto;
            min-height: 320px;
        }
    }
</style>
"""


@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "year",
        "obesity_pct",
        "obesity_pct_low",
        "obesity_pct_high",
        "sugar_supply_kg_per_capita",
        "sugar_supply_kcal_per_capita_day",
        "obesity_rank_in_year",
        "sugar_rank_in_year",
        "obesity_change_from_previous_year",
        "sugar_change_from_previous_year",
        "avg_obesity_pct",
        "avg_sugar_kg_per_capita",
        "avg_sugar_kcal_per_capita_day",
        "sugar_obesity_corr",
        "obesity_start_pct",
        "obesity_latest_pct",
        "obesity_change_pct_points",
        "sugar_start_kg_per_capita",
        "sugar_latest_kg_per_capita",
        "sugar_change_kg_per_capita",
        "obesity_increase_rank",
        "countries",
        "forecast_year",
        "forecast_obesity_pct",
        "input_year",
        "sugar_supply_kg_per_capita_assumption",
        "sugar_supply_kcal_per_capita_day_assumption",
        "mae",
        "rmse",
        "r2",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(ttl=600)
def load_data():
    engine = get_engine()

    latest = pd.read_sql(
        """
        SELECT *
        FROM v_sugar_obesity_latest;
        """,
        engine,
    )

    country_year = pd.read_sql(
        """
        SELECT *
        FROM v_sugar_obesity_country_year;
        """,
        engine,
    )

    region_summary = pd.read_sql(
        """
        SELECT *
        FROM v_sugar_obesity_region_summary;
        """,
        engine,
    )

    country_change = pd.read_sql(
        """
        SELECT *
        FROM v_sugar_obesity_country_change;
        """,
        engine,
    )

    if FORECAST_PATH.exists():
        forecasts = pd.read_csv(FORECAST_PATH)
    else:
        forecasts = pd.DataFrame()

    if METRICS_PATH.exists():
        model_metrics = pd.read_csv(METRICS_PATH)
    else:
        model_metrics = pd.DataFrame()

    return (
        clean_numeric_columns(latest),
        clean_numeric_columns(country_year),
        clean_numeric_columns(region_summary),
        clean_numeric_columns(country_change),
        clean_numeric_columns(forecasts),
        clean_numeric_columns(model_metrics),
    )


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


(
    latest_df,
    country_year_df,
    region_summary_df,
    country_change_df,
    forecast_df,
    model_metrics_df,
) = load_data()

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


hero_logo_col, hero_text_col = st.columns([1.0, 2.2])

with hero_logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.info("Add logo at assets/sugarbelly_logo.png")

with hero_text_col:
    st.markdown(
        """
        <div class="hero-text-card">
            <div class="hero-eyebrow">
                Sugar Belly / Public Health Intelligence System
            </div>
            <h1>
                Global obesity analytics with ML-SQL forecasting.
            </h1>
            <p>
                Sugar Belly is an end-to-end analytics product that combines WHO obesity estimates
                and FAOSTAT sugar availability data into a PostgreSQL-backed intelligence layer.
                The platform uses SQL feature engineering, country-year trend analysis, ML model
                benchmarking, and scenario-based forecasting to surface public-health risk signals
                across countries and WHO regions.
            </p>
            <div class="badge-row">
                <div class="hero-badge">PostgreSQL analytics layer</div>
                <div class="hero-badge">SQL feature engineering</div>
                <div class="hero-badge">Naive baseline vs ML model</div>
                <div class="hero-badge">2030 obesity forecast</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


latest_year = int(latest_df["year"].max())

with st.sidebar:
    st.header("Filters")

    regions = sorted(latest_df["who_region"].dropna().unique().tolist())

    selected_regions = st.multiselect(
        "WHO region",
        options=regions,
        default=regions,
    )

    map_metric = st.radio(
        "Map metric",
        options=[
            "Obesity prevalence (%)",
            "Sugar availability (kg/capita/year)",
        ],
    )

    st.markdown("---")

    st.caption(
        "Sugar values represent country-level food supply availability, not exact individual consumption."
    )

latest_filtered = latest_df[latest_df["who_region"].isin(selected_regions)].copy()

country_year_filtered = country_year_df[
    country_year_df["who_region"].isin(selected_regions)
].copy()

if latest_filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

with kpi_1:
    metric_card("Latest joined year", str(latest_year))

with kpi_2:
    metric_card("Countries / areas", f"{latest_filtered['iso3'].nunique():,}")

with kpi_3:
    metric_card(
        "Average obesity",
        f"{latest_filtered['obesity_pct'].mean():.2f}%",
    )

with kpi_4:
    metric_card(
        "Average sugar availability",
        f"{latest_filtered['sugar_supply_kg_per_capita'].mean():.2f} kg",
    )


st.markdown(
    '<div class="section-title">Global overview</div>',
    unsafe_allow_html=True,
)

map_col, scatter_col = st.columns([1.2, 1])

with map_col:
    if map_metric == "Obesity prevalence (%)":
        color_col = "obesity_pct"
        title = f"Adult obesity prevalence by country/area, {latest_year}"
        label = "Obesity prevalence (%)"
    else:
        color_col = "sugar_supply_kg_per_capita"
        title = f"Sugar & sweeteners availability by country/area, {latest_year}"
        label = "Sugar availability (kg/capita/year)"

    fig_map = px.choropleth(
        latest_filtered,
        locations="iso3",
        color=color_col,
        hover_name="country",
        hover_data={
            "iso3": True,
            "obesity_pct": ":.2f",
            "sugar_supply_kg_per_capita": ":.2f",
            "who_region": True,
        },
        color_continuous_scale="Blues",
        projection="natural earth",
        title=title,
        labels={color_col: label},
    )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
    )

    st.plotly_chart(fig_map, use_container_width=True)

with scatter_col:
    fig_scatter = px.scatter(
        latest_filtered,
        x="sugar_supply_kg_per_capita",
        y="obesity_pct",
        color="who_region",
        size="sugar_supply_kcal_per_capita_day",
        hover_name="country",
        hover_data={
            "iso3": True,
            "sugar_supply_kg_per_capita": ":.2f",
            "sugar_supply_kcal_per_capita_day": ":.2f",
            "obesity_pct": ":.2f",
        },
        title=f"Sugar availability vs obesity, {latest_year}",
        labels={
            "sugar_supply_kg_per_capita": "Sugar availability (kg/capita/year)",
            "obesity_pct": "Adult obesity prevalence (%)",
            "who_region": "WHO region",
        },
        trendline="ols",
    )

    fig_scatter.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text="WHO region",
    )

    st.plotly_chart(fig_scatter, use_container_width=True)


st.markdown(
    '<div class="section-title">Country explorer</div>',
    unsafe_allow_html=True,
)

country_options = (
    country_year_filtered[["iso3", "country"]]
    .drop_duplicates()
    .sort_values("country")
)

country_lookup = {
    f"{row.country} ({row.iso3})": row.iso3
    for row in country_options.itertuples(index=False)
}

selected_country_label = st.selectbox(
    "Choose a country or area",
    options=list(country_lookup.keys()),
)

selected_iso3 = country_lookup[selected_country_label]

country_data = country_year_df[
    country_year_df["iso3"] == selected_iso3
].sort_values("year")

selected_country = country_data["country"].iloc[0]

trend_fig = make_subplots(specs=[[{"secondary_y": True}]])

trend_fig.add_trace(
    go.Scatter(
        x=country_data["year"],
        y=country_data["obesity_pct"],
        mode="lines+markers",
        name="Obesity prevalence (%)",
    ),
    secondary_y=False,
)

trend_fig.add_trace(
    go.Scatter(
        x=country_data["year"],
        y=country_data["sugar_supply_kg_per_capita"],
        mode="lines+markers",
        name="Sugar availability (kg/capita/year)",
    ),
    secondary_y=True,
)

trend_fig.update_layout(
    title=f"{selected_country}: obesity and sugar availability trends",
    height=460,
    margin=dict(l=0, r=0, t=50, b=0),
)

trend_fig.update_yaxes(title_text="Obesity prevalence (%)", secondary_y=False)
trend_fig.update_yaxes(
    title_text="Sugar availability (kg/capita/year)",
    secondary_y=True,
)
trend_fig.update_xaxes(title_text="Year")

st.plotly_chart(trend_fig, use_container_width=True)


st.markdown(
    '<div class="section-title">ML forecast to 2030</div>',
    unsafe_allow_html=True,
)

if forecast_df.empty:
    st.warning(
        "Forecast file not found. Run src/models/forecast_obesity_to_2030.py to generate forecasts."
    )
else:
    forecast_filtered = forecast_df[
        forecast_df["who_region"].isin(selected_regions)
    ].copy()

    selected_forecast = forecast_df[
        forecast_df["iso3"] == selected_iso3
    ].sort_values("forecast_year")

    forecast_col_1, forecast_col_2 = st.columns([1.2, 1])

    with forecast_col_1:
        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=country_data["year"],
                y=country_data["obesity_pct"],
                mode="lines+markers",
                name="Historical obesity",
            )
        )

        if not selected_forecast.empty:
            forecast_fig.add_trace(
                go.Scatter(
                    x=selected_forecast["forecast_year"],
                    y=selected_forecast["forecast_obesity_pct"],
                    mode="lines+markers",
                    name="ML forecast",
                    line=dict(dash="dash"),
                )
            )

        forecast_fig.update_layout(
            title=f"{selected_country}: historical obesity and ML forecast to 2030",
            height=460,
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis_title="Year",
            yaxis_title="Adult obesity prevalence (%)",
        )

        st.plotly_chart(forecast_fig, use_container_width=True)

    with forecast_col_2:
        forecast_2030 = forecast_filtered[
            forecast_filtered["forecast_year"] == 2030
        ].copy()

        top_2030 = forecast_2030.sort_values(
            "forecast_obesity_pct",
            ascending=False,
        ).head(10)

        fig_2030 = px.bar(
            top_2030.sort_values("forecast_obesity_pct"),
            x="forecast_obesity_pct",
            y="country",
            orientation="h",
            title="Highest forecast obesity prevalence, 2030",
            labels={
                "forecast_obesity_pct": "Forecast obesity prevalence (%)",
                "country": "",
            },
        )

        fig_2030.update_layout(
            height=460,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        st.plotly_chart(fig_2030, use_container_width=True)

    forecast_metric_col_1, forecast_metric_col_2, forecast_metric_col_3 = st.columns(3)

    if not selected_forecast.empty:
        selected_2030_value = selected_forecast[
            selected_forecast["forecast_year"] == 2030
        ]["forecast_obesity_pct"]

        if not selected_2030_value.empty:
            with forecast_metric_col_1:
                metric_card(
                    f"{selected_country} forecast obesity in 2030",
                    f"{selected_2030_value.iloc[0]:.2f}%",
                )

    if not model_metrics_df.empty:
        best_model_row = model_metrics_df.sort_values("mae").iloc[0]

        with forecast_metric_col_2:
            metric_card(
                "Best forecast model",
                str(best_model_row["model"]),
            )

        with forecast_metric_col_3:
            metric_card(
                "Best model MAE",
                f"{best_model_row['mae']:.3f}",
            )

    with st.expander("View model evaluation results"):
        if model_metrics_df.empty:
            st.info("Model metrics file not found.")
        else:
            st.dataframe(
                model_metrics_df.round(3),
                use_container_width=True,
            )

    st.markdown(
        """
        <div class="note">
            <strong>Forecast note:</strong> The 2030 forecast uses the trained Linear Regression
            model selected during model evaluation. Future sugar availability is held constant at
            the latest observed country-level value, creating a baseline constant-sugar scenario.
            Forecasts are scenario estimates, not guaranteed future outcomes.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="section-title">Rankings</div>',
    unsafe_allow_html=True,
)

rank_col_1, rank_col_2, rank_col_3 = st.columns(3)

with rank_col_1:
    top_obesity = latest_filtered.sort_values(
        "obesity_pct",
        ascending=False,
    ).head(10)

    fig_top_obesity = px.bar(
        top_obesity.sort_values("obesity_pct"),
        x="obesity_pct",
        y="country",
        orientation="h",
        title=f"Highest obesity prevalence, {latest_year}",
        labels={
            "obesity_pct": "Obesity prevalence (%)",
            "country": "",
        },
    )

    fig_top_obesity.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    st.plotly_chart(fig_top_obesity, use_container_width=True)

with rank_col_2:
    top_sugar = latest_filtered.sort_values(
        "sugar_supply_kg_per_capita",
        ascending=False,
    ).head(10)

    fig_top_sugar = px.bar(
        top_sugar.sort_values("sugar_supply_kg_per_capita"),
        x="sugar_supply_kg_per_capita",
        y="country",
        orientation="h",
        title=f"Highest sugar availability, {latest_year}",
        labels={
            "sugar_supply_kg_per_capita": "Sugar availability (kg/capita/year)",
            "country": "",
        },
    )

    fig_top_sugar.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    st.plotly_chart(fig_top_sugar, use_container_width=True)

with rank_col_3:
    change_filtered = country_change_df[
        country_change_df["who_region"].isin(selected_regions)
    ].copy()

    top_increase = change_filtered.sort_values(
        "obesity_change_pct_points",
        ascending=False,
    ).head(10)

    fig_change = px.bar(
        top_increase.sort_values("obesity_change_pct_points"),
        x="obesity_change_pct_points",
        y="country",
        orientation="h",
        title="Largest obesity increase, 2010–2023",
        labels={
            "obesity_change_pct_points": "Increase in obesity percentage points",
            "country": "",
        },
    )

    fig_change.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    st.plotly_chart(fig_change, use_container_width=True)


st.markdown(
    '<div class="section-title">Regional comparison</div>',
    unsafe_allow_html=True,
)

latest_region = region_summary_df[
    region_summary_df["year"] == region_summary_df["year"].max()
].copy()

latest_region = latest_region[
    latest_region["who_region"].isin(selected_regions)
].copy()

region_plot_df = latest_region.melt(
    id_vars=["who_region"],
    value_vars=["avg_obesity_pct", "avg_sugar_kg_per_capita"],
    var_name="metric",
    value_name="value",
)

region_plot_df["metric"] = region_plot_df["metric"].replace(
    {
        "avg_obesity_pct": "Average obesity (%)",
        "avg_sugar_kg_per_capita": "Average sugar availability (kg/capita/year)",
    }
)

fig_region = px.bar(
    region_plot_df,
    x="who_region",
    y="value",
    color="metric",
    barmode="group",
    title=f"Regional comparison, {latest_year}",
    labels={
        "who_region": "WHO region",
        "value": "Value",
        "metric": "",
    },
)

fig_region.update_layout(
    height=460,
    margin=dict(l=0, r=0, t=50, b=0),
)

st.plotly_chart(fig_region, use_container_width=True)

with st.expander("View latest regional summary table"):
    display_region = latest_region[
        [
            "who_region",
            "countries",
            "avg_obesity_pct",
            "avg_sugar_kg_per_capita",
            "avg_sugar_kcal_per_capita_day",
            "sugar_obesity_corr",
        ]
    ].copy()

    display_region = display_region.round(2)

    st.dataframe(display_region, use_container_width=True)


st.markdown(
    """
    <div class="note">
        <strong>Interpretation note:</strong> This dashboard shows associations, not causation.
        FAOSTAT sugar values represent national food supply availability, not exact individual
        consumption. WHO obesity values are country-level adult prevalence estimates.
    </div>
    """,
    unsafe_allow_html=True,
)