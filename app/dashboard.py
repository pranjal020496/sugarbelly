import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import create_engine


DATABASE_URL = "postgresql+psycopg2:///sugarbelly"


st.set_page_config(
    page_title="Sugar Belly",
    page_icon="🍬",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.6rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #fff7ed 0%, #ffe4e6 45%, #eef2ff 100%);
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 3.1rem;
        margin-bottom: 0.25rem;
        color: #111827;
    }

    .hero p {
        font-size: 1.05rem;
        color: #4b5563;
        max-width: 1000px;
        line-height: 1.6;
    }

    .metric-card {
        padding: 1.15rem;
        border-radius: 18px;
        background: white;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 8px 28px rgba(0,0,0,0.05);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 750;
        color: #111827;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        color: #111827;
        margin-top: 1.4rem;
        margin-bottom: 0.6rem;
    }

    .note {
        font-size: 0.92rem;
        color: #4b5563;
        background: #ffffff;
        border-left: 4px solid #f59e0b;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-top: 1rem;
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

    return (
        clean_numeric_columns(latest),
        clean_numeric_columns(country_year),
        clean_numeric_columns(region_summary),
        clean_numeric_columns(country_change),
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


latest_df, country_year_df, region_summary_df, country_change_df = load_data()

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>🍬 Sugar Belly</h1>
        <p>
            Global Sugar & Obesity Trend Intelligence. Explore how sugar and sweeteners
            availability relates to adult obesity prevalence across countries, WHO regions,
            and time. Built with Python, PostgreSQL, SQL views, Streamlit, and Plotly.
        </p>
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
        color_continuous_scale="Reds",
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