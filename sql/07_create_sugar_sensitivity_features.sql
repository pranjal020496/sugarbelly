DROP VIEW IF EXISTS v_ml_sugar_sensitivity_features;

CREATE VIEW v_ml_sugar_sensitivity_features AS
WITH base AS (
    SELECT
        iso3,
        country,
        year::INT AS year,
        obesity_pct::NUMERIC AS obesity_pct,
        sugar_supply_kg_per_capita::NUMERIC AS sugar_supply_kg_per_capita,
        sugar_supply_kcal_per_capita_day::NUMERIC AS sugar_supply_kcal_per_capita_day,
        who_region,
        who_region_code
    FROM v_sugar_obesity_country_year
    WHERE obesity_pct IS NOT NULL
      AND sugar_supply_kg_per_capita IS NOT NULL
      AND sugar_supply_kcal_per_capita_day IS NOT NULL
),
lagged AS (
    SELECT
        iso3,
        country,
        year,
        obesity_pct,
        sugar_supply_kg_per_capita,
        sugar_supply_kcal_per_capita_day,
        who_region,
        who_region_code,

        LAG(obesity_pct, 1) OVER (
            PARTITION BY iso3
            ORDER BY year
        ) AS obesity_lag_1,

        LAG(obesity_pct, 3) OVER (
            PARTITION BY iso3
            ORDER BY year
        ) AS obesity_lag_3,

        LAG(sugar_supply_kg_per_capita, 1) OVER (
            PARTITION BY iso3
            ORDER BY year
        ) AS sugar_lag_1,

        LAG(sugar_supply_kg_per_capita, 3) OVER (
            PARTITION BY iso3
            ORDER BY year
        ) AS sugar_lag_3,

        LEAD(obesity_pct, 3) OVER (
            PARTITION BY iso3
            ORDER BY year
        ) AS obesity_future_3yr
    FROM base
)
SELECT
    iso3,
    country,
    year,
    obesity_pct,
    obesity_lag_1,
    obesity_lag_3,
    obesity_pct - obesity_lag_1 AS obesity_change_1yr,
    obesity_pct - obesity_lag_3 AS obesity_change_3yr,

    sugar_supply_kg_per_capita,
    sugar_supply_kcal_per_capita_day,
    sugar_lag_1,
    sugar_lag_3,
    sugar_supply_kg_per_capita - sugar_lag_1 AS sugar_change_1yr,
    sugar_supply_kg_per_capita - sugar_lag_3 AS sugar_change_3yr,

    who_region,
    who_region_code,

    obesity_future_3yr,
    obesity_future_3yr - obesity_pct AS target_obesity_change_3yr
FROM lagged;