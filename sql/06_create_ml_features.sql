DROP VIEW IF EXISTS v_ml_obesity_features;


CREATE VIEW v_ml_obesity_features AS
WITH base AS (
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

        LAG(obesity_pct, 2) OVER (
            PARTITION BY iso3
            ORDER BY year
        ) AS obesity_lag_2,

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

        LEAD(obesity_pct, 1) OVER (
            PARTITION BY iso3
            ORDER BY year
        ) AS target_obesity_next_year

    FROM v_sugar_obesity_country_year
)

SELECT
    iso3,
    country,
    year,
    who_region,
    who_region_code,

    obesity_pct,
    obesity_lag_1,
    obesity_lag_2,
    obesity_lag_3,

    sugar_supply_kg_per_capita,
    sugar_supply_kcal_per_capita_day,
    sugar_lag_1,
    sugar_lag_3,

    obesity_pct - obesity_lag_1 AS obesity_change_1yr,
    obesity_pct - obesity_lag_3 AS obesity_change_3yr,

    sugar_supply_kg_per_capita - sugar_lag_1 AS sugar_change_1yr,
    sugar_supply_kg_per_capita - sugar_lag_3 AS sugar_change_3yr,

    target_obesity_next_year

FROM base;