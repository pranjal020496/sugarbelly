DROP VIEW IF EXISTS v_obesity_sex_gap_latest;
DROP VIEW IF EXISTS v_obesity_region_summary_latest;
DROP VIEW IF EXISTS v_obesity_country_change;
DROP VIEW IF EXISTS v_obesity_latest_country;
DROP VIEW IF EXISTS v_obesity_country_year;


-- Country-year obesity data for both sexes only.
-- This will be the main table used for dashboarding and ML.
CREATE VIEW v_obesity_country_year AS
SELECT
    iso3,
    year,
    obesity_pct,
    obesity_pct_low,
    obesity_pct_high,
    who_region,
    who_region_code
FROM who_obesity
WHERE sex = 'Both sexes';


-- Latest available obesity values by country/area.
CREATE VIEW v_obesity_latest_country AS
SELECT
    iso3,
    year,
    obesity_pct,
    obesity_pct_low,
    obesity_pct_high,
    who_region,
    who_region_code,
    RANK() OVER (ORDER BY obesity_pct DESC) AS obesity_rank
FROM v_obesity_country_year
WHERE year = (
    SELECT MAX(year)
    FROM v_obesity_country_year
);


-- Change in obesity from first available year to latest available year.
CREATE VIEW v_obesity_country_change AS
WITH year_bounds AS (
    SELECT
        MIN(year) AS start_year,
        MAX(year) AS end_year
    FROM v_obesity_country_year
),

start_data AS (
    SELECT
        o.iso3,
        o.year AS start_year,
        o.obesity_pct AS obesity_start_pct
    FROM v_obesity_country_year o
    CROSS JOIN year_bounds y
    WHERE o.year = y.start_year
),

end_data AS (
    SELECT
        o.iso3,
        o.year AS end_year,
        o.obesity_pct AS obesity_latest_pct,
        o.who_region,
        o.who_region_code
    FROM v_obesity_country_year o
    CROSS JOIN year_bounds y
    WHERE o.year = y.end_year
),

change_data AS (
    SELECT
        e.iso3,
        s.start_year,
        e.end_year,
        s.obesity_start_pct,
        e.obesity_latest_pct,
        e.obesity_latest_pct - s.obesity_start_pct AS obesity_change_pct_points,
        e.who_region,
        e.who_region_code
    FROM end_data e
    JOIN start_data s
        ON e.iso3 = s.iso3
)

SELECT
    *,
    RANK() OVER (ORDER BY obesity_change_pct_points DESC) AS increase_rank
FROM change_data;


-- Latest regional summary.
CREATE VIEW v_obesity_region_summary_latest AS
SELECT
    who_region,
    year,
    COUNT(DISTINCT iso3) AS countries_or_areas,
    AVG(obesity_pct) AS avg_obesity_pct,
    MIN(obesity_pct) AS min_obesity_pct,
    MAX(obesity_pct) AS max_obesity_pct
FROM v_obesity_country_year
WHERE year = (
    SELECT MAX(year)
    FROM v_obesity_country_year
)
GROUP BY
    who_region,
    year;


-- Latest female vs male obesity gap.
CREATE VIEW v_obesity_sex_gap_latest AS
WITH latest_year AS (
    SELECT MAX(year) AS year
    FROM who_obesity
)

SELECT
    f.iso3,
    f.year,
    f.obesity_pct AS female_obesity_pct,
    m.obesity_pct AS male_obesity_pct,
    f.obesity_pct - m.obesity_pct AS female_minus_male_gap,
    f.who_region,
    f.who_region_code
FROM who_obesity f
JOIN who_obesity m
    ON f.iso3 = m.iso3
   AND f.year = m.year
CROSS JOIN latest_year ly
WHERE f.sex = 'Female'
  AND m.sex = 'Male'
  AND f.year = ly.year;