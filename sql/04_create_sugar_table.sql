DROP TABLE IF EXISTS faostat_sugar_supply;

CREATE TABLE faostat_sugar_supply (
    iso3 CHAR(3) NOT NULL,
    country TEXT NOT NULL,
    year INT NOT NULL,
    sugar_supply_kg_per_capita NUMERIC,
    sugar_supply_kcal_per_capita_day NUMERIC,
    PRIMARY KEY (iso3, year)
);