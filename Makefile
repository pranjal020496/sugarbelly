setup:
	pip install -r requirements.txt

create-db:
	createdb sugarbelly

create-views:
	psql -d sugarbelly -f sql/01_create_tables.sql
	psql -d sugarbelly -f sql/04_create_sugar_table.sql
	psql -d sugarbelly -f sql/03_create_obesity_views.sql
	psql -d sugarbelly -f sql/05_create_sugar_obesity_views.sql
	psql -d sugarbelly -f sql/06_create_ml_features.sql
	psql -d sugarbelly -f sql/07_create_sugar_sensitivity_features.sql

train:
	python src/models/train_sugar_sensitivity_model.py

forecast:
	python src/models/forecast_sugar_sensitivity_to_2030.py

dashboard:
	streamlit run app/dashboard.py

check:
	python -m py_compile app/dashboard.py
	python -m py_compile src/models/train_sugar_sensitivity_model.py
	python -m py_compile src/models/forecast_sugar_sensitivity_to_2030.py