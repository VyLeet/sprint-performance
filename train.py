import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load

# Get Google Sheets data
SHEET_ID = '1faXfPd2w_jCAyy-XNj9wU4e9BX8zdEpFOYNEd5UEvsY'
SHEET_NAME = 'AAPL'
url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
df = pd.read_csv(url)

# Process data
## First row denotes mean values
df = df[1:]

## Rename columns to acquire column getters
column_names = []
for column_name in df.columns:
    new_column_name = column_name\
        .replace('\n ', '_')\
        .replace(' \n', '_')\
        .replace('\n', '_')\
        .replace(' ', '_')\
        .replace('-', '_')
    column_names.append(new_column_name)
df.columns = column_names

## Get columns of interest
predictors = ["Sprint_duration", "Team_FTE", "Max_days", "Intended_days", "Intended_days_off", 
           "Initially_resolved", "Planned_releases_count", "Had_Ship_It"]
X = df[predictors]
y = df["Performed"]

## Impute
X = X.fillna(X.median())
y = y.fillna(y.median())

# Train model
lr = LinearRegression()
lr.fit(X, y)

# Save model
dump(lr, 'lr.joblib') 