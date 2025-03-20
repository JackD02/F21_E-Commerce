import pandas as pd
import numpy as np
from datetime import datetime, timedelta

num_columns = int(input("Enter amount of columns: "))
columns = []

for i in range(num_columns):
    column_name = input(f"Enter name of column {i+1}:")
    columns.append(column_name)

boundaries = {}
col_types = {}

for column in columns:
    col_type = input(f"Enter type of column '{column}', int, float, str or date: ")
    col_types[column] = col_type

    if col_type in ['int', 'float']:
        min_val = input(f"Enter min value for column '{column}': ")
        max_val = input(f"Enter max value for column '{column}': ")
        boundaries[column] = (min_val, max_val)
    elif col_type == 'str':
        values = input(f"Enter strings for column '{column}' (comma seperated): ").split(',')
        boundaries[column] = values
    elif col_type == 'date':
        date = input(f"Enter date for column '{column}' (YYYY-MM-DD): ")
        date = datetime.strptime(date, "%Y-%m-%d")
        boundaries[column] = date

numentries = 10000
data = {}
prev_col_val = {}

for i, column in enumerate(columns):
    col_type = col_types[column]

    if col_type == 'int':
        min_val, max_val = boundaries[column]
        data[column] = np.random.randint(int(min_val), int(max_val) +1, size = numentries)
    elif col_type == 'float':
        min_val, max_val = boundaries[column]
        data[column] = np.random.uniform(float(min_val), float(max_val), size = numentries)
    elif col_type == 'str':
        values = boundaries[column]
        data[column] = [np.random.choice(values) for _ in range(numentries)]
    elif col_type == 'date':
        date = boundaries[column]

        d_days = 365
        date_column_val = []
        for index in range(numentries):
            if i > 0 and data[columns[i-1]][index] == 0:
                date_column_val.append('N/A')
            else :
                rand_date = date + timedelta(days = np.random.randint(0, d_days))
                date_column_val.append(rand_date)
            data[column] = date_column_val

df = pd.DataFrame(data)

filename = input("Enter name for this file, e.g example.csv: ")
df.to_csv(filename, index = False)
print(df.head())
