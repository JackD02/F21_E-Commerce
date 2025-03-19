import pandas as pd
import numpy as np

file = input("Enter the path of the CSV file: ").strip()
data = pd.read_csv(file)

column_ranges = {
    "customer_id": (15000000, 15999999),
    "credit_score": (300, 999),
    "age":(18, 80),
    "tenure":(0,10),
    "balance":(0, 200000.0),
    "products_number":(1,5),
    "credit_card":(0,1),
    "active_member":(0,1),
    "estimated_salary":(1500.0, 200000.0),
    "churn":(0,1)
}

change = int(data.size * 0.1)
rows = np.random.randint(0, data.shape[0], change)
columns = np.random.randint(0, data.shape[1], change)

for row, col in zip(rows, columns):
    col_name = data.columns[col]
    if col_name in column_ranges:
        min_val, max_val = column_ranges[col_name]
        if np.issubdtype(data[col_name].dtype, np.integer):
            data.iat[row, col] = np.random.randint(min_val, max_val)
        else:
            data.iat[row, col] = np.random.uniform(min_val, max_val) 

output_path = file.replace(".csv", "_new.csv")
data.to_csv(output_path, index = False)