import pandas as pd
import numpy as np
import random

numcust = 100

cust_ids =[f"C{101 + i}" for i in range(numcust)]

purchase_amount_range = (0.0, 300.0)
new_item_range = (0.0, 0.0)

productA = [round(random.uniform(*purchase_amount_range), 2) for _ in range(numcust)]
productB = [round(random.uniform(*purchase_amount_range), 2) for _ in range(numcust)]
productC = [round(random.uniform(*purchase_amount_range), 2) for _ in range(numcust)]
productD = [round(random.uniform(*purchase_amount_range), 2) for _ in range(numcust)]
ProductE = [round(random.uniform(*purchase_amount_range), 2) for _ in range(numcust)]
new_item = [np.nan for _ in range(numcust)]

df = pd.DataFrame ({
    "Customer ID": cust_ids, 
    "Product A": productA, 
    "Product B": productB, 
    "Product C": productC, 
    "Product D": productD, 
    "Product E": ProductE,
    "New Product": new_item
})

df.to_csv('purchase_data.csv', index = False)
print(df.head())