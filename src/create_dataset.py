import pandas as pd
import numpy as np

np.random.seed(42)

products = ["Laptop", "Mobile", "Tablet", "Monitor", "Keyboard"]

records = []

for i in range(120):
    product = np.random.choice(products)

    quantity = np.random.randint(1, 11)

    if product == "Laptop":
        unit_price = 50000
    elif product == "Mobile":
        unit_price = 15000
    elif product == "Tablet":
        unit_price = 20000
    elif product == "Monitor":
        unit_price = 12000
    else:
        unit_price = 2000

    discount = np.random.choice([0, 5, 10, 15])
    advertising_spend = np.random.randint(500, 5001)

    sales = quantity * unit_price * (1 - discount / 100) + advertising_spend * 0.5

    records.append(
        {
            "Product": product,
            "Quantity": quantity,
            "UnitPrice": unit_price,
            "Discount": discount,
            "AdvertisingSpend": advertising_spend,
            "Sales": round(sales, 2),
        }
    )

df = pd.DataFrame(records)

df.to_csv("data/raw/retail_sales.csv", index=False)

print("Dataset created successfully")
print(df.head())
print("Total records:", len(df))
