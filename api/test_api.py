import requests

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Input transaction
data = {
    "Product": "Laptop",
    "Quantity": 3,
    "UnitPrice": 50000,
    "Discount": 5,
    "AdvertisingSpend": 2500,
}

# Send POST request
response = requests.post(url, json=data)

# Display response
print("Status Code:", response.status_code)

print("Prediction Response:")

print(response.json())
