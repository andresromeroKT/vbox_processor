import requests
import json

# Tu endpoint actual
url = "https://lka5yy6ij6.execute-api.us-east-1.amazonaws.com/default/ProcessVboxanalyser"

# Test simple
response = requests.post(
    url,
    json={"reference_code": "773761VBOX"},
    headers={"Content-Type": "application/json"},
    timeout=900
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Formatear JSON si es posible
try:
    result = response.json()
    print("\nJSON Formateado:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except:
    pass