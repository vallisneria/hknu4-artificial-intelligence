import numpy as np
import json
import requests
from tensorflow import keras

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_new = X_test[:3]

input_json_data = json.dumps({"signature_name": "serving_default", "instances": X_new.tolist()})

SERVER_URL = "http://localhost:8501/v1/models/my_mnist_model:predict"
response = requests.post(SERVER_URL, data=input_json_data)
response.raise_for_status()
response = response.json()

y_proba = np.array(response["predictions"])
print("\n\n###############################")
print("Predict")
print(y_proba.round(2))

