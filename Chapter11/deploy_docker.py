import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis].astype(np.float32) / 255.0
X_test = X_test[..., np.newaxis].astype(np.float32) / 255.0
X_train, X_valid = X_train_full[5000:], X_train_full[:5000]
y_train, y_valid = y_train_full[5000:], y_train_full[:5000]
X_new = X_test[:3]

model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28, 1]),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")])

model.summary()
model.compile(loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.SGD(lr=1e-1),
        metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

model_version = "0001"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
tf.saved_model.save(model, model_path)

print("\n\n##################################")
print("tensorflow serving")
