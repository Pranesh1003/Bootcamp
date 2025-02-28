import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target  # Features & target price

# Normalize the features (for better model performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training (67%) and test (33%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# Define the model (Linear Regression with no hidden layers)
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(X_train.shape[1],))  # 1 output neuron (predicting price)
])

# Compile the model
model.compile(optimizer=keras.optimizers.RMSprop(), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Plot Training vs Validation Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Housing Price Prediction - Loss Curve')
plt.legend()
plt.show()
