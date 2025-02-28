import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

# Load the California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split into training and testing sets (67% train, 33% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model (Single input & output layer)
def build_model():
    model = Sequential([
        Dense(1, input_shape=(X_train.shape[1],))  # Single output for regression
    ])
    model.compile(optimizer=RMSprop(learning_rate=0.01), loss='mse')
    return model

# Train with different batch sizes
batch_sizes = [16, 32, 64]
histories = {}

for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    model = build_model()
    history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    histories[batch_size] = history.history

# Plot loss curves for different batch sizes
plt.figure(figsize=(12, 5))

for batch_size in batch_sizes:
    plt.plot(histories[batch_size]['loss'], label=f'Train Loss (Batch {batch_size})')
    plt.plot(histories[batch_size]['val_loss'], label=f'Validation Loss (Batch {batch_size})', linestyle='dashed')

plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Batch Size Effect on Training')
plt.legend()
plt.show()
