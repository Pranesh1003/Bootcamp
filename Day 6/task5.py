import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a simple synthetic housing dataset
np.random.seed(42)
num_samples = 500
X = np.random.rand(num_samples, 5) * 100  # 5 features (e.g., size, rooms, location score, etc.)
y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(num_samples) * 10  # Target price with some noise

# Convert to DataFrame and split into train/test
df = pd.DataFrame(X, columns=['Size', 'Rooms', 'LocationScore', 'Age', 'Distance'])
df['Price'] = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to create models
def create_model(hidden_units=0):
    model = Sequential()
    model.add(Dense(5, input_shape=(5,), activation='linear'))  # Input layer

    if hidden_units > 0:
        model.add(Dense(hidden_units, activation='relu'))  # Hidden layer

    model.add(Dense(1))  # Output layer
    model.compile(optimizer=RMSprop(learning_rate=0.5), loss='mse')
    return model

# Train models and store history
history_baseline = create_model(0).fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=0)
history_2units = create_model(2).fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=0)
history_4units = create_model(4).fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=0)

# Plot loss curves
plt.figure(figsize=(12, 5))
plt.plot(history_baseline.history['loss'], label='Baseline (No Hidden Layer)')
plt.plot(history_2units.history['loss'], label='Hidden Layer (2 Units)')
plt.plot(history_4units.history['loss'], label='Hidden Layer (4 Units)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Hidden Layer Tuning - Loss Curves')
plt.show()
