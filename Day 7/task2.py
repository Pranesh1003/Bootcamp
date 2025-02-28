import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the digit dataset (MNIST)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data (scale pixel values to 0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images
    Dense(128, activation='relu'),  # Hidden layer
    Dense(10, activation='softmax') # Output layer (10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with batch_size=1000
history_1000 = model.fit(X_train, y_train, 
                          validation_data=(X_test, y_test),
                          epochs=10, batch_size=1000)

# Train the model with batch_size=2000
history_2000 = model.fit(X_train, y_train, 
                          validation_data=(X_test, y_test),
                          epochs=10, batch_size=2000)

# Function to plot accuracy & loss
def plot_history(history, batch_size):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy (Batch Size {batch_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss (Batch Size {batch_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot results for batch sizes 1000 and 2000
plot_history(history_1000, 1000)
plot_history(history_2000, 2000)
