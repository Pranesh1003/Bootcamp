import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta, Adam

# Load the digit dataset (MNIST)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data (scale pixel values to 0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten 28x28 images
        Dense(128, activation='relu'),  # Hidden layer
        Dense(10, activation='softmax') # Output layer (10 classes)
    ])
    return model

# Train with AdaDelta optimizer (learning rate = 0.01)
model_adadelta = create_model()
model_adadelta.compile(optimizer=Adadelta(learning_rate=0.01),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

history_adadelta = model_adadelta.fit(X_train, y_train, 
                                      validation_data=(X_test, y_test),
                                      epochs=20, batch_size=128)

# Train with Adam optimizer (learning rate = 0.05)
model_adam = create_model()
model_adam.compile(optimizer=Adam(learning_rate=0.05),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

history_adam = model_adam.fit(X_train, y_train, 
                              validation_data=(X_test, y_test),
                              epochs=20, batch_size=128)

# Function to plot accuracy & loss
def plot_history(history, optimizer_name):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy with {optimizer_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss with {optimizer_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot results for both optimizers
plot_history(history_adadelta, "AdaDelta (lr=0.01)")
plot_history(history_adam, "Adam (lr=0.05)")
