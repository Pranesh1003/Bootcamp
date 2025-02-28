import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Function to build the model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(10, activation='softmax')
    ])
    return model

# Train and evaluate function
def train_and_evaluate(optimizer, optimizer_name):
    model = create_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)
    
    return history

# Train with RMSprop
history_rmsprop = train_and_evaluate(RMSprop(), "RMSprop")

# Train with Adam
history_adam = train_and_evaluate(Adam(), "Adam")

# Plot Loss Curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_rmsprop.history['loss'], label='Train Loss - RMSprop')
plt.plot(history_rmsprop.history['val_loss'], label='Val Loss - RMSprop')
plt.plot(history_adam.history['loss'], label='Train Loss - Adam')
plt.plot(history_adam.history['val_loss'], label='Val Loss - Adam')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves: RMSprop vs. Adam')

# Plot Accuracy Curves
plt.subplot(1, 2, 2)
plt.plot(history_rmsprop.history['accuracy'], label='Train Acc - RMSprop')
plt.plot(history_rmsprop.history['val_accuracy'], label='Val Acc - RMSprop')
plt.plot(history_adam.history['accuracy'], label='Train Acc - Adam')
plt.plot(history_adam.history['val_accuracy'], label='Val Acc - Adam')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves: RMSprop vs. Adam')

plt.show()
