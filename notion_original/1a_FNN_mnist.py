"""
Problem 1.a (original code from the shared Notion page, unmodified)

1A. Implement Feedforward Neural Network (FNN) with Keras and TensorFlow for
classification of hand-written MNIST dataset using below steps:
a. Import the necessary packages
b. Load the training and testing data
c. Define the network architecture using Keras
d. Train the model using SGD with 11 epochs
e. Evaluate the network
f. Plot the training loss and accuracy
"""

# 1. Import necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 2. Load training and testing data

train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')


# 3. Separate features and labels

X_train = train.drop('label', axis=1).values
y_train = train['label'].values

X_test = test.drop('label', axis=1).values
y_test = test['label'].values


# 4. Normalize pixel values

X_train = X_train / 255.0
X_test = X_test / 255.0


# 5. Define FNN architecture

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# 6. Compile the model using SGD

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 7. Train the model for 11 epochs

history = model.fit(
    X_train, y_train,
    epochs=11,
    batch_size=32,
    validation_split=0.2
)


# 8. Evaluate the model

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# 9. Plot training loss and accuracy

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()








"""
Notes (from the Notion page):

What is keras.Sequential()?

It is used to create a neural network by adding layers one after another in sequence.

Our network has 4 layers:

Line 1:
layers.Input(shape=(784,))



- This is the input layer.
- Each MNIST image is 28 × 28 pixels.
- 28 × 28 = 784 pixels.
- We give these 784 pixel values as input to the network.
- The comma in (784,) indicates a one-dimensional shape tuple.
Line 2:
layers.Dense(128, activation='relu')



- Creates the first hidden layer.
- It contains 128 neurons.
- Dense means every neuron is connected to all neurons in the previous layer.
- ReLU is an activation function.
- It outputs 0 for negative values and keeps positive values unchanged.
Line 3:
layers.Dense(64, activation='relu')



- Creates the second hidden layer.
- It contains 64 neurons.
- It also uses ReLU activation.
- It learns useful patterns from the input.
Line 4:
layers.Dense(10, activation='softmax')



- This is the output layer.
- It contains 10 neurons because we have 10 digits (0–9).
- Softmax converts the outputs into probabilities.
- The neuron with the highest probability represents the predicted digit.

"""