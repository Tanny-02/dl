"""
Problem 1.c (original code from the shared Notion page, unmodified)

1C. Build Feedforward Neural Network (FNN) with Keras and TensorFlow for
classification ofCIFAR10 image dataset using the following steps:
a. Import the necessary packages
b. Load the training and testing data
c. Define the network architecture using Keras
d. Train the model using SGD/Adam optimizer
e. Evaluate the network
f. Plot the training loss and accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# a. Import packages (done above)

# b. Load data
data = pd.read_csv('cifar-10.csv')

# Remove rows with missing labels
data = data.dropna(subset=['label'])

# Separate input and output
X = data.drop('label', axis=1).values
y = data['label'].values

# Convert labels to integers
y = y.astype('int32')

# Normalize pixel values
X = X / 255.0

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# c. Define FNN architecture
model = keras.Sequential([
    layers.Input(shape=(3072,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# d. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=11,
    batch_size=32,
    validation_split=0.2
)

# e. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# f. Plot training loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
