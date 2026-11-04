"""
Problem 3.a (original code from the shared Notion page, unmodified)

Implement credit card fraud detection for given credit card fraud detection Kaggle
dataset using Autoencoder and build the model by using the following steps:
• Dataset: The dataset contains transactions labeled as fraudulent or non-fraudulent, with
a severe class imbalance.

- Goal: Identify fraudulent transactions by training an autoencoder only on non-
fraudulent transactions.

Steps:
• Preprocess the data by scaling features (features are already PCA-transformed, so
they’re anonymized).
• Train an autoencoder on non-fraudulent transactions to learn typical spending patterns.
• Calculate reconstruction errors for transactions and set a threshold to detect fraud.
• Compile the models with Optimizer, Loss, and Evaluation Metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# a. Load and preprocess data
data = pd.read_csv('creditcard.csv')

# Separate normal and fraud transactions
normal = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

# Separate input features
X_normal = normal.drop('Class', axis=1).values
X_fraud = fraud.drop('Class', axis=1).values

# Scale the features
scaler = StandardScaler()
X_normal = scaler.fit_transform(X_normal)
X_fraud = scaler.transform(X_fraud)

# b. Define Autoencoder architecture
input_dim = X_normal.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

# c. Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train only on normal transactions
history = model.fit(
    X_normal, X_normal,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# d. Calculate reconstruction error
normal_predictions = model.predict(X_normal)
fraud_predictions = model.predict(X_fraud)

normal_error = np.mean(
    np.square(X_normal - normal_predictions), axis=1
)

fraud_error = np.mean(
    np.square(X_fraud - fraud_predictions), axis=1
)

# Set threshold
threshold = np.percentile(normal_error, 95)

print("Threshold:", threshold)

# Detect fraud
fraud_detected = fraud_error > threshold

print("Detected Fraud Transactions:", np.sum(fraud_detected))

# Plot reconstruction error
plt.hist(normal_error, bins=50, alpha=0.6, label='Normal')
plt.hist(fraud_error, bins=50, alpha=0.6, label='Fraud')
plt.axvline(threshold, color='red', linestyle='dashed',
            label='Threshold')

plt.title('Credit Card Fraud Detection')
plt.xlabel('Reconstruction Error')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()

# Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
