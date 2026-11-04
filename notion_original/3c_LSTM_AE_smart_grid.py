"""
Problem 3.c (original code from the shared Notion page, unmodified)

3C. Implement Anomaly Detection in Energy Consumption Data (Smart Grid
Dataset) using Autoencoder and build the model by using the following steps:
Steps:
• Preprocess the data to fill any missing values, normalize, and create time windows for
sequential input.
• Use an LSTM-based autoencoder to learn temporal patterns in normal consumption.
• Set an error threshold and detect anomalies when the reconstruction error exceeds this
threshold.
Dataset: The Smart Grid dataset records electricity consumption across multiple timestamps
for various households.
Goal: Detect unusual consumption patterns by training an autoencoder on typical (normal)
energy usage patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

# a. Load and preprocess data
data = pd.read_csv('energy_consumption.csv')

# Remove rows with missing labels
data = data.dropna(subset=['Anomaly_Label'])

# Sort data according to timestamp
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

if data['Timestamp'].notna().any():
    data = data.sort_values('Timestamp')

# Select input features and output label
features = [
    'Electricity_Consumed',
    'Temperature',
    'Humidity',
    'Wind_Speed',
    'Avg_Past_Consumption'
]

X = data[features].copy()
y = data['Anomaly_Label'].astype('int32').values

# Fill missing feature values
X = X.interpolate().ffill().bfill()

# Split data into training and testing
split = int(len(X) * 0.8)

X_train = X.iloc[:split].values
X_test = X.iloc[split:].values

y_train = y[:split]
y_test = y[split:]

# Normalize data
scaler = MinMaxScaler()

normal_train = X_train[y_train == 0]
scaler.fit(normal_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create time windows
window_size = 10

def create_sequences(X, y, window_size):
    sequences = []
    labels = []

    for i in range(len(X) - window_size + 1):
        sequences.append(X[i:i + window_size])
        labels.append(max(y[i:i + window_size]))

    return np.array(sequences), np.array(labels)

X_train_seq, y_train_seq = create_sequences(
    X_train, y_train, window_size
)

X_test_seq, y_test_seq = create_sequences(
    X_test, y_test, window_size
)

# Use only normal sequences for training
X_train_normal = X_train_seq[y_train_seq == 0]

# b. Define LSTM Autoencoder architecture
model = keras.Sequential([
    layers.Input(shape=(window_size, len(features))),

    layers.LSTM(32, return_sequences=False),
    layers.RepeatVector(window_size),

    layers.LSTM(32, return_sequences=True),

    layers.TimeDistributed(layers.Dense(len(features)))
])

# c. Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train only on normal data
history = model.fit(
    X_train_normal, X_train_normal,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# d. Calculate reconstruction error
train_predictions = model.predict(X_train_normal)
test_predictions = model.predict(X_test_seq)

train_error = np.mean(
    np.square(X_train_normal - train_predictions),
    axis=(1, 2)
)

test_error = np.mean(
    np.square(X_test_seq - test_predictions),
    axis=(1, 2)
)

# Set threshold
threshold = np.percentile(train_error, 95)

print("Threshold:", threshold)

# Detect anomalies
y_pred = (test_error > threshold).astype(int)

print("Detected Anomalies:", np.sum(y_pred))

# Evaluate model
print("Classification Report:")
print(classification_report(
    y_test_seq, y_pred, labels=[0, 1], zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test_seq, y_pred, labels=[0, 1]))

# Plot reconstruction error
plt.figure(figsize=(10, 5))
plt.plot(test_error, label='Reconstruction Error')
plt.axhline(threshold, color='red', linestyle='dashed',
            label='Threshold')
plt.title('Energy Consumption Anomaly Detection')
plt.xlabel('Time Windows')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()

# Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
