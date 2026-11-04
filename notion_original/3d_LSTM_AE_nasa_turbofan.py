"""
Problem 3.d (original code from the shared Notion page, unmodified)

3D. Detect the faults in Sensor Data from Machinery (NASA Turbofan Engine Dataset)
using Autoencoder and build the model by using the following steps:
• Dataset: The dataset contains sensor readings from turbofan engines with labels
indicating whether a reading is faulty.
• Goal: Detect machinery faults by training an autoencoder on sensor data from healthy
engines.
Steps:
• Preprocess data by scaling sensor readings.
• Use an LSTM-based autoencoder (to capture sequential data) to learn patterns in
normal sensor readings.
• Calculate reconstruction error for test data, with a threshold to classify high-error
instances as faults.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

# a. Load and preprocess data
data = pd.read_csv('nasa_turbofan.csv')

# Remove missing values
data = data.dropna(subset=['Fault_Label'])

# Sort data by engine and cycle
data = data.sort_values(['unit', 'cycle'])

# Select sensor features
features = [
    'setting1', 'setting2', 'setting3',
    'sensor1', 'sensor2', 'sensor3', 'sensor4',
    'sensor5', 'sensor6', 'sensor7', 'sensor8',
    'sensor9', 'sensor10', 'sensor11', 'sensor12',
    'sensor13', 'sensor14', 'sensor15', 'sensor16',
    'sensor17', 'sensor18', 'sensor19', 'sensor20',
    'sensor21'
]

# Separate input and output
X = data[features].copy()
y = data['Fault_Label'].astype('int32').values

# Fill missing sensor values
X = X.groupby(data['unit']).transform(
    lambda group: group.ffill().bfill()
)

# Split engines into training and testing
units = data['unit'].unique()

train_units, test_units = train_test_split(
    units, test_size=0.2, random_state=42
)

# Select training and testing data
train_data = data[data['unit'].isin(train_units)]
test_data = data[data['unit'].isin(test_units)]

X_train = X.loc[train_data.index].values
X_test = X.loc[test_data.index].values

y_train = train_data['Fault_Label'].astype('int32').values
y_test = test_data['Fault_Label'].astype('int32').values

# Normalize sensor readings
scaler = MinMaxScaler()

normal_train = X_train[y_train == 0]
scaler.fit(normal_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create time windows for each engine
window_size = 10

def create_sequences(X, y, units, window_size):
    sequences = []
    labels = []

    for unit in np.unique(units):
        indices = np.where(units == unit)[0]

        unit_X = X[indices]
        unit_y = y[indices]

        for i in range(len(unit_X) - window_size + 1):
            sequences.append(unit_X[i:i + window_size])
            labels.append(max(unit_y[i:i + window_size]))

    return np.array(sequences), np.array(labels)

X_train_seq, y_train_seq = create_sequences(
    X_train, y_train, train_data['unit'].values, window_size
)

X_test_seq, y_test_seq = create_sequences(
    X_test, y_test, test_data['unit'].values, window_size
)

# Use only normal sequences for training
X_train_normal = X_train_seq[y_train_seq == 0]

# b. Define LSTM Autoencoder architecture
model = keras.Sequential([
    layers.Input(shape=(window_size, len(features))),

    layers.LSTM(32, return_sequences=False),
    layers.RepeatVector(window_size),

    layers.LSTM(32, return_sequences=True),

    layers.TimeDistributed(
        layers.Dense(len(features))
    )
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

# Detect faults
y_pred = (test_error > threshold).astype(int)

print("Detected Faults:", np.sum(y_pred))

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
plt.title('NASA Turbofan Fault Detection')
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
