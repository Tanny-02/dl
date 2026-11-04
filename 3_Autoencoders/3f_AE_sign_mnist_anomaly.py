"""
Problem 3.f (code from the shared Notion page)

3F. Using the Sign Language MNIST dataset, select any one of the 24 alphabet classes
(label Y-24) as an anomaly. Train an Autoencoder–Decoder model using the remaining
23 classes and detect the selected class as an anomaly based on reconstruction error.
"""

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, classification_report

# 2. Load dataset
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

# 3. Separate features and labels
X_train = train.drop('label', axis=1).values
y_train = train['label'].values

X_test = test.drop('label', axis=1).values
y_test = test['label'].values

# 4. Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# 5. Select anomaly class
anomaly_label = 24

# Check whether selected class exists
if anomaly_label not in np.unique(y_train):
    raise ValueError("Label 24 is not present in the training dataset.")

# 6. Select only normal data for training
normal_data = X_train[y_train != anomaly_label]

print("Normal training data shape:", normal_data.shape)

# 7. Build Autoencoder model
model = keras.Sequential([
    layers.Input(shape=(784,)),

    # Encoder
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),

    # Decoder
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])

# 8. Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 9. Train Autoencoder using normal data only
history = model.fit(
    normal_data,
    normal_data,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# 10. Calculate reconstruction error on normal training data
train_pred = model.predict(normal_data)

train_error = np.mean(
    np.square(normal_data - train_pred),
    axis=1
)

# 11. Set anomaly threshold
threshold = np.percentile(train_error, 95)

print("Anomaly Threshold:", threshold)

# 12. Calculate reconstruction error on test data
test_pred = model.predict(X_test)

test_error = np.mean(
    np.square(X_test - test_pred),
    axis=1
)

# 13. Detect anomalies
# Error above threshold = anomaly
y_pred = (test_error > threshold).astype(int)

# Actual labels: 1 = selected anomaly class, 0 = other classes
y_actual = (y_test == anomaly_label).astype(int)

# 14. Evaluate model
print("\nConfusion Matrix:")
print(confusion_matrix(y_actual, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_actual,
    y_pred,
    target_names=['Normal', 'Anomaly'],
    zero_division=0
))

# 15. Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.show()

# 16. Plot reconstruction error
plt.hist(train_error, bins=50, alpha=0.6, label='Normal Training Data')
plt.hist(test_error, bins=50, alpha=0.6, label='Test Data')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')

plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Anomaly Detection using Reconstruction Error')
plt.legend()
plt.show()
