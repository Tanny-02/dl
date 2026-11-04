"""
Problem 3.e (original code from the shared Notion page, unmodified)

3E. Use Autoencoder to implement Network Intrusion Detection on NSL-KDD dataset.
Build the model
• Dataset: NSL-KDD is a network intrusion detection dataset with various types of
network traffic, labeled as either "normal" or "attack."
• Goal: Identify network intrusions by training an autoencoder on normal network traffic
data, then flagging traffic as anomalous if the reconstruction error is high.
Steps:
• Preprocess the NSL-KDD dataset, particularly by encoding categorical features and
scaling numerical features.

- Define an autoencoder similar to the ECG example, focusing on the reconstruction of
normal network patterns.
• Train on normal data and set an error threshold to detect intrusions.
"""

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras import layers

# 2. Define column names
columns = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# 3. Load dataset
train = pd.read_csv('KDDTrain+.txt', names=columns)
test = pd.read_csv('KDDTest+.txt', names=columns)

# 4. Remove difficulty column
train = train.drop('difficulty', axis=1)
test = test.drop('difficulty', axis=1)

# 5. Separate features and labels
X_train = train.drop('label', axis=1)
X_test = test.drop('label', axis=1)

# Convert labels into binary values
# 0 = Normal, 1 = Attack
y_test = (test['label'].str.strip() != 'normal').astype(int)

# 6. Convert categorical columns into numerical columns
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Make both datasets have the same columns
X_train, X_test = X_train.align(
    X_test, join='left', axis=1, fill_value=0
)

# 7. Normalize the data
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Select only normal traffic for training
normal_data = X_train[
    train['label'].str.strip().values == 'normal'
]

print("Training data shape:", normal_data.shape)
print("Testing data shape:", X_test.shape)

# 9. Build Autoencoder model
input_dim = normal_data.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

# 10. Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 11. Train the Autoencoder
history = model.fit(
    normal_data,
    normal_data,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# 12. Calculate reconstruction error for training data
train_predictions = model.predict(normal_data)

train_error = np.mean(
    np.square(normal_data - train_predictions),
    axis=1
)

# 13. Set threshold using 95th percentile
threshold = np.percentile(train_error, 95)

print("Threshold:", threshold)

# 14. Calculate reconstruction error for test data
test_predictions = model.predict(X_test)

test_error = np.mean(
    np.square(X_test - test_predictions),
    axis=1
)

# 15. Detect anomalies
y_pred = (test_error > threshold).astype(int)

# 16. Evaluate model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 17. Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.show()

# 18. Plot reconstruction error
plt.hist(train_error, bins=50, alpha=0.6, label='Normal Training Data')
plt.hist(test_error, bins=50, alpha=0.6, label='Test Data')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.show()
