"""
Problem 3.a - Credit card fraud detection using an Autoencoder (Kaggle creditcard.csv)

Dataset: transactions labelled fraud / non-fraud, with severe class imbalance (~0.17% fraud).
Goal: train an autoencoder ONLY on non-fraud transactions, then flag transactions whose
reconstruction error is above a threshold as fraud.

Fixes vs the Notion version:
  - Holds out 20% of normal rows as a real test set (Notion measured errors on the training data).
  - Adds evaluation metrics: confusion matrix, precision/recall/F1, ROC-AUC.
  - Drops "Time" (seconds since the first transaction, carries no fraud pattern).
  - Log-scale histogram (284k normal vs 492 fraud is unreadable on a linear scale).
"""

# 1. Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers

# 2. Load data
data = pd.read_csv("creditcard.csv")
print(data["Class"].value_counts())        # 0 = normal, 1 = fraud (heavily imbalanced)

X = data.drop(["Time", "Class"], axis=1).values   # V1..V28 (PCA) + Amount = 29 features
y = data["Class"].values

# 3. Split: train ONLY on normal, test on unseen normal + all fraud
X_normal = X[y == 0]
X_fraud = X[y == 1]
X_train, X_test_normal = train_test_split(X_normal, test_size=0.2, random_state=42)

X_test = np.vstack([X_test_normal, X_fraud])
y_test = np.hstack([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])

# 4. Scale features (fit on training data only, to avoid leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build the autoencoder
input_dim = X_train.shape[1]
model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16, activation="relu"),          # encoder
    layers.Dense(8, activation="relu"),           # bottleneck
    layers.Dense(16, activation="relu"),          # decoder
    layers.Dense(input_dim, activation="linear")  # reconstruct the 29 features (can be negative)
])
model.summary()

# 6. Compile: optimizer, loss, metric
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 7. Train (input = target: the network learns to copy normal transactions)
history = model.fit(X_train, X_train, epochs=10, batch_size=256,
                    validation_split=0.1)

# 8. Reconstruction error = mean squared error per row
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_error = np.mean(np.square(X_train - train_pred), axis=1)
test_error = np.mean(np.square(X_test - test_pred), axis=1)

# 9. Threshold = 95th percentile of the error on normal training data
threshold = np.percentile(train_error, 95)
print("Threshold:", threshold)

# 10. Detect fraud and evaluate
y_pred = (test_error > threshold).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
print("ROC-AUC:", roc_auc_score(y_test, test_error))

# 11. Plot training loss
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Autoencoder loss"); plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.legend(); plt.show()

# 12. Plot reconstruction error
plt.hist(test_error[y_test == 0], bins=100, alpha=0.6, label="Normal", log=True)
plt.hist(test_error[y_test == 1], bins=100, alpha=0.6, label="Fraud", log=True)
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.title("Credit card fraud detection")
plt.xlabel("Reconstruction error"); plt.ylabel("Count (log)")
plt.legend(); plt.show()
