# Problem 1 - Feedforward Neural Network (FNN) on CIFAR-10
# Steps: a) imports  b) load data  c) architecture  d) train with SGD/Adam  e) evaluate  f) plot

# %% a. Import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %% b. Load the training and testing data
PATH = "../LP IV Updated Datasets/FNN and CNN/CIFAR10/"
# ONE csv with 50,000 rows: pixel_0 ... pixel_3071 and "label" as the LAST column.
# dtype float32 halves memory (default float64 needs ~1.2 GB for this file).
data = pd.read_csv(PATH + "cifar-10.csv", dtype="float32")
data = data.dropna(subset=["label"])     # safety only - this file has no missing values

X = data.drop("label", axis=1).values / 255.0
y = data["label"].values.astype("int32")
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# No separate test file, so we make one (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, " Test:", X_test.shape)

# %% c. Define the network architecture using Keras
# 32x32 colour image = 32*32*3 = 3072 inputs. An FNN ignores the spatial layout,
# so expect only ~45-50% accuracy here - that is normal and is why CNNs exist.
model = keras.Sequential([
    layers.Input(shape=(3072,)),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.summary()

# %% d. Train the model using Adam (SGD also allowed - swap the optimizer)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train,
                    epochs=15, batch_size=64,
                    validation_split=0.1)

# %% Evaluate the network on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}   Test accuracy: {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)   # probabilities -> class index
print(classification_report(y_test, y_pred, zero_division=0))

# %% Plot the training loss and accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.tight_layout()
plt.show()
