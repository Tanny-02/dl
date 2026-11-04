# Problem 2 - CNN image classification on CIFAR-10
# 4 stages: a) load + preprocess  b) architecture  c) train  d) estimate performance

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %% a. Loading and preprocessing the image data
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

# IMPORTANT: the CSV stores each image CHANNEL-FIRST:
#   pixel_0..1023 = red plane, 1024..2047 = green, 2048..3071 = blue.
# So reshape to (N, 3, 32, 32) and then move channels last -> (N, 32, 32, 3).
# A plain reshape(-1, 32, 32, 3) scrambles the colours/pixels.
X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
print("Train:", X_train.shape, " Test:", X_test.shape)

# %% b. Defining the model's architecture
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),                      # 32x32 -> 16x16
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),                      # 16x16 -> 8x8
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])
model.summary()

# %% c. Training the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train,
                    epochs=10, batch_size=64,
                    validation_split=0.1)

# %% d. Estimating the model's performance
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}   Test accuracy: {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)   # probabilities -> class index
print(classification_report(y_test, y_pred, zero_division=0))

# Show 9 test images with predicted vs actual label
plt.figure(figsize=(7, 7))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i])
    color = "green" if y_pred[i] == y_test[i] else "red"
    plt.title(f"P: {class_names[y_pred[i]]}  A: {class_names[y_test[i]]}", color=color, fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.show()

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
