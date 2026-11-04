# Problem 2 - CNN image classification on Fashion MNIST
# 4 stages: a) load + preprocess  b) architecture  c) train  d) estimate performance

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report

# %% a. Loading and preprocessing the image data
PATH = "../LP IV Updated Datasets/FNN and CNN/Fashion MNIST/"
train = pd.read_csv(PATH + "fashion-mnist_train.csv")
test = pd.read_csv(PATH + "fashion-mnist_test.csv")

X_train = train.drop("label", axis=1).values
y_train = train["label"].values
X_test = test.drop("label", axis=1).values
y_test = test["label"].values

# A CNN needs images, not flat rows: (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0     # 1 channel = grayscale
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print("Train:", X_train.shape, " Test:", X_test.shape)

# %% b. Defining the model's architecture
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),   # 32 filters learn edges/strokes -> 26x26x32
    layers.MaxPooling2D((2, 2)),                     # halve size -> 13x13x32
    layers.Conv2D(64, (3, 3), activation="relu"),   # deeper features -> 11x11x64
    layers.MaxPooling2D((2, 2)),                     # -> 5x5x64
    layers.Flatten(),                                # -> 1600 numbers
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),                             # randomly drop 50% -> less overfitting
    layers.Dense(10, activation="softmax")
])
model.summary()

# %% c. Training the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train,
                    epochs=5, batch_size=64,
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
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
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
