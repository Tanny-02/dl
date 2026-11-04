# Problem 1 - Feedforward Neural Network (FNN) on Fashion MNIST
# Steps: a) imports  b) load data  c) architecture  d) train (SGD with 11 epochs)  e) evaluate  f) plot

# %% a. Import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report

# %% b. Load the training and testing data
PATH = "../LP IV Updated Datasets/FNN and CNN/Fashion MNIST/"          # change if your CSVs are elsewhere
train = pd.read_csv(PATH + "fashion-mnist_train.csv")
test = pd.read_csv(PATH + "fashion-mnist_test.csv")

# column "label" = class, other 784 columns = pixels of a 28x28 image
X_train = train.drop("label", axis=1).values / 255.0    # scale 0-255 -> 0-1
y_train = train["label"].values
X_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print("Train:", X_train.shape, " Test:", X_test.shape)

# %% c. Define the network architecture using Keras
model = keras.Sequential([
    layers.Input(shape=(784,)),              # 28*28 = 784 inputs (flattened image)
    layers.Dense(128, activation="relu"),    # hidden layer 1
    layers.Dense(64, activation="relu"),     # hidden layer 2
    layers.Dense(10, activation="softmax")    # one output per class -> probabilities
])
model.summary()

# %% d. Train the model using SGD with 11 epochs
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss="sparse_categorical_crossentropy",   # labels are integers, not one-hot
              metrics=["accuracy"])

history = model.fit(X_train, y_train,
                    epochs=11, batch_size=32,
                    validation_split=0.1)       # 10% of train held out to watch overfitting

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
