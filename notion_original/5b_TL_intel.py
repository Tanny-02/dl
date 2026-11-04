"""
Problem 5.b (original code from the shared Notion page, unmodified)

Object detection using Transfer Learning of CNN architectures for the Intel Image
Classification Dataset using the below steps:
a. Set up the environment
b. Loading and preprocessing data
c. Choose the pre-trained Model (Transfer Learning)

d. Freeze the base layers and add custom layers for fine tuning
e. Object detection training
f.
Compute loss
g. Evaluate the Model
h. Visualize the prediction

i. Fine Tune the Model: Fine-tuning involves unfreezing the top layers of the pre
trained model and retraining it on the new dataset to adapt it better to your task.
"""

# 1. Import necessary packages

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16


# 2. Load and preprocess dataset

dataset_path = r"C:\Users\Prajwal\PICT\TE SEM 7\LP3\LP IV Updated Datasets\Object Detection\Intel Image Classification"

train_data = keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

test_data = keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)


# Get class names

class_names = train_data.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of Classes:", num_classes)


# 3. Choose pretrained VGG16 model

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)


# 4. Freeze base model layers

base_model.trainable = False


# 5. Define model architecture

model = keras.Sequential([

    layers.Input(shape=(224, 224, 3)),

    layers.Lambda(
        keras.applications.vgg16.preprocess_input
    ),

    base_model,

    layers.Flatten(),

    layers.Dense(256, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')

])


# 6. Compile model

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 7. Train model

history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)


# 8. Compute loss and evaluate model

loss, accuracy = model.evaluate(test_data)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# 9. Fine Tune the model

base_model.trainable = True

# Freeze all layers except last 4 layers

for layer in base_model.layers[:-4]:
    layer.trainable = False


# Compile model again

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Fine tune model

history_fine = model.fit(
    train_data,
    epochs=3,
    validation_data=test_data
)


# 10. Evaluate fine tuned model

loss, accuracy = model.evaluate(test_data)

print("Fine Tuned Test Loss:", loss)
print("Fine Tuned Test Accuracy:", accuracy)


# 11. Visualize predictions

for images, labels in test_data.take(1):

    predictions = model.predict(images)

    plt.figure(figsize=(10, 10))

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class = class_names[
            np.argmax(predictions[i])
        ]

        actual_class = class_names[labels[i].numpy()]

        plt.title(
            "Predicted: " + predicted_class +
            "\nActual: " + actual_class
        )

        plt.axis('off')

    plt.show()


# 12. Plot training loss and accuracy

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
