"""
Problem 5.a - LIGHTWEIGHT version of transfer learning with VGG16 (for slow lab computers).

Same steps as 5_TL_caltech101.py, but it trains on only a few batches (train_ds.take(10),
test_ds.take(3)) and for very few epochs, so it finishes quickly on a CPU. The accuracy
will be lower: it's meant to demonstrate the steps, not to get a good score.

Works for any one-folder-per-class dataset: point DATA_DIR at caltech-101 (5.a)
or Intel Image Classification (5.b).
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input


# 1. Set up paths (change to where the dataset is on your machine)
DATA_DIR = "../LP IV Updated Datasets/Object Detection/caltech-101"
# Local VGG16 weights file from the lab dataset folder (no internet needed).
# With internet you can use weights='imagenet' instead.
WEIGHTS = "../LP IV Updated Datasets/Object Detection/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# 2. Get class names (skip Caltech's BACKGROUND_Google clutter class)
class_names = sorted([
    f for f in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, f))
    and f != "BACKGROUND_Google"
])

print("Total Classes:", len(class_names))


# 3. Load and preprocess data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=class_names,
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=class_names,
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Use only a few batches so it runs fast on lab computers
train_ds = train_ds.take(10)
test_ds = test_ds.take(3)


# 4. Load pretrained VGG16 model
base_model = VGG16(
    weights=WEIGHTS,
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


# 5. Add custom layers
inputs = Input(shape=(224, 224, 3))

x = tf.keras.layers.Lambda(
    tf.keras.applications.vgg16.preprocess_input
)(inputs)

x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)


# 6. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 7. Train model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=2
)


# 8. Compute loss and evaluate model
loss, accuracy = model.evaluate(test_ds)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# 9. Visualize predictions
images, labels = next(iter(test_ds))
predictions = model.predict(images)

plt.figure(figsize=(10, 8))

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i].numpy().astype("uint8"))

    actual = labels[i].numpy()
    pred = np.argmax(predictions[i])
    plt.title("Actual: " + class_names[actual] + "\nPredicted: " + class_names[pred])
    plt.axis('off')

plt.show()


# 10. Fine-tune: unfreeze the last 4 layers of VGG16, retrain with a tiny learning rate
base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=1
)

print("Fine-tuning completed!")
