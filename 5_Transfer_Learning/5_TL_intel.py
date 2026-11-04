# Problem 5 - Object detection (image classification) using Transfer Learning: VGG16 on Intel Image Classification
# a) load pretrained CNN  b) freeze lower conv layers  c) add custom classifier
# d) train classifier  e) fine-tune: unfreeze top layers + low learning rate

# %% Set up the environment
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

DATA_DIR = "../LP IV Updated Datasets/Object Detection/Intel Image Classification"
WEIGHTS = "../LP IV Updated Datasets/Object Detection/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
IMG_SIZE = (128, 128)   # VGG16's native size is 224; smaller trains MUCH faster on a CPU
BATCH = 32
# 6 scene classes (buildings, forest, glacier, mountain, sea, street), ~12,600 images.

# %% Loading and preprocessing data (one folder per class -> label = folder name)
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH)

class_names = train_ds.class_names
num_classes = len(class_names)
print(num_classes, "classes:", class_names[:10], "...")

train_ds = train_ds.prefetch(2)   # load next batch while the current one trains
val_ds = val_ds.prefetch(2)

# %% a. Load a pre-trained CNN (VGG16 trained on ImageNet)
# include_top=False drops VGG's own 1000-class classifier, keeping only the conv "feature extractor".
# Weights come from the local .h5 file (no internet needed). With internet you could use weights="imagenet".
base_model = VGG16(weights=WEIGHTS, include_top=False, input_shape=IMG_SIZE + (3,))

# %% b. Freeze the lower convolutional layers
base_model.trainable = False      # stage 1: freeze ALL of VGG16, train only our new layers

# %% c. Add a custom classifier with trainable layers
model = keras.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Lambda(preprocess_input),          # same preprocessing VGG16 was trained with
    base_model,
    layers.GlobalAveragePooling2D(),          # 512 feature maps -> 512 numbers
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])
model.summary()

# %% d. Train the classifier layers (compute loss + accuracy)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"After training head  -> loss: {loss:.4f}  accuracy: {acc:.4f}")

# %% e. Fine-tune: unfreeze the top conv block, retrain with a tiny learning rate
base_model.trainable = True
for layer in base_model.layers[:-4]:          # keep everything except block5 frozen
    layer.trainable = False
print("Trainable VGG layers:", [l.name for l in base_model.layers if l.trainable])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),   # must recompile after changing trainable
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=3)

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"After fine-tuning    -> loss: {loss:.4f}  accuracy: {acc:.4f}")

# %% Visualize predictions
images, labels = next(iter(val_ds))
preds = np.argmax(model.predict(images, verbose=0), axis=1)
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    ok = preds[i] == labels[i].numpy()
    plt.title(f"P: {class_names[preds[i]]}\nA: {class_names[labels[i]]}",
              color="green" if ok else "red", fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.show()

# %% Plot loss and accuracy for both stages
acc_all = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc_all = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss_all = history.history["loss"] + history_fine.history["loss"]
val_loss_all = history.history["val_loss"] + history_fine.history["val_loss"]
start_ft = len(history.history["loss"]) - 0.5

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_all, label="Training loss"); plt.plot(val_loss_all, label="Validation loss")
plt.axvline(start_ft, color="gray", linestyle="--", label="Fine-tuning starts")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc_all, label="Training accuracy"); plt.plot(val_acc_all, label="Validation accuracy")
plt.axvline(start_ft, color="gray", linestyle="--", label="Fine-tuning starts")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout()
plt.show()
