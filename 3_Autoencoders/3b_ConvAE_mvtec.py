"""
Problem 3.b - Industrial equipment anomaly detection on MVTec AD using a Convolutional Autoencoder
(The Notion page had no code for this one; written from the problem statement.)

Steps:
  - Preprocess images (resize, normalize) for input into a convolutional autoencoder.
  - Build a convolutional autoencoder (CNN encoder-decoder) to learn normal (defect-free) images.
  - Use pixel reconstruction error to detect anomalies (defects) in test images.

MVTec AD folder layout (one folder per object category, e.g. bottle, cable, screw, ...):
  <DATA_DIR>/<CATEGORY>/train/good/*.png        <- only defect-free images (used for training)
  <DATA_DIR>/<CATEGORY>/test/good/*.png         <- defect-free test images  (label 0)
  <DATA_DIR>/<CATEGORY>/test/<defect_type>/*.png <- defective test images   (label 1)
"""

# 1. Import packages
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "mvtec_anomaly_detection"   # change to where the dataset is extracted
CATEGORY = "bottle"                    # any MVTec category: bottle, cable, capsule, screw, ...
IMG_SIZE = 128                         # originals are ~900-1024 px; 128 keeps it fast on a CPU


# 2. Load and preprocess images (resize + normalize to 0-1)
def load_images(paths):
    images = []
    for p in paths:
        img = keras.utils.load_img(p, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
        images.append(keras.utils.img_to_array(img) / 255.0)
    return np.array(images, dtype="float32")


train_paths = sorted(glob.glob(os.path.join(DATA_DIR, CATEGORY, "train", "good", "*.png")))
X_train = load_images(train_paths)

test_paths, y_test = [], []
for defect_type in sorted(os.listdir(os.path.join(DATA_DIR, CATEGORY, "test"))):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, CATEGORY, "test", defect_type, "*.png")))
    test_paths += paths
    y_test += [0 if defect_type == "good" else 1] * len(paths)   # 0 = normal, 1 = defect
X_test = load_images(test_paths)
y_test = np.array(y_test)

print("Train (good only):", X_train.shape)
print("Test:", X_test.shape, " normal:", np.sum(y_test == 0), " defective:", np.sum(y_test == 1))

# 3. Build the convolutional autoencoder
model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Encoder: Conv + MaxPool shrinks 128 -> 64 -> 32 -> 16
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),                       # bottleneck: 16 x 16 x 128

    # Decoder: Conv + UpSampling grows 16 -> 32 -> 64 -> 128
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")   # pixels back in 0-1
])
model.summary()

# 4. Compile
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 5. Train ONLY on defect-free images (input = target)
history = model.fit(X_train, X_train, epochs=30, batch_size=16,
                    validation_split=0.1, shuffle=True)

# 6. Reconstruction error per image = mean squared pixel error
train_recon = model.predict(X_train)
test_recon = model.predict(X_test)
train_error = np.mean(np.square(X_train - train_recon), axis=(1, 2, 3))
test_error = np.mean(np.square(X_test - test_recon), axis=(1, 2, 3))

# 7. Threshold = 95th percentile of the error on good training images
threshold = np.percentile(train_error, 95)
print("Threshold:", threshold)

# 8. Detect defects and evaluate
y_pred = (test_error > threshold).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Good", "Defective"], zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, test_error))

# 9. Plot training loss
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Conv autoencoder loss"); plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.legend(); plt.show()

# 10. Plot reconstruction error distribution
plt.hist(test_error[y_test == 0], bins=30, alpha=0.6, label="Good")
plt.hist(test_error[y_test == 1], bins=30, alpha=0.6, label="Defective")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.title(f"MVTec AD ({CATEGORY}) - reconstruction error")
plt.xlabel("Reconstruction error"); plt.ylabel("Number of images")
plt.legend(); plt.show()

# 11. Visualize: original, reconstruction and pixel error map for a few defective images
defect_idx = np.where(y_test == 1)[0][:4]
plt.figure(figsize=(9, 3 * len(defect_idx)))
for row, i in enumerate(defect_idx):
    error_map = np.mean(np.square(X_test[i] - test_recon[i]), axis=-1)   # per-pixel error
    for col, (img, title) in enumerate([(X_test[i], "Original"),
                                        (test_recon[i], "Reconstruction"),
                                        (error_map, "Error map (defect = bright)")]):
        plt.subplot(len(defect_idx), 3, row * 3 + col + 1)
        plt.imshow(img, cmap="hot" if col == 2 else None)
        plt.title(title, fontsize=9); plt.axis("off")
plt.tight_layout(); plt.show()
