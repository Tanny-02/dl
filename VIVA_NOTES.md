# Deep Learning Practical: viva notes for problems 1, 2 and 5

## Fixes made to the Notion code (know these, examiners love them)

1. **CIFAR-10 CNN reshape was wrong.** `cifar-10.csv` stores every image *channel-first*: pixel_0 to pixel_1023 are the red plane, 1024 to 2047 green, 2048 to 3071 blue. `reshape(-1, 32, 32, 3)` scrambles the images. The correct form is `reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)`. (The FNN doesn't care, because it only sees a flat vector.)
2. **CIFAR-10 has no NaNs**, despite the Notion comment. `dropna` is harmless and kept as a safety net. Read it with `dtype="float32"` because the default float64 needs about 1.2 GB of RAM.
3. **Sign Language MNIST** has labels 0 to 24 with 9 (J) missing. There are 24 real classes, but the output layer needs `Dense(25)` because sparse CE needs `units > max label`.
4. **Transfer learning: no internet needed.** The dataset folder already includes `vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5`, so pass that path as `weights=` instead of `"imagenet"` (which downloads 58 MB).
5. **Fashion-MNIST image folder has a stray `train/` subfolder.** Without `class_names=[...]` Keras treats it as an 11th class.
6. **Transfer learning used 224×224 and Flatten.** On a lab CPU that is very slow. Use 128×128 (Caltech/Intel) or 64×64 (Fashion), plus `GlobalAveragePooling2D`: 512 features instead of 25,088, so the Dense layer goes from 6.4M to 131k weights.
7. The fine-tuning plot only showed stage 1. The plots now join both stages with a marker line.
8. Notion's "2.a" is titled MNIST but the code is CIFAR-10. The files are now named by dataset.

---

## Problem 1: Feedforward Neural Network (FNN)

**What it is:** layers of fully connected (Dense) neurons. Data flows input → hidden → output with no loops. Each neuron computes `activation(w·x + b)`.

| Concept | One-line answer |
|---|---|
| Why divide by 255? | Normalises pixels to 0–1, so gradients are stable and training is faster. |
| Why 784 inputs? | 28×28 image flattened. CIFAR is 32×32×3 = 3072. |
| ReLU | `max(0, x)`. Cheap, avoids vanishing gradients. |
| Softmax | Turns the output scores into probabilities that sum to 1. Argmax gives the predicted class. |
| sparse_categorical_crossentropy | Multi-class loss when labels are integers. Use `categorical_crossentropy` if labels are one-hot. |
| SGD | Stochastic Gradient Descent: `w = w − lr × gradient`, computed on mini-batches (batch_size=32). |
| Adam | SGD + momentum + per-weight adaptive learning rate. Usually converges faster. |
| Epoch / batch | Epoch = one full pass over the training data. Batch = samples per weight update. |
| validation_split | Holds out part of the training data to detect overfitting (when val loss rises while train loss falls). |
| Backpropagation | Chain rule computes dLoss/dw layer by layer, from output back to input. |
| Why does the FNN get only ~50% on CIFAR? | Flattening throws away the spatial structure, and colour objects vary a lot. CNNs fix this. |
| Params of Dense(128) after 784 inputs | 784×128 + 128 = 100,480 |

## Problem 2: CNN

**Four stages:** load and preprocess → architecture → train → evaluate.

| Concept | One-line answer |
|---|---|
| Conv2D(32, (3,3)) | 32 learnable 3×3 filters slide over the image, and each produces a feature map (edges, textures). |
| Why a CNN beats an FNN on images | Local connectivity and weight sharing mean far fewer parameters. It also detects features anywhere in the image. |
| MaxPooling2D(2,2) | Keeps the max of each 2×2 window, halving H and W. This reduces computation and adds a small translation invariance. |
| padding="same" vs "valid" | "same" keeps the size by zero-padding. "valid" (the default) shrinks it: 28 → 26 with a 3×3 filter. |
| Flatten | Converts the 3-D feature maps to a 1-D vector for the Dense layers. |
| Dropout(0.5) | Randomly zeroes 50% of neurons during training only, which reduces overfitting. |
| Reshape (-1, 28, 28, 1) | Conv layers need (samples, height, width, channels). The -1 means "work it out". |
| Conv params | (kh × kw × in_channels + 1) × filters. For example, the first conv on MNIST is (3×3×1+1)×32 = 320. |
| Output size formula | (W − K + 2P)/S + 1 |
| classification_report | Per-class precision (of those predicted X, how many were X), recall (of the real X, how many were found) and F1. |

## Problem 5: Transfer learning (VGG16)

**Idea:** reuse a CNN already trained on ImageNet (1.2M images, 1000 classes). Its early layers detect generic edges and textures, which are useful for any image task, so you only train a small new head.

| Step | Code | Why |
|---|---|---|
| a. Load pretrained | `VGG16(weights=..., include_top=False)` | `include_top=False` removes VGG's 1000-class Dense layers and keeps the 13 conv layers. |
| b. Freeze | `base_model.trainable = False` | Keeps the ImageNet features intact. Large gradients from the random new head would destroy them. |
| c. Custom classifier | GAP → Dense(256) → Dropout → Dense(num_classes, softmax) | Maps VGG features to *our* classes. |
| d. Train head | Adam lr=1e-3, 5 epochs | Only the head's weights update, so it is fast. |
| e. Fine-tune | Unfreeze the last 4 layers (block5), Adam **lr=1e-5**, recompile, 3 epochs | Adapts high-level features to the new data. The tiny lr avoids wrecking the pretrained weights. You **must recompile** after changing `trainable`. |

**Likely viva questions**
- **VGG16:** 13 conv + 3 dense = 16 weight layers, all 3×3 convs, 5 blocks each ending in max-pool, about 138M params with the top (14.7M without).
- **preprocess_input:** converts RGB to BGR and subtracts the ImageNet mean per channel, matching how VGG was trained.
- **Is this really "object detection"?** No. Strictly it is image *classification* (one label per image). Detection also predicts bounding boxes (YOLO, SSD, Faster R-CNN). The syllabus uses the word loosely.
- **Why not train VGG from scratch?** It needs millions of images and days of GPU time. Caltech-101 has only about 9k images, so it would overfit.
- **GlobalAveragePooling vs Flatten:** GAP averages each feature map to 1 number, giving far fewer params and less overfitting.
- **Which layers to unfreeze first?** The top or later ones. They hold task-specific features, while lower layers are generic.
- **Other pretrained models:** ResNet50 (skip connections), MobileNet (light, depthwise convs), InceptionV3, EfficientNet.

## Exam-day checklist
- Change `PATH` / `DATA_DIR` to wherever the CSVs and folders are on the lab machine.
- Run `model.summary()` and be ready to explain one param count.
- If training is too slow, cut epochs (and IMG_SIZE for transfer learning) and say why.
- Always show: test accuracy, the classification report, the loss/accuracy plots and (for CNN/TL) sample predictions.

---

## Problem 3: Autoencoders for anomaly detection

**Core idea:** an autoencoder compresses its input through a small bottleneck and rebuilds it, trained with `fit(X, X)`. Train it on **normal data only**. Anomalies reconstruct badly, so their **reconstruction error is high**. Error above a threshold (for example, the 95th percentile of the training error) means anomaly. Labels are used only for evaluation.

| Problem | Data | Model |
|---|---|---|
| 3.a Credit card fraud | Tabular, 29 features | Dense AE 29→16→8→16→29 |
| 3.b MVTec AD | Images | Conv AE (Conv+MaxPool encoder, Conv+UpSampling decoder) |
| 3.c Smart grid | Time series | LSTM AE on sliding windows |
| 3.d NASA turbofan | Sensor time series per engine | LSTM AE on per-engine windows |
| 3.e NSL-KDD | Tabular, categorical features | One-hot encoding + MinMax scaling → Dense AE |
| 3.f Sign MNIST | Images (flattened) | Dense AE; class 24 (Y) is held out as the anomaly |

**Fixes made to 3.a:** a held-out test set (Notion measured errors on its own training data), added metrics (confusion matrix, precision/recall/F1, ROC-AUC), dropped `Time`, and a log-scale histogram.

| Question | Answer |
|---|---|
| Why is the bottleneck smaller than the input? | Otherwise the network just copies the input. The bottleneck forces it to learn structure. |
| Why MSE? | Reconstruction is regression on continuous values. |
| Linear vs sigmoid output | Linear when the scaled data can be negative (StandardScaler). Sigmoid when the data is in 0–1 (MinMax scaling or pixels/255). |
| Why not accuracy? | Fraud is 0.17% of the data, so "always normal" scores 99.8%. Use precision, recall, F1 and ROC-AUC. |
| Why is precision low in 3.a? | The 95th-percentile threshold flags 5% of about 57k test normals, roughly 2,800 false alarms against about 400 frauds. That's the class imbalance, not a broken model. |
| Threshold trade-off | A higher percentile gives fewer false alarms but more missed anomalies. |
| Fit the scaler on what? | Training data only, to avoid leakage. |
| Why an LSTM AE (3.c/3.d)? | Anomalies there are patterns over time. `RepeatVector` copies the encoded vector to every timestep for the decoder, and `TimeDistributed(Dense)` rebuilds each timestep. |
| Why a Conv AE (3.b)? | It keeps spatial structure, and the per-pixel error map shows *where* the defect is. |
