# Deep Learning Practical (LP IV)

Code for all 20 problem statements of the Deep Learning practical, using Keras / TensorFlow.

| Folder | Problems | Notes |
|---|---|---|
| `1_FNN/` | 1.a–1.d: Feedforward NN on MNIST, Fashion MNIST, CIFAR-10, Sign Language MNIST | Cleaned up and tested |
| `2_CNN/` | 2.a–2.c: CNN on CIFAR-10, MNIST, Fashion MNIST, Sign Language MNIST | Cleaned up and tested |
| `3_Autoencoders/` | 3.a–3.f: anomaly detection with autoencoders | 3.a corrected, 3.b newly written, 3.c–3.f from Notion |
| `4_CBOW/` | 4.a–4.d: Word2Vec CBOW (classification, Twitter, text document, recommender) | From Notion |
| `5_Transfer_Learning/` | 5.a–5.c: VGG16 transfer learning on Caltech-101, Intel, Fashion MNIST | Cleaned up |
| `notion_original/` | The original Notion code for every problem, unmodified, with its problem statement on top | Reference |
| `VIVA_NOTES.md` | The bugs that were fixed and likely viva questions | |

## Running
```bash
pip install -r requirements.txt
```
Each script runs top to bottom. You can also paste it into Jupyter cell by cell, since the `# %%` markers split it into cells.

**Datasets are not in this repo** because they're too large. Before running a script, change its `PATH`, `DATA_DIR` or `read_csv(...)` line to point at your copy.
- Problems 1, 2 and 5 expect the lab's `LP IV Updated Datasets/` folder next to the code folder.
- Problems 3 and 4 read the file name directly, for example `creditcard.csv` or `KDDTrain+.txt`.

The VGG16 weights file (`vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5`) is already in the lab dataset folder, so problem 5 works without internet.

## Known gotchas
- **CIFAR-10 CSV is channel-first.** Use `reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)`, not `reshape(-1, 32, 32, 3)`.
- **Sign Language MNIST** labels run 0–24 with no 9 (J), so the output layer needs `Dense(25)`.
- **Fashion MNIST image folder** has a stray `train/` subfolder, so pass `class_names=[...]` to `image_dataset_from_directory`.
- **Notion's "2.a"** is titled MNIST but the code is CIFAR-10.
