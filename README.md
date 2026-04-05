# Handwritten Digit Recognition using Deep Learning

### Dataset: MNIST | Models: ANN · Basic CNN · Improved CNN | Framework: TensorFlow / Keras

---

## Problem Statement

Handwritten digit recognition is a foundational problem in computer vision with real-world applications in postal automation, bank cheque processing, form digitization, and document scanning. The challenge is to build a model that can correctly identify any handwritten digit (0–9) from a raw 28×28 grayscale image — accounting for natural variation in writing style, stroke thickness, and slant across different people.

This project addresses the problem by building and comparing **three deep learning models of increasing architectural complexity**: a plain Artificial Neural Network (ANN), a Basic CNN, and an Improved CNN with Batch Normalization and Dropout. The goal is not just to maximize accuracy, but to understand *why* each architecture performs differently and what role spatial learning plays in image classification.

---

## Dataset

**MNIST — Modified National Institute of Standards and Technology Database**

| Property | Details |
|---|---|
| Total Images | 70,000 grayscale images |
| Training Set | 60,000 images |
| Test Set | 10,000 images |
| Image Size | 28 × 28 pixels, single channel |
| Classes | 10 (digits 0 through 9) |
| Pixel Range | 0 – 255 (normalized to 0.0 – 1.0) |
| Class Balance | Nearly balanced (~6,000 samples per class in training set) |

**Dataset Link:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

Available directly via Keras (no manual download needed):
```python
from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

---

## Project Structure

```
├── DL_MNIST_Digit_Recognition_DL_Project.ipynb   # Main notebook
├── mnist_samples.png                              # Sample images grid
├── comparison.png                                 # Accuracy + val curve comparison
└── misclassified.png                              # Misclassified examples (Improved CNN)
```

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Deep Learning Framework | TensorFlow 2.x / Keras |
| Numerical Computing | NumPy |
| Visualization | Matplotlib, Seaborn |
| Evaluation Metrics | Scikit-learn (`confusion_matrix`, `classification_report`) |
| Training Utility | `EarlyStopping` callback (patience = 5, restore best weights) |

---

## Preprocessing

Before model training, the raw MNIST data undergoes the following steps:

**Normalization** — Pixel values are scaled from [0, 255] to [0.0, 1.0] by dividing by 255. This prevents large gradient updates and speeds up convergence.

**Reshaping for ANN** — Images are flattened from (28, 28) to a 784-dimensional vector, treating each pixel as an independent feature.

**Reshaping for CNNs** — Images are reshaped to (28, 28, 1) to add a channel dimension, as required by Conv2D layers.

**Label Encoding** — Integer labels are one-hot encoded into 10-class vectors using `to_categorical`.

```
ANN input shape  : (60000, 784)
CNN input shape  : (60000, 28, 28, 1)
Label shape      : (60000, 10)
```

---

## Model Architectures

### Model 1 — ANN (Artificial Neural Network)

A fully connected network with no convolutional layers. Treats each of the 784 pixels as an independent input feature with no concept of spatial relationships.

| Layer | Output Shape | Parameters |
|---|---|---|
| Dense (256, ReLU) | (None, 256) | 200,960 |
| Dropout (0.3) | (None, 256) | 0 |
| Dense (128, ReLU) | (None, 128) | 32,896 |
| Dropout (0.3) | (None, 128) | 0 |
| Dense (10, Softmax) | (None, 10) | 1,290 |

**Limitation:** Cannot detect that the same edge feature appearing at slightly different positions is the same thing. Every pixel position is treated independently.

---

### Model 2 — Basic CNN

Introduces two convolutional blocks before the dense head. Learns local spatial patterns (edges, curves, strokes) through shared filter weights.

| Layer | Output Shape | Notes |
|---|---|---|
| Conv2D (32 filters, 3×3, ReLU) | (28, 28, 32) | padding='same' |
| MaxPooling2D (2×2) | (14, 14, 32) | Downsamples by 2 |
| Conv2D (64 filters, 3×3, ReLU) | (14, 14, 64) | padding='same' |
| MaxPooling2D (2×2) | (7, 7, 64) | Downsamples by 2 |
| Flatten | (3136,) | |
| Dense (128, ReLU) | (128,) | |
| Dropout (0.5) | (128,) | |
| Dense (10, Softmax) | (10,) | |

**Key improvement over ANN:** Weight sharing in convolutions means the model can detect the same feature at any position in the image (translation invariance).

---

### Model 3 — Improved CNN with BatchNorm + Dropout

Extends the Basic CNN with three convolutional blocks, Batch Normalization after every conv and dense layer, and multi-stage Dropout for aggressive regularization. Built using the Keras Functional API.

| Block | Layers |
|---|---|
| Conv Block 1 | Conv2D (32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25) |
| Conv Block 2 | Conv2D (64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25) |
| Conv Block 3 | Conv2D (128, 3×3) → BatchNorm → ReLU → Dropout(0.25) |
| Dense Block | Flatten → Dense(256) → BatchNorm → ReLU → Dropout(0.5) |
| Output | Dense(10, Softmax) |

**What BatchNorm does:** Normalizes layer inputs during training to have zero mean and unit variance. This stabilizes gradients, allows higher learning rates, and reduces sensitivity to weight initialization.

**What the extra Dropout does:** Randomly zeroes neuron outputs during training, forcing the network to learn redundant representations and preventing over-reliance on specific features.

---

## Training Configuration

All three models share the same training setup:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 128 |
| Max Epochs | 30 |
| Validation Split | 10% of training data |
| Early Stopping | patience=5, restore_best_weights=True |

Training and validation accuracy/loss curves are plotted for each model after training.

---

## Evaluation

Each model is evaluated with:
- **Test Accuracy** — Overall percentage of correctly classified digits
- **Classification Report** — Per-class precision, recall, and F1-score (3 decimal places)
- **Confusion Matrix** — 10×10 heatmap showing true vs. predicted labels

Additionally, for the Improved CNN, all misclassified test images are identified and the first 10 are displayed with their true and predicted labels.

---

## Results

| Model | Test Accuracy | Approx. Parameters |
|---|---|---|
| ANN | ~97.5% | ~235,000 |
| Basic CNN | ~99.0% | ~93,000 |
| Improved CNN + BN | ~99.3% | ~320,000 |

The Improved CNN misclassifies fewer than ~70 images out of 10,000.

---

## Key Observations & Conclusions

**ANN achieves ~97.5% — MNIST is not a hard dataset.** Even treating pixels as independent features gets you close. But the ANN hits a ceiling because it has no notion of spatial structure.

**CNNs break through that ceiling.** By learning filters that detect local patterns — horizontal edges, curves, diagonal strokes — and applying them across the entire image via weight sharing, the Basic CNN jumps to ~99% with *fewer* parameters than the ANN.

**BatchNorm + Dropout is the final push.** The Improved CNN's main gains come from training stability (BatchNorm) and better generalization (multi-stage Dropout). The validation accuracy curve converges faster and more smoothly compared to the Basic CNN.

**Where models still fail:** The remaining misclassified examples are genuinely ambiguous — a poorly written `4` that looks like a `9`, or a slanted `1` that resembles a `7`. These are edge cases that even humans might misread.

---

## How to Run

1. Clone the repository and open the notebook:
```bash
jupyter notebook DL_MNIST_Digit_Recognition_DL_Project.ipynb
```

2. Run all cells in order. MNIST downloads automatically via Keras on first run.

3. GPU is optional but recommended for the Improved CNN. The notebook prints GPU availability on startup.

**Dependencies:**
```
tensorflow>=2.0
numpy
matplotlib
seaborn
scikit-learn
```
