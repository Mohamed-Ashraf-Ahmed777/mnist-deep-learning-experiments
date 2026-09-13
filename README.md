# mnist-deep-learning-experiments
PyTorch-based neural network experiments on the MNIST handwritten digit dataset, including hyperparameter tuning, CNNs, and regularization analysis.
# MNIST Deep Learning Experiments

A PyTorch-based deep learning study on the **MNIST handwritten digit dataset**, covering fully connected neural networks, hyperparameter tuning, convolutional neural networks, and regularization techniques.

This project was developed as **Assignment 2** and focuses on understanding how different architectures, hyperparameters, and regularization methods affect neural network performance.

---

## Project Overview

The project is organized into four main stages:

* **Part B — Baseline Fully Connected Neural Network**
* **Part C1/C2 — Hyperparameter Analysis and Model Selection**
* **Part D1 — Convolutional Neural Network (CNN)**
* **Part D2 — Regularization Analysis**

All models are implemented and trained using **PyTorch**, with additional evaluation and analysis using Scikit-learn.

---

## Dataset

The project uses the **MNIST handwritten digit dataset**.

* **60,000 images** used from the original MNIST training set
* Image size: **28 × 28 pixels**
* Image type: **grayscale**
* Number of classes: **10**
* Classes: digits `0`–`9`

Images are normalized from:

```text
[0, 255] → [0, 1]
```

### Data Split

A stratified split with `random_state=42` is used:

| Dataset    |    Samples | Percentage |
| ---------- | ---------: | ---------: |
| Training   |     36,000 |        60% |
| Validation |     12,000 |        20% |
| Test       |     12,000 |        20% |
| **Total**  | **60,000** |   **100%** |

`torch.utils.data.Subset` is used to construct the three datasets, while `DataLoader` handles batching.

Training data is shuffled, while validation and test data are not shuffled.

---

# Part B — Baseline Fully Connected Neural Network

## B1 — Architecture

The baseline model is a configurable fully connected neural network implemented using PyTorch.

### Baseline Architecture

```text
Input: 784
   ↓
Linear: 128
   ↓
ReLU
   ↓
Linear: 128
   ↓
ReLU
   ↓
Output: 10
```

Since each MNIST image is `28 × 28`:

```text
28 × 28 = 784 input features
```

The image is flattened before being passed through the fully connected layers.

### Weight Initialization

The baseline model uses **He/Kaiming Normal Initialization** for its linear layers.

Biases are initialized to zero.

### Trainable Parameters

```text
118,282
```

---

## B2 — Training Configuration

| Parameter     | Value              |
| ------------- | ------------------ |
| Loss Function | Cross Entropy Loss |
| Optimizer     | SGD                |
| Learning Rate | 0.01               |
| Batch Size    | 64                 |
| Epochs        | 20                 |
| Random Seed   | 42                 |

---

## B3 — Baseline Results

The baseline model was trained and evaluated using both single-run and repeated experiments.

### Repeated Experiment Results

| Metric              |             Result |
| ------------------- | -----------------: |
| Training Accuracy   | **96.31% ± 0.05%** |
| Validation Accuracy | **95.18% ± 0.11%** |

### Single-Run Results

| Metric                   |     Result |
| ------------------------ | ---------: |
| Best Validation Accuracy | **95.10%** |
| Test Accuracy            | **95.24%** |

The project also generates training and validation loss/accuracy curves to analyze convergence behavior.

### Repeated Experiments

The baseline experiment is repeated using different random seeds.

The resulting learning curves are summarized using:

* Mean performance
* Standard deviation
* Error bars

This provides a more reliable view of model stability rather than relying on a single training run.

---

# Part C — Hyperparameter Analysis

Part C investigates how different hyperparameters affect the performance and convergence of the fully connected neural network.

The experiments use a shorter **3-epoch initial test** to efficiently compare different configurations.

The following hyperparameters are investigated:

1. Learning Rate
2. Batch Size
3. Network Depth
4. Network Width
5. Depth × Width Interaction

---

## C1 — Learning Rate Analysis

The following learning rates were tested:

```text
0.001
0.01
0.1
1.0
```

### Results

| Learning Rate | Final Validation Accuracy | Convergence       |
| ------------: | ------------------------: | ----------------- |
|         0.001 |                    72.49% | Did not reach 90% |
|          0.01 |                    90.22% | Epoch 3           |
|           0.1 |                **95.57%** | Epoch 1           |
|           1.0 |                    73.58% | Did not reach 90% |

### Observation

A learning rate of:

```text
0.1
```

provided the best performance in the tested configurations.

A learning rate of `0.001` converged too slowly, while `1.0` produced unstable behavior.

---

## C2 — Batch Size Analysis

The following batch sizes were evaluated:

```text
16
32
64
128
```

The experiments compare:

* Validation accuracy
* Convergence behavior
* Gradient norm

Gradient norms are tracked to investigate how batch size affects gradient noise and training behavior.

The best batch size is selected according to validation performance and is reused in subsequent experiments.

**Best batch size:** `[INSERT BEST BATCH SIZE]`

---

## C3 — Network Depth Analysis

Different numbers of hidden layers were tested.

### Architectures

```text
Depth 2 → [128, 64]

Depth 3 → [128, 64, 32]

Depth 4 → [128, 64, 32, 16]

Depth 5 → [128, 64, 32, 16, 8]
```

The objective is to study how increasing network depth affects model capacity and validation performance.

---

## C4 — Network Width Analysis

Two-hidden-layer networks with different widths were compared.

### Architectures

```text
Width 64  → [64, 64]

Width 128 → [128, 128]

Width 256 → [256, 256]

Width 512 → [512, 512]
```

This experiment studies the effect of increasing the number of neurons in each hidden layer.

---

## C5 — Depth × Width Analysis

A combined depth × width experiment is used to investigate the interaction between:

* Number of hidden layers
* Number of neurons per layer

The results are summarized in an architecture comparison table and a heatmap.

The best configuration is selected according to validation accuracy.

### Selected Configuration

```text
Best Learning Rate: [INSERT RESULT]

Best Batch Size: [INSERT RESULT]

Best Depth: [INSERT RESULT]

Best Width: [INSERT RESULT]

Best Architecture: [INSERT RESULT]
```

The selected architecture is then retrained and evaluated on the held-out test set.

---

# Part C2 — Final Fully Connected Model

After hyperparameter analysis, the selected configuration is retrained using the best hyperparameters.

The final fully connected model is evaluated on the test set using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Final FC Model Results

| Metric              |            Result |
| ------------------- | ----------------: |
| Validation Accuracy | `[INSERT RESULT]` |
| Test Accuracy       | `[INSERT RESULT]` |
| Precision           | `[INSERT RESULT]` |
| Recall              | `[INSERT RESULT]` |
| F1-score            | `[INSERT RESULT]` |

---

## Error Analysis

The project visualizes misclassified MNIST examples to investigate the types of digits that are most difficult for the model to classify.

This helps identify visually similar digits and common sources of classification errors.

---

# Part D1 — Convolutional Neural Network

A CNN is implemented to compare convolution-based feature extraction with the fully connected architecture.

## CNN Architecture

```text
Input
28 × 28 × 1
      ↓
Conv2D
1 → 32 channels
Kernel = 3 × 3
Padding = 1
      ↓
ReLU
      ↓
MaxPool
2 × 2
      ↓
Conv2D
32 → 64 channels
Kernel = 3 × 3
Padding = 1
      ↓
ReLU
      ↓
MaxPool
2 × 2
      ↓
Flatten
      ↓
Linear
64 × 7 × 7 → 128
      ↓
ReLU
      ↓
Linear
128 → 10
```

After the two pooling layers:

```text
28 × 28
   ↓
14 × 14
   ↓
7 × 7
```

Therefore, the flattened feature representation contains:

```text
64 × 7 × 7 = 3136
```

features.

---

## CNN Training

The CNN uses the best learning rate and batch size selected during the hyperparameter analysis.

Training configuration:

* Optimizer: SGD
* Loss: Cross Entropy Loss
* Epochs: 3
* Learning Rate: Best LR from Part C
* Batch Size: Best Batch Size from Part C

---

## CNN vs Fully Connected Network

The CNN is compared directly with the best fully connected neural network.

| Model                   | Validation Accuracy |     Test Accuracy |
| ----------------------- | ------------------: | ----------------: |
| Best Fully Connected NN |   `[INSERT RESULT]` | `[INSERT RESULT]` |
| CNN                     |   `[INSERT RESULT]` | `[INSERT RESULT]` |

The comparison demonstrates the effect of using convolutional feature extraction for image classification.

---

# Part D2 — Regularization Analysis

The project investigates several regularization techniques to study their effect on generalization and validation performance.

The following configurations are compared:

* Dropout 0.1
* Dropout 0.3
* Dropout 0.5
* Dropout 0.7
* Batch Normalization
* Batch Normalization + Dropout

All experiments use a base architecture of:

```text
[256, 256]
```

---

## D2.1 — Dropout

Dropout is applied after the ReLU activation of each hidden layer.

The following dropout rates are tested:

```text
0.1
0.3
0.5
0.7
```

This experiment investigates how different dropout probabilities affect overfitting and validation performance.

---

## D2.2 — Batch Normalization

The Batch Normalization model uses:

```text
Linear
   ↓
BatchNorm1d
   ↓
ReLU
```

for each hidden layer.

Batch normalization is evaluated as an alternative regularization and optimization technique.

---

## D2.3 — Batch Normalization + Dropout

A combined model uses:

```text
Linear
   ↓
BatchNorm1d
   ↓
ReLU
   ↓
Dropout(0.3)
```

This experiment investigates whether combining the two techniques provides better generalization.

---

## Regularization Results

The final validation accuracy of all six configurations is compared.

| Method              | Validation Accuracy |
| ------------------- | ------------------: |
| Dropout 0.1         |   `[INSERT RESULT]` |
| Dropout 0.3         |   `[INSERT RESULT]` |
| Dropout 0.5         |   `[INSERT RESULT]` |
| Dropout 0.7         |   `[INSERT RESULT]` |
| Batch Normalization |   `[INSERT RESULT]` |
| BatchNorm + Dropout |   `[INSERT RESULT]` |

### Best Regularization Method

```text
Method: [INSERT BEST METHOD]

Validation Accuracy: [INSERT ACCURACY]
```

---

# Evaluation Metrics

The classification models are evaluated using several metrics.

### Accuracy

The percentage of correctly classified samples.

### Precision

Measures how many samples predicted as a class are actually members of that class.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1-score

The harmonic mean of precision and recall.

### Confusion Matrix

Used to analyze class-by-class classification performance and identify commonly confused digits.

---

# Visualizations

The notebook generates several visualizations for analyzing the experiments:

* Training Loss vs Epoch
* Validation Loss vs Epoch
* Training Accuracy vs Epoch
* Validation Accuracy vs Epoch
* Mean ± Standard Deviation Learning Curves
* Learning Rate Comparison
* Batch Size Analysis
* Gradient Norm Analysis
* Network Depth Comparison
* Network Width Comparison
* Depth × Width Heatmap
* Confusion Matrix
* Misclassified MNIST Examples
* CNN vs Fully Connected Network
* Regularization Comparison

---

# Reproducibility

A fixed seed is used throughout the experiments:

```python
SEED = 42
```

Randomness is controlled for:

* Python `random`
* NumPy
* PyTorch
* CUDA when available

The notebook automatically selects the available device:

```text
CUDA GPU → if available
CPU      → otherwise
```

The original experiments were performed using a **Google Colab Tesla T4 GPU** when available.

---

# Model Checkpoint

The baseline model can be saved using PyTorch:

```text
checkpoints/baseline_nn.pth
```

The checkpoint contains information including:

* Model state dictionary
* Architecture
* Input size
* Output size
* Learning rate
* Batch size
* Number of epochs
* Best validation accuracy
* Test accuracy

Model checkpoints are excluded from Git tracking by default because binary model files can become large.

---

# Project Structure

```text
mnist-deep-learning-experiments/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── MNIST_Neural_Network_Assignment_2.ipynb
│
├── images/
│   ├── training_curves.png
│   ├── hyperparameter_analysis.png
│   ├── confusion_matrix.png
│   ├── cnn_comparison.png
│   └── regularization_comparison.png
│
└── checkpoints/
    └── baseline_nn.pth
```

> The MNIST dataset is downloaded automatically through `torchvision` and is not required to be stored in the repository.

---

# Technologies

| Technology   | Purpose                                    |
| ------------ | ------------------------------------------ |
| Python       | Programming language                       |
| PyTorch      | Neural network implementation and training |
| Torchvision  | MNIST dataset                              |
| NumPy        | Numerical computation                      |
| Pandas       | Data analysis                              |
| Scikit-learn | Data splitting and evaluation              |
| Matplotlib   | Visualization                              |
| Seaborn      | Statistical visualization                  |
| Google Colab | Development and GPU training               |

---

# Key Learning Outcomes

This project demonstrates practical experience with:

* Building neural networks using PyTorch
* Implementing configurable fully connected architectures
* Weight initialization
* Forward propagation
* Loss functions
* SGD optimization
* Training and validation pipelines
* Hyperparameter tuning
* Learning-rate analysis
* Batch-size analysis
* Gradient norm analysis
* Network depth and width experiments
* CNN architecture design
* Image classification
* Dropout
* Batch Normalization
* Model comparison
* Error analysis
* Confusion matrices
* Reproducible experiments
* GPU-accelerated deep learning

---

# Conclusion

This project explores the progression from a basic fully connected neural network to more advanced deep learning approaches for MNIST classification.

The experiments demonstrate how:

```text
Baseline NN
     ↓
Hyperparameter Optimization
     ↓
Architecture Analysis
     ↓
CNN
     ↓
Regularization
```

can be used to systematically improve and understand neural network performance.

Rather than focusing only on the final accuracy, the project analyzes **why model performance changes** as architecture, optimization parameters, and regularization techniques are modified.

---

## Author

**Mohamed Ashraf Ahmed**

Computer & Communication Engineering Student
Alexandria University

**Focus:** Artificial Intelligence · Machine Learning · Deep Learning
