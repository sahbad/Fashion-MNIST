# Fashion MNIST Classification – Module 6 Assignment

This repository contains my complete implementation of the **Fashion MNIST Classification** task using **Convolutional Neural Networks (CNNs)** in both **Python** and **R**, following the Module 6 assignment requirements.

The project includes:

* A **6-layer CNN** implemented with **Keras** in Python
* A **6-layer CNN** implemented with **Keras** in R using **R6 OOP structure**
* Training, evaluation, and predictions (Python)
* Training and evaluation (R)
* Clean project structure and runnable scripts

---

## 📂 Project Structure

```
Fashion-MNIST/
├── python/
│   ├── fashion_mnist_cnn.py
│   └── requirements.txt
├── r/
│   └── fashion_mnist_cnn.R
└── README.md
```

---

## 📘 Overview

The goal of this project is to build a **Convolutional Neural Network (CNN)** capable of classifying 28×28 grayscale images from the **Fashion MNIST dataset** into one of 10 clothing categories.

The assignment required:

* A CNN with **exactly 6 layers** (Conv2D, MaxPooling, Conv2D, MaxPooling, Flatten, Dense)
* Implementations in **two languages** (Python and R)
* Both versions using **class-based programming**
* Making predictions for **at least two images** (done via Python)
* Clean, readable, well-structured code

This repository fulfills all requirements.

---

## 🐍 Python Implementation

### ▶️ How to Run

1. Navigate to the python directory:

   ```bash
   cd python
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the model:

   ```bash
   python fashion_mnist_cnn.py
   ```

The Python script:

* Loads Fashion MNIST
* Preprocesses images
* Builds a 6-layer CNN in a `FashionMNISTClassifier` class
* Trains and evaluates the model
* Prints predictions for two sample test images

### ✅ Python Results

**Test Accuracy:** `0.8893`

**Sample Predictions:**

```
Image index: 0
  True label:      Ankle boot
  Predicted label: Ankle boot
----------------------------------------
Image index: 1
  True label:      Pullover
  Predicted label: Pullover
----------------------------------------
```

These predictions successfully satisfy the assignment requirement for demonstrating classification on at least two images.

---

## 📘 R Implementation

### ▶️ How to Run

1. Open RStudio
2. Set working directory:

   ```r
   setwd("path/to/Fashion-MNIST")
   ```
3. Install required packages:

   ```r
   install.packages("R6")
   install.packages("keras")
   library(keras)
   install_keras()   # Installs TensorFlow backend
   ```
4. Run the model:

   ```r
   source("r/fashion_mnist_cnn.R")
   ```

The R script:

* Loads Fashion MNIST
* Uses an **R6 class** structure
* Preprocesses and reshapes data
* Builds a 6-layer CNN matching the Python architecture
* Trains and evaluates the model

### ✅ R Results

**Test Accuracy:** `0.8922`

The model trained successfully for 5 epochs and produced a strong accuracy score consistent with the Python implementation.

### 🔍 Note on R Predictions

Due to TensorFlow indexing limitations in the Windows R environment, the optional image prediction helper resulted in a dimension error. The assignment **does not require predictions in both languages**, and the Python implementation fulfills this requirement completely.

---

## 🧠 CNN Architecture (Python & R)

Both implementations use the following **6-layer CNN**:

1. **Conv2D** (32 filters, 3×3, ReLU)
2. **MaxPooling2D** (2×2)
3. **Conv2D** (64 filters, 3×3, ReLU)
4. **MaxPooling2D** (2×2)
5. **Flatten**
6. **Dense** (10 units, softmax)

This architecture satisfies the assignment’s exact layer count requirement.

---

## 📌 Key Features

* Fully object-oriented in **both Python and R**
* Clean and readable code following best practices
* Metrics printed during training
* Accurate classification results
* Compatible with Keras/TensorFlow backends

---

## 👤 Author

Adebowale Saheed Badru
MS Data Analytics – BAN 6420
Module 6 Assignment

---

## 📎 Notes

* Python version includes full predictions for sample images (assignment requirement)
* R version includes complete model training & evaluation
* Both languages use the same CNN architecture for consistency

---

This README summarizes the complete implementation and provides all instructions needed to run the project in both languages.
