# Classify-Produce-DL-Optimization
Hybrid Deep Learning Framework: Comparative Analysis of CNN and PCA-driven FFNN Classification. 
# Hybrid Image Classification: CNN vs. PCA-FFNN Framework

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)

## Project Overview
This project implements and benchmarks two different computational strategies for image classification using the Kaggle Fruits and Vegetables dataset. The objective is to compare **Spatial Feature Extraction** (CNN) against **Statistical Dimensionality Reduction** (PCA + FFNN).

As an MSc Candidate in Computational Biology, I developed this framework to demonstrate the transferability of high-dimensional data processing techniques—moving from Computer Vision to the logic required for **Genomic Signal Analysis**.

## Architectures

### 1. Convolutional Neural Network (CNN)
A Deep Learning model optimized for hierarchical pattern recognition.
* **Core Layers:** Conv2D for spatial patterns, MaxPooling2D for downsampling.
* **Regularization:** L2 weight regularization and Dropout to ensure generalization.
* **Optimization:** Adam Optimizer with Early Stopping to prevent overfitting.

### 2. PCA + Feed-Forward Neural Network (FFNN)
A hybrid approach focused on latent space representation, mirroring workflows used in **Omics data analysis**.
* **Pre-processing:** StandardScaler for feature normalization.
* **Dimensionality Reduction:** Principal Component Analysis (PCA) to extract the most informative components from raw pixel data.
* **Classification:** A Dense Neural Network (FFNN) trained on the reduced feature space.

## Tech Stack
* **Data & OS:** `numpy`, `pandas`, `os`, `json`
* **Deep Learning:** `tensorflow.keras`
* **Machine Learning & Stats:** `sklearn` (PCA, StandardScaler, Metrics)
* **Visualization:** `matplotlib`

## Evaluation Metrics
The models are benchmarked using:
* **Confusion Matrix:** To visualize per-class misclassifications.
* **Classification Report:** Detailed Precision, Recall, and F1-Score metrics.
* **Training Logs:** Convergence analysis via accuracy/loss curves.

## Quick Start & Usage
Since this project is optimized for cloud environments (Kaggle/Google Colab), you can run the analysis without downloading the dataset locally:

1. **Access the Notebook:** Open the `.ipynb` file included in this repository.
2. **Environment Setup:** If running locally, install dependencies via:
   ```bash
   pip install numpy pandas tensorflow scikit-learn matplotlib
