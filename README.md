# 🏦 Explainable Customer Churn Prediction using TabNet & SHAP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-TabNet-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview
This repository contains the implementation of my undergraduate thesis titled **"Customer Churn Prediction in Banking Industry using TabNet Architecture with SMOTE-Tomek and SHAP Interpretation"**.

Customer churn is a critical issue in the banking sector. This project proposes a deep learning approach using **TabNet (Tabular Network)**, which combines the benefits of decision trees and neural networks. To address data imbalance, **SMOTE-Tomek** hybrid sampling is applied. Furthermore, **SHAP (SHapley Additive exPlanations)** is integrated to provide interpretability, ensuring the model is not a "black box".

## 💾 Dataset
The dataset used in this project is obtained from Kaggle. Due to size constraints, the raw data is not included in this repository.

**[Download Dataset Here (Kaggle)](https://www.kaggle.com/datasets/sandiledesmondmfazi/bank-customer-churn)**

> **Instruction:** After downloading, please create a folder named `data/` in your local directory and extract the CSV file there before running the notebook.

## 🚀 Key Features
* **Sequential Attention Mechanism:** Uses TabNet to perform instance-wise feature selection.
* **Hybrid Resampling:** Implements **SMOTE-Tomek** to handle highly imbalanced datasets effectively.
* **Explainable AI (XAI):** Distinct global and local interpretability using **SHAP values** to identify key churn drivers.
* **Statistical Validation:** Performance superiority is validated using **McNemar's Test**.

## 🛠️ Methodology & Pipeline

The system design consists of two main stages: **Model Construction** and **Evaluation/Interpretation**.

### 1. Architecture Flow
![TabNet Architecture](Customer-Churn-TabNet-SHAP/images/flowchart_tabnet.png)
*(Note: Pastikan nama file gambar di folder images sesuai dengan ini)*

### 2. Experimental Workflow
1.  **Data Preprocessing:** Cleaning, Encoding, and Normalization.
2.  **Handling Imbalance:** Applying SMOTE (Oversampling) + Tomek Links (Undersampling).
3.  **Training:** TabNet with Sparsemax activation and Ghost Batch Normalization.
4.  **Evaluation:** Metrics calculation (F1, Recall) and Statistical Testing.
5.  **Interpretation:** SHAP Summary Plots and Force Plots.

## 📊 Results

The proposed TabNet model outperformed baseline models (1D-ResNet, Simple RNN, CNN-1D). Below is the summary of the performance on the test set:

| Model | Accuracy | F1-Score (Churn) | Recall (Churn) | Precision (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| **TabNet (Proposed)** | **99.79%** | **99.15%** | **99.08%** | **99.22%** |
| 1D-ResNet | 99.49% | 97.94% | 98.86% | 97.04% |
| Simple RNN | 99.59% | 98.34% | 98.86% | 97.82% |
| CNN 1D | 93.56% | 76.11% | 84.18% | 69.45% |

### Statistical Significance (McNemar's Test)
To ensure the results are not due to chance, McNemar's test was conducted:
* **TabNet vs 1D-ResNet:** Significant ($p < 0.05$)
* **TabNet vs Simple RNN:** Significant ($p < 0.05$)
* **TabNet vs CNN 1D:** Significant ($p < 0.05$)

> **Note:** Without SMOTE-Tomek, the difference between TabNet and ResNet was *not significant*. The hybrid sampling technique proved crucial for TabNet's superior performance.

## 🔍 Model Interpretability (SHAP)

### Global Importance
![SHAP Summary](Customer-Churn-TabNet-SHAP/images/shap_summary_plot.png)
*(Note: Pastikan nama file gambar di folder images sesuai dengan ini)*

* **Balance:** Customers with lower/zero balances tend to churn.
* **NumOfProducts:** Using too many (3-4) or too few products impacts retention.
* **Age:** Older customers show a higher tendency to churn compared to younger ones.

## 💻 Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/ryansyahfathin/Bank-Customer-Churn-Prediction-TabNet-SHAP.git](https://github.com/ryansyahfathin/Bank-Customer-Churn-Prediction-TabNet-SHAP.git)
    cd Bank-Customer-Churn-Prediction-TabNet-SHAP
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Data**
    * Download the dataset from the Kaggle link above.
    * Create a folder named `data` inside the project directory.
    * Place the CSV file inside `data/`.

4.  **Run the Notebook**
    Open `notebooks/Churn_Prediction_TabNet.ipynb` to see the training and evaluation process.

## 📜 References
* Arik, S. O., & Pfister, T. (2021). **TabNet: Attentive Interpretable Tabular Learning**. *AAAI*.
* Lundberg, S. M., & Lee, S. I. (2017). **A Unified Approach to Interpreting Model Predictions**. *NIPS*.
