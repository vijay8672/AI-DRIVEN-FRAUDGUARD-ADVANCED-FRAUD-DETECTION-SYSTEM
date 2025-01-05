## Fraud Detect AI Advanced Fraud Detection ML Model

Here's a **ready-to-copy-paste** `README.md` file with a navigable Table of Contents and detailed content for your Fraud Detection project:

---

# Fraud Detection Using Machine Learning  

Welcome to the **Fraud Detection Using Machine Learning** project! This repository showcases an end-to-end machine learning pipeline to detect fraudulent transactions in financial datasets.  

---

## Table of Contents  

1. [Introduction](#introduction)  
2. [Problem Statement](#problem-statement)  
3. [Dataset Description](#dataset-description)  
   - [Context](#context)  
   - [Features Overview](#features-overview)  
4. [Data Ingestion](#data-ingestion)  
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
6. [Feature Engineering](#feature-engineering)  
7. [Model Building](#model-building)  
8. [Evaluation Metrics](#evaluation-metrics)  
9. [Deployment](#deployment)  
10. [Conclusion](#conclusion)  

---

## 1. Introduction  

Fraud detection is a critical issue in the financial industry, where fraudulent transactions can lead to significant financial losses. This project leverages machine learning to identify and flag suspicious transactions in a dataset inspired by real-world scenarios.  

---

## 2. Problem Statement  

The goal of this project is to develop a machine learning model capable of detecting fraudulent financial transactions. The key challenge is to minimize false positives and negatives while identifying fraud accurately in a dataset with imbalanced classes.  

---

## 3. Dataset Description  

### Context  
The dataset used in this project is synthetic and generated using the PaySim simulator. It mimics mobile money transactions and includes both normal and fraudulent behaviors to evaluate fraud detection methods.  

### Features Overview  

| **Feature**        | **Description**                                                                                         |  
|---------------------|-------------------------------------------------------------------------------------------------------|  
| `step`             | Time unit in hours (e.g., 1 step = 1 hour). Total steps = 744 (30 days).                              |  
| `type`             | Type of transaction (e.g., CASH-IN, CASH-OUT, TRANSFER, PAYMENT).                                     |  
| `amount`           | Amount of the transaction in local currency.                                                          |  
| `nameOrig`         | ID of the customer initiating the transaction.                                                        |  
| `oldbalanceOrg`    | Initial balance of the sender before the transaction.                                                 |  
| `newbalanceOrig`   | Final balance of the sender after the transaction.                                                    |  
| `nameDest`         | ID of the recipient customer.                                                                         |  
| `oldbalanceDest`   | Initial balance of the recipient before the transaction.                                              |  
| `newbalanceDest`   | Final balance of the recipient after the transaction.                                                 |  
| `isFraud`          | Indicates if the transaction is fraudulent (1 = Fraud, 0 = Not Fraud).                               |  
| `isFlaggedFraud`   | Flags illegal attempts such as transferring over a specific threshold (e.g., > 200,000).              |  

---

## 4. Data Ingestion  

- Data was loaded from a `.csv` file.  
- Preprocessing included:  
  - Removing unnecessary columns (`nameOrig`, `nameDest`) as they don't contribute to fraud detection.  
  - Checking for missing values and handling duplicates.  
  - Converting categorical variables (`type`) into numerical formats using one-hot encoding.  

---

## 5. Exploratory Data Analysis (EDA)  

- Identified class imbalance: Fraudulent transactions were only 0.1% of the dataset.  
- Key findings:  
  - Fraud is mostly associated with `CASH_OUT` and `TRANSFER` types.  
  - Non-fraudulent transactions had significantly higher transaction amounts compared to fraudulent ones.  
- Visualizations included:  
  - Distribution of transaction types.  
  - Fraud vs. non-fraud transaction amounts.  
  - Correlation heatmaps to identify relationships between features.  

---

## 6. Feature Engineering  

- Created new features such as:  
  - `balanceDifference`: Difference between `oldbalanceOrg` and `newbalanceOrig`.  
  - `transactionEfficiency`: Ratio of `newbalanceOrig` to `amount`.  
- Removed highly correlated features to prevent multicollinearity.  

---

## 7. Model Building  

- Models used:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Data split into training (80%) and testing (20%).  
- Handled class imbalance using techniques like SMOTE (Synthetic Minority Oversampling Technique).  
- Hyperparameter tuning was performed using GridSearchCV.  

---

## 8. Evaluation Metrics  

- Metrics used to evaluate the model:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - ROC-AUC score  
- Best-performing model: **XGBoost** with an AUC score of **0.98**.  

---

## 9. Deployment  

The model was deployed as a web application using Flask and Docker.  
- **Steps:**  
  - Built a REST API with Flask to handle predictions.  
  - Containerized the application using Docker.  
  - Deployed the containerized app to Azure Web App Services.  
- **Access URL:**  
  `https://frauddetectai.azurewebsites.net`  

---

## 10. Conclusion  

This project demonstrates the effectiveness of machine learning in fraud detection. The XGBoost model achieved high accuracy and robustness in detecting fraudulent transactions. Future work could involve:  
- Incorporating real-time data for prediction.  
- Enhancing the web app's user interface for better usability.  

---

Let me know if you'd like to adjust any section!
