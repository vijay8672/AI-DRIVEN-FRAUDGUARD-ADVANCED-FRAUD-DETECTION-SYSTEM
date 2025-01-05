# Fraud Detect AI - Fraud Detection Model Created Using Machine Learning

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

## Introduction

Fraud detection is a critical issue in the financial industry, where fraudulent transactions can lead to significant financial losses. This project leverages machine learning to identify and flag suspicious transactions in a dataset inspired by real-world scenarios.

---

## Problem Statement

The goal of this project is to develop a machine learning model capable of detecting fraudulent financial transactions. The key challenge is to minimize false positives and negatives while identifying fraud accurately in a dataset with imbalanced classes.

---

## Dataset Description

### Context

The dataset used in this project is synthetic and generated using the PaySim simulator. It mimics mobile money transactions and includes both normal and fraudulent behaviors to evaluate fraud detection methods.

### Features Overview

| **Feature**        | **Description**                                                                                         |
|---------------------|-------------------------------------------------------------------------------------------------------|
| `step`             | Time unit in hours (e.g., 1 step = 1 hour). Total steps = 744 (30 days).                              |
| `type`             | Type of transaction (e.g., CASH-IN, CASH-OUT, TRANSFER, PAYMENT).                                     |
| `amount`           | Amount of the transaction in local currency.                                                         |
| `nameOrig`         | Customer who initiated the transaction.                                                              |
| `oldbalanceOrg`    | Initial balance of the origin account before the transaction.                                         |
| `newbalanceOrig`   | New balance of the origin account after the transaction.                                              |
| `nameDest`         | Customer who is the recipient of the transaction.                                                    |
| `oldbalanceDest`   | Initial balance of the destination account before the transaction.                                    |
| `newbalanceDest`   | New balance of the destination account after the transaction.                                         |
| `isFraud`          | Indicates if the transaction is fraudulent (1 = Fraud, 0 = Not Fraud).                               |
| `isFlaggedFraud`   | Indicates if the transaction was flagged as potentially fraudulent by the system.                     |

---

## Data Ingestion

In this step, the dataset was loaded into the system, and initial checks for missing values, incorrect data types, and anomalies were performed.

---

## Exploratory Data Analysis (EDA)

EDA was conducted to understand the distribution of transactions, the balance between fraud and non-fraud cases, and other insights like transaction types, amounts, and flagged fraud patterns.

---

## Feature Engineering

Features such as `balanceDifference` and `transactionFrequency` were created to improve the model's ability to detect fraud effectively.

---

## Model Building

Several machine learning models, including Logistic Regression, Random Forest, and XGBoost, were tested. XGBoost was found to perform the best with an accuracy of 98%.

---

## Evaluation Metrics

Evaluation was based on precision, recall, F1-score, and the confusion matrix to measure the model's performance on imbalanced data effectively.

---

## Deployment

The model was deployed using Flask and hosted on Azure Web Services. Users can interact with the app to input transaction details and receive real-time fraud detection results.

---

## Conclusion

This project demonstrates the potential of machine learning in detecting fraudulent transactions, highlighting the importance of robust data preprocessing and feature engineering in building effective fraud detection systems.
