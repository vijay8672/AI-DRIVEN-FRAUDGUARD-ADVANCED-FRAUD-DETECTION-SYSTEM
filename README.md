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

Fraud detection is a critical challenge faced by financial institutions, e-commerce platforms, and other industries. The Fraud Detect AI project presents an end-to-end machine learning solution to detect fraudulent activities in real-time with high accuracy. This project adheres to industry standards and showcases a robust pipeline, from data ingestion to deployment.

---

## Problem Statement

Fraudulent activities, such as unauthorized transactions and identity theft, cause significant financial losses and damage to brand reputation. Identifying fraud accurately and swiftly is essential to mitigate risks. The challenge lies in distinguishing between genuine and fraudulent activities in a massive dataset with an imbalance between fraudulent and non-fraudulent records.

Objective:
To build a machine learning model capable of detecting fraudulent transactions with high precision and recall while ensuring scalability and real-time inference capabilities.

---

## Dataset 📂 Description

### Context

The dataset used in this project is synthetic and generated using the PaySim simulator. It mimics mobile money transactions and includes both normal and fraudulent behaviors to evaluate fraud detection methods.

 Dataset Link: https://www.kaggle.com/datasets/ealaxi/paysim1/data

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

In this step:  
- The dataset was first **downloaded** from **Kaggle**.  
- It was then **uploaded** to **Azure Blob Storage** for secure storage and accessibility.  
- The data was **read** from Blob Storage and **ingested** into the **local project folder** in **CSV format**.  

This **CSV file** served as the basis for further **analysis** and **processing**.  


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
