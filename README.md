# Clickstream Conversion Prediction System

## Overview

This project is a machine learning system that analyzes clickstream data to predict user behavior (conversion classes) and provides an interactive Streamlit dashboard for real-time predictions, clustering, and data visualization.

The system follows an end-to-end ML pipeline:
Data preprocessing → Feature engineering → Model training → Evaluation → Deployment via Streamlit

---

## Problem Statement

To predict user navigation behavior (page/category conversion) using historical clickstream data and provide actionable insights through an interactive web application.

---

## Dataset Description

The dataset contains user session-level clickstream data including:

- Session details
- User behavior features
- Product interaction features
- Target variable: `page` (conversion class)

---

## Features

### Data Processing

- Missing value handling (mean/median/mode)
- Feature encoding (Label/One-Hot Encoding)
- Feature scaling (StandardScaler)
- Train-test split

### Machine Learning Models

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Imbalance Handling

- SMOTE (Synthetic Minority Oversampling Technique)

### Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Cross-validation

---

## Streamlit Application Features

The deployed web app includes:

### 1. CSV Upload

- Upload dataset for batch prediction
- View predictions instantly

### 2. Manual Input Prediction

- Enter values manually
- Get real-time prediction output

### 3. Customer Segmentation

- KMeans clustering
- Visual cluster distribution

### 4. Data Visualization

- Bar charts (class distribution)
- Pie charts (prediction distribution)
- Histograms (feature analysis)

---

## Machine Learning Pipeline

Raw Data
↓
Preprocessing (Imputation + Encoding + Scaling)
↓
SMOTE Balancing
↓
Model Training (LR / RF / XGBoost)
↓
Evaluation
↓
Streamlit Deployment

---

## Project Structure

clickstream_project/
│
├── app.py
├── preprocess.py
├── model_train.py
├── save_model.py
│
├── model.pkl
├── preprocessor.pkl
├── label_encoder.pkl
│
├── requirements.txt
├── README.md
│
├── data/

---

## Installation

```bash
pip install -r requirements.txt


## Run application

streamlit run app.py

---
📦 Requirements

Python 3.8+
Streamlit
Pandas
NumPy
Scikit-learn
XGBoost
Imbalanced-learn
Joblib
Matplotlib

 Results

Logistic Regression: High accuracy (baseline model)
Random Forest: Strong performance on non-linear patterns
XGBoost: Best overall performance

Key Learnings


End-to-end ML pipeline development
Handling imbalanced datasets using SMOTE
Model comparison and evaluation
Deployment using Streamlit
Real-time ML inference system

## Future Improvements

Add deep learning models
Deploy on Streamlit Cloud / AWS
Add database integration
Improve feature engineering (session-based modeling)
Add explainability (SHAP values)
Author Dhayalan
Clickstream ML Project


---

#  DONE

This README gives you:

✔ Professional portfolio project  
✔ Interview-ready explanation  
✔ Clear ML pipeline  
✔ Deployment clarity  
✔ GitHub credibility  

---

