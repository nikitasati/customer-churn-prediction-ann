# Customer Churn Prediction using ANN

An ANN-based customer churn prediction system with data preprocessing, model prediction, and Streamlit deployment.

## 📌 Project Overview

This project predicts whether a customer is likely to leave (churn) or stay with a company based on customer-related information.

An Artificial Neural Network (ANN) is used to perform the prediction, and the trained model is deployed using Streamlit to provide an interactive web application.

## 🚀 Features

- Customer churn prediction using ANN
- Data preprocessing and feature encoding
- Feature scaling
- Trained deep learning model
- Churn probability prediction
- Interactive Streamlit interface
- Real-time prediction based on user input

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## 🧠 Model

The project uses an Artificial Neural Network (ANN) for binary classification.

The model predicts:

- **STAY** – Customer is likely to remain
- **EXIT** – Customer is likely to leave

The application also displays the customer's exit probability.

## 📂 Project Structure

```text
customer-churn-prediction-ann/
│
├── app.py
├── model.h5
├── sc.pkl
├── label_encoder.pkl
├── onehot_encoder.pkl
├── requirements.txt
└── README.md
