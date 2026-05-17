# Telecom Churn Predictor

An interactive web app that predicts customer churn for telecom companies using a trained XGBoost model. Built with Streamlit and deployed as a live demo.

## Overview

This app is the deployment component of a full data science project on telecom customer churn. Enter a customer's details and get an instant churn prediction with probability score.

> 📊 See the full data science project here: [telecom_data_science_project](https://github.com/MarwanElAshry04/telecom_data_science_project)

## Features

- Real-time churn prediction via a trained XGBoost model
- Simple input form for customer attributes
- Probability score output

## Tech Stack

- **Python**
- **Streamlit** — web app framework
- **XGBoost** — classification model
- **scikit-learn** — preprocessing

## Run Locally

```bash
pip install -r requirements.txt
streamlit run churn_app.py
```

## Project Structure

```
churn_app.py               # Streamlit app
xgboost_final_model.pkl    # Trained XGBoost model
requirements.txt           # Dependencies
```
