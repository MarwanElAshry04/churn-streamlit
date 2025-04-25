import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("xgboost_final_model.pkl")
feature_names = model.get_booster().feature_names

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict the likelihood of churn.")

def user_input():
    # Inputs you want to collect from the user
    status = st.selectbox("Customer Status (1 = Active, 0 = Inactive)", [0, 1])
    complains = st.slider("Number of Complaints", 0, 10, 1)
    freq_use = st.slider("Frequency of Use", 0, 250, 50)
    tariff_plan = st.selectbox("Tariff Plan (0 = Pay as you go, 1 = Contract)", [0, 1])
    seconds_use = st.slider("Seconds of Use", 0, 1000, 200)
    age = st.slider("Age", 18, 80, 35)

    # Create a full list of features with default values
    input_dict = {name: 0 for name in feature_names}

    # Override the inputs that the user actually provides
    input_dict.update({
        'Status': status,
        'Complains': complains,
        'Frequency of use': freq_use,
        'Tariff Plan': tariff_plan,
        'Seconds of Use': seconds_use,
        'Age': age,
        'Seconds of Use (Raw)': seconds_use,
        'Charge Amount (Raw)': 5,
        'Customer Value': 3,
        'Estimated_Cost': 2.21,
        'Utilization_Ratio': 0.2,
        'Age_Subscription': 2,
        'Freq_Call_SMS': 4,
        'Avg_Call_Duration': 2,
        'Distinct Called Numbers': 1,
        'Frequency of SMS': 0,
        'Age Group': 3,
        'Call Failure': 0,
        'Subscription Length': 12,
        'Charge Amount': 5
    })

    # Create the final dataframe in the correct order
    df = pd.DataFrame([[input_dict[feature] for feature in feature_names]], columns=feature_names)
    return df

input_df = user_input()

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"This customer is likely to churn. (Probability: {proba:.2f})")
    else:
        st.success(f"This customer is likely to stay. (Probability: {proba:.2f})")
