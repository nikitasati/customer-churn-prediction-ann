##import all the required lin 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
import streamlit as st
import pickle

##load the model 
# model=tf.keras.load_model('model.h5')
model = tf.keras.models.load_model('model.h5')

#load the encoderes and scaler
import pickle

##load the encoder and scaler
with open('label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot_encoder=pickle.load(file)

with open('sc.pkl','rb') as file:
    sc=pickle.load(file)



##streamlit app
import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model("model.h5")

with open("sc.pkl", "rb") as file:
    sc = pickle.load(file)


# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer will exit.")


# ==========================================
# INPUTS
# ==========================================

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=600
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=60000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

active_member = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)


# ==========================================
# PREDICT
# ==========================================

if st.button("🔮 Predict"):

    # --------------------------------------
    # 1. Gender Label Encoding
    # --------------------------------------
    
    # Female = 0
    # Male   = 1

    gender_encoded = 1 if gender == "Male" else 0


    # --------------------------------------
    # 2. Yes / No → 1 / 0
    # --------------------------------------

    has_card_encoded = 1 if has_card == "Yes" else 0

    active_member_encoded = 1 if active_member == "Yes" else 0


    # --------------------------------------
    # 3. Create Input DataFrame
    # --------------------------------------

    input_df = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_card_encoded],
        "IsActiveMember": [active_member_encoded],
        "EstimatedSalary": [salary]
    })


    # --------------------------------------
    # 4. Geography → One Hot Encoding
    # --------------------------------------

    geo_encoded = pd.get_dummies(
        input_df["Geography"],
        prefix="Geography"
    )


    # --------------------------------------
    # 5. Remove original Geography
    #    and concatenate encoded columns
    # --------------------------------------

    input_df = pd.concat(
        [
            input_df.drop("Geography", axis=1),
            geo_encoded
        ],
        axis=1
    )


    # --------------------------------------
    # 6. Make columns EXACTLY same as training
    # --------------------------------------

    input_df = input_df.reindex(
        columns=sc.feature_names_in_,
        fill_value=0
    )


    # --------------------------------------
    # 7. Scale input
    # --------------------------------------

    input_scaled = sc.transform(input_df)


    # --------------------------------------
    # 8. ANN Prediction
    # --------------------------------------

    prediction = model.predict(
        input_scaled,
        verbose=0
    )

    probability = prediction[0][0]


    # --------------------------------------
    # 9. Display Result
    # --------------------------------------

    st.divider()

    if probability >= 0.5:

        st.error("⚠️ Customer is likely to EXIT")

        st.write(
            f"Exit Probability: **{probability * 100:.2f}%**"
        )

    else:

        st.success("✅ Customer is likely to STAY")

        st.write(
            f"Exit Probability: **{probability * 100:.2f}%**"
        )

model.save("ann_model.h5")


