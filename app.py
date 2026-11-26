import streamlit as st
import pickle
import os

print(os.getcwd())

model = pickle.load(
    open("model.pkl","rb")
)

st.set_page_config(
    page_title="Customer Prediction"
)

st.title("🛒 Customer Purchase Prediction")

st.write(
"Predict whether customer will buy product"
)

age = st.number_input(
    "Enter Age",
    18,
    70
)

income = st.number_input(
    "Enter Income",
    10000,
    200000
)

purchase = st.number_input(
    "Previous Purchase Count",
    0,
    20
)


if st.button("Predict"):


    result = model.predict(
        [[
            age,
            income,
            purchase
        ]]
    )


    if result[0] == 1:

        st.success(
        "Customer Will Buy Product ✅"
        )

    else:

        st.error(
        "Customer Will Not Buy Product ❌"
        )