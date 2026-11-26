
import pickle
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="centered"
)

# Load saved model and training columns
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        saved_data = pickle.load(file)

    # Supports the dictionary saved by the corrected training code
    if isinstance(saved_data, dict) and "model" in saved_data:
        return saved_data["model"], saved_data["columns"]

    # Supports a model.pkl containing only the model
    return saved_data, None


try:
    model, saved_columns = load_model()
except Exception as e:
    st.error(f"Could not load model.pkl: {e}")
    st.stop()

# Header
st.title("🏡 House Price Predictor")
st.write("Estimate a property's price using machine learning.")
st.divider()

# Input form
st.subheader("Enter Property Details")

with st.form("house_form"):
    area = st.number_input(
        "Area (square feet)",
        min_value=100,
        max_value=50000,
        value=1200,
        step=100
    )

    bedrooms = st.number_input(
        "Number of bedrooms",
        min_value=1,
        max_value=20,
        value=3
    )

    bathrooms = st.number_input(
        "Number of bathrooms",
        min_value=1,
        max_value=20,
        value=2
    )

    age = st.number_input(
        "Age of property (years)",
        min_value=0,
        max_value=200,
        value=5
    )

    location = st.selectbox(
        "Location",
        ["Madurai", "Chennai", "Coimbatore", "Nagercoil"]
    )

    submitted = st.form_submit_button(
        "Predict House Price",
        type="primary",
        use_container_width=True
    )

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": location
    }])

    # Apply the same one-hot encoding used during training
    input_data = pd.get_dummies(
        input_data,
        columns=["Location"]
    )

    if saved_columns is not None:
        input_data = input_data.reindex(
            columns=saved_columns,
            fill_value=0
        )

    try:
        predicted_price = float(model.predict(input_data)[0])

        if predicted_price < 0:
            st.warning("The model returned a negative price. Check your dataset and model.")
        else:
            st.success("Prediction completed!")
            st.metric(
                label="Estimated House Price",
                value=f"₹{predicted_price:,.0f}"
            )

            st.caption(
                "This is a machine-learning estimate, not a verified market valuation."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
