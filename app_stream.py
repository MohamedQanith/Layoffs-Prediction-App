import streamlit as st
import pandas as pd
import pickle
import datetime

# Title of the app
st.title("Company Layoffs Data Analysis and Prediction App")

# Filter by Company
company = st.sidebar.multiselect("Select Company", options=data["Company"].unique())
if company:
    data = data[data["Company"].isin(company)]

# Filter by Country
country = st.sidebar.multiselect("Select Country", options=data["Country"].unique())
if country:
    data = data[data["Country"].isin(country)]

# Filter by Industry
industry = st.sidebar.multiselect("Select Industry", options=data["Industry"].unique())
if industry:
    data = data[data["Industry"].isin(industry)]

# Filter by Date Range
date_range = st.sidebar.date_input("Select Date Range", [])
if date_range:
    if len(date_range) == 2:
        start_date, end_date = date_range
        data = data[(pd.to_datetime(data["Date"]) >= pd.to_datetime(start_date)) & 
                    (pd.to_datetime(data["Date"]) <= pd.to_datetime(end_date))]

# Display filtered data
st.subheader("Filtered Data")
st.write(data)

# Display summary statistics
st.subheader("Summary Statistics")
st.write(data.describe())

# Display total layoffs
st.subheader("Total Layoffs")
total_layoffs = data["Laid_Off_Count"].sum()
st.write(f"Total layoffs: {total_layoffs}")

# Load the pre-trained machine learning model
pickle_file = "best_model.pkl"  # Replace with the path to your pickle file
with open(pickle_file, 'rb') as file:
    model = pickle.load(file)

# Get user input for prediction
st.sidebar.header("Prediction Input")
input_data = {}
for feature in ["Feature1", "Feature2", "Feature3", "Feature4"]:  # Replace with actual feature names
    input_data[feature] = st.sidebar.number_input(f"Input {feature}", value=0.0)

input_df = pd.DataFrame([input_data])

# Make predictions
if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)
    st.subheader("Prediction")
    st.write(prediction)

