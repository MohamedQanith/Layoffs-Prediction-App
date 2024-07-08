import pickle
# Save the model to a file
with open('best_model.pkl', 'wb') as file:
    pickle.dump(model, file)

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict using the model
def predict_layoffs(data):
    prediction = model.predict(data)
    return prediction

# Streamlit application
st.title('Layoff Prediction App')

# Collect user input
st.header('Input Data')
location_hq = st.selectbox('Location HQ', ['Location_1', 'Location_2', 'Location_3'])
industry = st.selectbox('Industry', ['Industry_1', 'Industry_2', 'Industry_3'])
funds_raised = st.number_input('Funds Raised')
stage = st.selectbox('Stage', ['Stage_1', 'Stage_2', 'Stage_3'])
percentage = st.number_input('Percentage')
country = st.selectbox('Country', ['Country_1', 'Country_2', 'Country_3'])

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Location_HQ': [location_hq],
    'Industry': [industry],
    'Funds_Raised': [funds_raised],
    'Stage': [stage],
    'Percentage': [percentage],
    'Country': [country]
})

# Show the user input
st.write('User Input:')
st.write(input_data)

# Make prediction
if st.button('Predict'):
    prediction = predict_layoffs(input_data)
    st.write(f'Predicted Laid Off Count: {prediction[0]}')
