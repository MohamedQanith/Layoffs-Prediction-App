#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[2]:


df = pd.read_csv(r"C:\Users\Mohamed Qanith\Downloads\Arthat Projects\layoffs_data.csv")


# In[3]:


numerical_cols = ['Funds_Raised', 'Percentage']
imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])


# In[4]:


df['Laid_Off_Count'] = df['Laid_Off_Count'].fillna(df['Laid_Off_Count'].mean())


# In[5]:


features = ['Location_HQ', 'Industry', 'Funds_Raised', 'Stage', 'Percentage', 'Country']
X = df[features]
y = df['Laid_Off_Count']


# In[6]:


categorical_cols = ['Location_HQ', 'Industry', 'Stage', 'Country']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


# In[9]:


model.fit(X_train, y_train)


# In[10]:


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# In[11]:


mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)


# In[12]:


# Calculate metrics for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)


# In[13]:


print(f"Training Data Metrics:")
print(f"MAE: {mae_train}, MSE: {mse_train}, RMSE: {rmse_train}, R2: {r2_train}")

print(f"Test Data Metrics:")
print(f"MAE: {mae_test}, MSE: {mse_test}, RMSE: {rmse_test}, R2: {r2_test}")


# In[14]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# In[15]:


plt.figure(figsize=(12, 6))

# Training data
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
plt.title('Training Data')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Test data
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title('Test Data')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()


# In[16]:


import pickle
# Save the model to a file
with open('best_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[17]:


import streamlit as st
import pandas as pd
import pickle
import datetime

# Title of the app
st.title("Company Layoffs Data Analysis and Prediction App")

# Function to load the dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the data
file_path = "layoffs_data.csv"  # Replace with the path to your CSV file
data = load_data(file_path)

# Display the dataset
st.subheader("Dataset")
st.write(data)

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



# In[ ]:





# In[ ]:




