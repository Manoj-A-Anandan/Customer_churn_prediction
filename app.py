import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data  # Updated to use st.cache_data for data-related caching
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

# Preprocessing
def preprocess_data(df):
    df_encoded = pd.get_dummies(df.drop(columns=['customerID']), drop_first=True)
    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# Model Training
@st.cache_resource  # Updated to use st.cache_resource for model caching
def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# User Input for New Customer Data
def user_input_features():
    gender = st.selectbox("Gender", ("Male", "Female"))
    SeniorCitizen = st.selectbox("Senior Citizen", (0, 1))
    Partner = st.selectbox("Partner", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", ("Yes", "No"))
    tenure = st.slider("Tenure (months)", 1, 72, 1)
    PhoneService = st.selectbox("Phone Service", ("Yes", "No"))
    MultipleLines = st.selectbox("Multiple Lines", ("Yes", "No", "No phone service"))
    InternetService = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    OnlineSecurity = st.selectbox("Online Security", ("Yes", "No", "No internet service"))
    OnlineBackup = st.selectbox("Online Backup", ("Yes", "No", "No internet service"))
    DeviceProtection = st.selectbox("Device Protection", ("Yes", "No", "No internet service"))
    TechSupport = st.selectbox("Tech Support", ("Yes", "No", "No internet service"))
    StreamingTV = st.selectbox("Streaming TV", ("Yes", "No", "No internet service"))
    StreamingMovies = st.selectbox("Streaming Movies", ("Yes", "No", "No internet service"))
    Contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    PaperlessBilling = st.selectbox("Paperless Billing", ("Yes", "No"))
    PaymentMethod = st.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=400.0)
    
    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    
    features = pd.DataFrame([data])
    return features

# CSS styling for the app
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f5;
        font-family: Arial, sans-serif;
    }
    .title {
        text-align: center;
        color: #4B0082;
        margin-bottom: 20px;
    }
    .prediction {
        text-align: center;
        font-size: 1.5em;
        color: #ff6347;
        margin-top: 20px;
    }
    .input-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Streamlit App
st.title("Customer Churn Prediction")

# Load and preprocess the data
df = load_data()
X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess_data(df)

# Train the model
model = train_model(X_train, y_train)

# User input
st.sidebar.header("Enter Customer Data")
input_df = user_input_features()

# Preprocess user input
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
input_scaled = scaler.transform(input_encoded)

# Predict churn
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)[:,1]

# Display results
st.subheader('Prediction')
churn_label = 'Yes' if prediction[0] == 1 else 'No'
st.markdown(f'<div class="prediction">Customer Churn Prediction: {churn_label}</div>', unsafe_allow_html=True)
st.write(f"Churn Probability: {prediction_proba[0]:.2f}")

# Interactive Features
if st.button("Get Detailed Analysis"):
    st.write("### Detailed Analysis")
    st.write("This model predicts whether a customer will churn based on the features provided.")
    st.write("You can adjust the inputs on the sidebar to see how changes affect the prediction.")
