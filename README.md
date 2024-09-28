
# Customer Churn Prediction

This project is a Streamlit application designed to predict customer churn for a telecommunications company. The application utilizes machine learning techniques, specifically a Random Forest classifier, to analyze customer data and predict whether a customer is likely to churn.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

Customer churn refers to the loss of clients or customers. In this project, we aim to identify which customers are likely to churn based on their historical data. The model uses various features such as customer demographics, account information, and service usage patterns to make predictions.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Random Forest Classifier
- Matplotlib (optional for visualizations)

## Dataset

The dataset used in this project is the Telco Customer Churn dataset. It contains information about customers, including their demographic details and account information. The dataset can be found [here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

### Key Features

- `customerID`: Unique identifier for each customer.
- `gender`: Gender of the customer.
- `SeniorCitizen`: Indicates if the customer is a senior citizen (1 or 0).
- `Partner`: Indicates if the customer has a partner (Yes or No).
- `Dependents`: Indicates if the customer has dependents (Yes or No).
- `tenure`: Number of months the customer has been with the company.
- `PhoneService`: Indicates if the customer has phone service (Yes or No).
- `MonthlyCharges`: The monthly charges for the customer.
- `TotalCharges`: The total charges for the customer.
- `Churn`: Indicates if the customer has churned (Yes or No).

## Installation

To run this project, you will need to have Python installed on your machine. Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Manoj-A-Anandan/Customer_churn_prediction.git

Navigate to the project directory:
bash
cd Customer_churn_prediction

Create a virtual environment:
bash
python -m venv venv
Activate the virtual environment:

On Windows:
bash
venv\Scripts\activate

On macOS/Linux:
bash
source venv/bin/activate

Install the required packages:
bash
pip install streamlit scikit-learn pandas numpy


Usage
Run the Streamlit application:

bash
streamlit run app.py
Open your web browser and navigate to http://localhost:8501 to access the application.

Enter customer data in the provided fields and click on "Get Detailed Analysis" to view the churn prediction and probability.
