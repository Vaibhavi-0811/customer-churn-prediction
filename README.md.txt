# Customer Churn Prediction & Revenue Risk Analytics

This project is a machine learning-based customer churn prediction system.

It predicts whether a customer is likely to leave a telecom service and calculates the customer's churn probability and risk level.

The project also considers monthly charges to identify potential revenue exposure from high-risk customers.

## Project Objective

The main objective of this project is to help businesses:

- Identify customers who are at risk of churning
- Estimate the probability of customer churn
- Classify customers into Low, Medium, and High Risk
- Understand potential monthly revenue exposure
- Support customer retention decisions using data

## Problem Statement

Customer churn is a major challenge for telecom companies because losing customers can reduce recurring revenue.

The dataset contains customer information such as:

- Customer demographics
- Tenure
- Contract type
- Internet service
- Payment method
- Monthly charges
- Total charges
- Services used
- Churn status

The goal is to use this information to build a machine learning model that can identify customers who are more likely to churn.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SQL
- Power BI
- Jupyter Notebook / Google Colab
- GitHub

## Dataset

The project uses the IBM Telco Customer Churn dataset.

- Number of customers: 7,043
- Number of features: 20 input features
- Target variable: Churn
- Churn = Yes: 1,869 customers
- Churn = No: 5,174 customers

The dataset contains customer demographics, services, contract information, payment methods, tenure, and billing information.

## Data Preparation

The following preprocessing steps were performed:

1. Checked the dataset structure and data types.
2. Checked for missing values and duplicate customer IDs.
3. Converted `TotalCharges` from object type to numeric.
4. Replaced invalid/blank `TotalCharges` values with 0.
5. Removed the `customerID` column because it does not provide useful predictive information.
6. Converted categorical variables into numerical features using one-hot encoding.
7. Split the dataset into training and testing sets using an 80:20 ratio.
8. Used stratified splitting to maintain the churn class distribution.
9. Applied StandardScaler to the features used by the Logistic Regression model.

## Exploratory Data Analysis

I performed exploratory data analysis to understand the relationship between customer characteristics and churn.

The analysis focused on:

- Churn distribution
- Churn by contract type
- Churn by tenure
- Churn by payment method
- Churn by internet service
- Monthly charges and churn
- Contract type combined with tenure

### Key Observations

- Month-to-month customers had a much higher churn rate than customers with one-year or two-year contracts.
- Customers with shorter tenure showed higher churn rates.
- Electronic check customers showed a relatively high churn rate compared with other payment methods.
- Fiber optic customers showed a higher churn rate than DSL and customers without internet service.
- Customers who churned had higher average monthly charges than customers who stayed.
- The combination of month-to-month contracts and short tenure showed particularly high churn in the dataset.

These observations helped me understand the important patterns in the data before building the machine learning models.

## SQL Analysis

I used SQL to perform business-oriented analysis on the customer data.

The analysis included:

- Total number of customers
- Total number of churned customers
- Churned customers by contract type
- Churn rate by contract type
- Monthly revenue associated with churned customers
- Churned monthly revenue by contract type
- Churned customers and associated monthly revenue by payment method

### Business Insights from SQL

- Month-to-month contracts accounted for the largest number of churned customers.
- Month-to-month customers also had the highest observed churn rate among the contract types.
- Electronic check customers represented a large share of churned customers and associated monthly charges.
- The analysis helped connect customer churn with business metrics such as recurring monthly charges.

## Machine Learning

I trained and compared multiple classification models to predict customer churn.

### Models Used

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Training

The dataset was divided into:

- 80% training data
- 20% testing data

Categorical variables were converted into numerical features using one-hot encoding.

StandardScaler was used for Logistic Regression because the model benefits from features being on a comparable scale.

For the Decision Tree and Random Forest models, the encoded features were used without scaling.

## Model Evaluation

Since the dataset contains more customers who did not churn than customers who churned, accuracy alone was not sufficient for evaluating the models.

I evaluated the models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Special attention was given to the recall and F1-score for the churn class because identifying customers who may churn is important for the business use case.

### Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|-------|----------|
| Logistic Regression | 80.70% | 66% | 57% | 61% |
| Decision Tree | 79.42% | 63% | 54% | 58% |
| Random Forest | 80.27% | 67% | 50% | 57% |

## Customer Risk Classification

After obtaining the churn probability from the Logistic Regression model, I converted the probability into three risk levels.

| Churn Probability | Risk Level |
|-------------------|------------|
| Below 30% | Low Risk |
| 30% to 60% | Medium Risk |
| Above 60% | High Risk |

This allows the prediction output to be easier for a business user to understand.

### Revenue Risk

For customers classified as High Risk, their monthly charges were treated as potential monthly revenue exposure.

The system calculates:

Monthly Revenue Exposure = Monthly Charges of High-Risk Customers

This represents the current monthly charges associated with high-risk customers. It should not be interpreted as guaranteed future revenue loss because the dataset does not provide a counterfactual estimate of how long each customer would otherwise remain subscribed.

## Streamlit Application

I deployed the trained machine learning model as an interactive Streamlit web application.

The application allows a user to enter customer information such as:

- Tenure
- Monthly charges
- Total charges
- Contract type
- Internet service
- Payment method
- Customer demographics
- Additional services

After entering the details, the application provides:

- Churn probability
- Churn prediction
- Customer risk level
- Monthly revenue exposure
- A simple customer risk insight

The application makes the machine learning model easier to use without requiring the user to write or run Python code.

## How to Run the Project

### 1. Clone the Repository

Clone this repository to your computer.

### 2. Open the Project Folder

Open Command Prompt or Terminal inside the project folder.

### 3. Install Required Libraries

Run:

```bash
pip install -r requirements.txt
