# Customer Churn Prediction System

A Machine Learning-based Customer Churn Prediction System built using Python, Scikit-learn, Pandas, and Streamlit.

This project predicts whether a telecom customer is likely to leave the company based on customer information and service usage data.

---

# Features

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning Model Training
- Customer Churn Prediction
- Interactive Streamlit Dashboard
- Model Accuracy Evaluation

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# Dataset

Dataset used:
- Telco Customer Churn Dataset

Dataset includes:
- Customer demographics
- Service subscriptions
- Monthly charges
- Contract information
- Churn status

---

# Project Structure

```bash
customer-churn-prediction/
│
├── telco_churn.csv
├── main.py
├── app.py
├── churn_model.pkl
├── model_columns.pkl
├── requirements.txt
└── README.md
```

---

# Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Encode Categorical Data
6. Train/Test Split
7. Train Random Forest Model
8. Evaluate Model Accuracy
9. Save Model
10. Build Streamlit Dashboard

---

# Model Used

- Random Forest Classifier

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone <your-github-repository-link>
```

---

## 2. Navigate to Project Folder

```bash
cd customer-churn-prediction
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Machine Learning Script

```bash
python main.py
```

---

## 5. Run Streamlit Dashboard

```bash
python -m streamlit run app.py
```

---

# Dashboard Features

- Customer information input
- Churn prediction
- Real-time prediction results
- Interactive interface

---

# Results

- Successfully trained a customer churn prediction model
- Achieved strong prediction accuracy using Random Forest Classifier
- Built a real-time prediction dashboard using Streamlit

---

# Future Improvements

- Add XGBoost Model
- Hyperparameter Tuning
- Deploy on Streamlit Cloud
- Add More Visualizations
- Improve User Interface

---

# Author

Ramesh Krishan

GitHub:
https://github.com/ramesh-2003-krishan