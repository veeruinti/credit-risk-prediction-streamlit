# Credit Risk Prediction System (End-to-End ML + Streamlit)

This project predicts whether a credit customer is **Good** or **Risky** using an end-to-end machine learning pipeline and a Streamlit web app.

## ✨ Features

- Real-world dataset with **150,000+ records**
- Full preprocessing pipeline:
  - Handling missing values (Monthly Income, Number of Dependents)
  - Log transformation of skewed numeric features
  - 5th–95th percentile outlier capping
  - One-Hot encoding (Gender, Region)
  - Ordinal encoding (Occupation, Education, Rented/Owning house)
  - Feature scaling using `StandardScaler`
- Trained and evaluated multiple ML models:
  - KNN, Naive Bayes, Logistic Regression, Decision Tree, Random Forest
- Selected the best model and saved the full pipeline (`model + scaler + encoders + stats`) as `credit_final_model.pkl`
- Deployed via **Streamlit** with a user-friendly UI for entering customer details and viewing:
  - Predicted label: **Good / Bad**
  - Probability of **Good**

## 🧠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-Learn
- Streamlit
- Pickle

## 📂 Project Structure

```text
creditcard_project/
│
├─ app.py                     # Streamlit app (inference UI)
├─ credit_fraud_detection.ipynb   # Training & evaluation notebook
├─ predictions.ipynb          # Testing unseen data locally
├─ credit_final_model.pkl     # Saved model + preprocessing pipeline
├─ creditcard.csv             # Dataset (if included)
├─ requirements.txt           # Python dependencies
└─ README.md                  # Project documentation
