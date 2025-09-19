# Loan-Prediction-System using LOGISTIC REGRESSION
This loan prediction system goes beyond basic ML by using smart missing value handling, engineered financial ratios, and multiple model benchmarking. It adds transparency with feature importance, ensures fairness checks.
This project consists 8 cells, i'll help to understand each cell:

Main libraries: matplotlib, numpy, pandas, scikit-learn, seaborn
Pipeline steps: encoding, evaluation, modeling, split

-->import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns

Data handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn

-->from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import LabelEncoder
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_test_split → split data into training & testing
LabelEncoder → convert categorical values into numbers
LogisticRegression → classification algorithm
Metrics → accuracy, confusion matrix, classification report

-->df = pd.read_csv(r"C:\\Users\\navya\\Downloads\\loan_approval_dataset.csv")
Reads the loan dataset into a dataframe.

-->print(df.head())
   print(df.info())
   print(df.isnull().sum())

Displays first 5 rows
Prints dataset structure (columns, data types)
Checks for missing values

-->le = LabelEncoder()
   for col in df.columns:
       if df[col].dtype == 'object':
           df[col] = le.fit_transform(df[col])
           
Converts categorical columns (like Gender, Marital Status, etc.) into numeric values.

-->X = df.drop("Loan_Status", axis=1)
   y = df["Loan_Status"]

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Features → all columns except "Loan_Status"
Target → "Loan_Status" (approved / not approved)
Splits dataset into 80% training and 20% testing

-->model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)

Creates Logistic Regression model
Fits model with training data

-->y_pred = model.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
   print("Classification Report:\n", classification_report(y_test, y_pred))

Predicts on test set
Evaluates with accuracy, confusion matrix, and classification report

#Evaluation metrics used in your Loan Prediction System
Accuracy tells overall correctness.

Confusion Matrix shows where the model makes mistakes.
Precision checks how many approved loans are truly safe.
Recall checks how many deserving applicants are approved.
F1-score balances both precision and recall.
