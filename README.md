## 📌 Project Overview
This project is a data-driven machine learning system designed to evaluate plastic waste and predict:

- Resale Value (INR)
- Reuse Score (0–10)
- Recycle Score (0–10)

The system aims to support sustainable waste management by providing objective, scalable, and accurate predictions based on item characteristics.

---

## 🎯 Problem Statement
Plastic waste assessment is often manual, inconsistent, and subjective.  
There is a need for an automated system that can reliably estimate:

- Economic resale potential  
- Reusability  
- Recyclability  

This project solves that problem using machine learning models trained on structured plastic waste data.

---

## 🚀 Key Features
- End-to-end machine learning pipeline
- Separate prediction models for resale, reuse, and recycle
- Handles both categorical and numerical features
- High model accuracy with optimized evaluation metrics
- Deployment-ready using Flask

---

## 🧠 Machine Learning Approach
### Data Preprocessing
- Duplicate removal
- Data validation
- Outlier handling
- One-Hot Encoding for categorical features
- Standard Scaling for numerical features

### Models Used
- CatBoost Regressor (Primary Model)
- Gradient Boosting Regressor
- Linear Regression (Baseline)

### Evaluation Metrics
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

CatBoost was chosen due to its superior performance on categorical and tabular data.

---

## 📊 Dataset Description
### Input Features
- Item Category  
- Material Texture  
- Item Color  
- Approximate Size  
- Approximate Quantity  
- Condition  
- Item Usage  
- Recycling Symbol  
- Location  

### Target Variables
- Resale Value (INR)
- Reuse Score (0–10)
- Recycle Score (0–10)

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, CatBoost
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Flask
- **Model Serialization:** Joblib

---

## 📁 Project Structure
