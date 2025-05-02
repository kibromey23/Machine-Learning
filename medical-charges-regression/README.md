# 🧠 Medical Charges Regression - Supervised Learning Project

## 🔍 Project Overview
This project aims to predict individual **medical insurance charges** using demographic and lifestyle data. It follows a complete supervised learning pipeline: data exploration, preprocessing, model training, evaluation, and analysis.

---

## 📁 Dataset
- Source: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Records: 1,338 individuals
- Features:
  - `age`, `sex`, `bmi`, `children`, `smoker`, `region`
  - `charges` (target variable)

---

## 📊 Exploratory Data Analysis (EDA)
- **Histograms** for numerical distributions (`age`, `bmi`, `charges`)
- **Countplots** for `sex`, `smoker`, and `region`
- **Correlation Heatmap** and **Pairplots** revealed:
  - High correlation of `smoker`, `age`, and `bmi` with `charges`

---

## ⚙️ Methodology

### 🔄 Preprocessing
- Label Encoding: `sex`, `smoker`
- One-hot Encoding: `region`
- Standard Scaling: Numerical features
- Train/Test Split: 80/20

### 🤖 Models Tested
1. **Linear Regression**
2. **Decision Tree Regressor**
3. ✅ **Random Forest Regressor** *(Best Performer)*

### ✅ Why Random Forest?
- Handles non-linearity and interactions well
- Reduces overfitting through ensembling
- Superior accuracy (R² = 0.8655)

---

## 🧪 Model Performance

| Metric | Value |
|--------|-------|
| **RMSE** | 4569.31 |
| **MAE**  | 2545.83 |
| **R² Score** | 0.8655 |

- Random Forest provided the best generalization and predictive performance.

---

## 📈 Visual Results
- Scatter plot of Actual vs Predicted charges shows a strong fit
- Outliers (very high charges) slightly affect error metrics

---

## 🧠 Conclusion
- Smoking status, age, and BMI are the most influential features
- Random Forest outperformed simpler models
- The pipeline closely follows standard ML workflows

---

## ⚡ Future Improvements
- Apply log transformation to reduce skew
- Use cross-validation and hyperparameter tuning
- Experiment with Gradient Boosting or XGBoost

---

## 📎 Repository Contents
- `medical_charges_regression_project.ipynb` – Jupyter Notebook
- `README.md` – Project documentation
- `insurance.csv` – Dataset
