```markdown
# 🛡️ Health Insurance Premium Prediction Engine

## 🌟 Project Purpose & Value
This repository hosts a comprehensive data science solution for predicting annual health insurance premiums. 

### Why is this useful for everyone?
*   **For Insurance Providers:** Enables data-driven risk assessment, ensuring premiums are competitive yet sustainable, reducing financial loss from underpriced risks.
*   **For Customers:** Provides transparency into how health and lifestyle choices (like weight management or preventive checkups) directly impact their insurance costs.
*   **For Policy Makers:** Offers insights into which public health factors (e.g., BMI, smoking) are driving healthcare costs, aiding in the design of better wellness programs.

---

## 📊 The Dataset at a Glance
We analyzed **25,000 records** containing 24 distinct variables. The data captures a holistic view of an applicant, covering:
*   **Biometrics:** BMI, Glucose, Weight, Fat Percentage.
*   **Medical History:** Previous hospitalizations, disease history.
*   **Lifestyle:** Step counts, alcohol/smoking habits, and exercise intensity.
*   **Demographics:** Age, Occupation, and Location (covering 15 major Indian cities).

---

## 🏗️ Technical Pipeline & Logic

### 1. Data Cleaning (Ensuring Quality)
*   **Imputation:** Missing BMI values were filled using the **Median**. Unlike the mean, the median prevents extreme outliers from biasing the "typical" health profile.
*   **Dimensionality Reduction:** The `Year_last_admitted` column was removed. With nearly 50% missing data, any attempt to fill it would introduce more noise than value.

### 2. Advanced Preprocessing
*   **Winsorization:** Instead of deleting outliers (which represent real, high-risk customers), we used **IQR-based Winsorization** to cap extreme values. This retains the signal of a "high-cost" individual without letting extreme mathematical variance destabilize the model.
*   **Feature Engineering:** We created interaction terms like `Weight × BMI` because health risks often compound; a high weight is statistically more dangerous as BMI increases.

### 3. Machine Learning Excellence
We benchmarked multiple algorithms to find the optimal balance of speed and precision:
*   **Linear & Regularized Regression (Ridge/Lasso):** Provided a strong baseline ($R^2 \approx 0.94$).
*   **Random Forest:** Captured non-linear patterns through ensemble decision trees.
*   **XGBoost (Winner):** Our final choice. It utilizes gradient boosting and L1/L2 regularization to achieve a superior **$R^2$ of 0.956** and the lowest error (RMSE).

---

## 🔍 Strategic Insights (Explainable AI)
Using **SHAP (SHapley Additive exPlanations)**, we moved beyond the "black box" to see what drives costs:
1.  **Weight (96% Impact):** The primary driver of premium variance.
2.  **Existing Coverage:** Applicants already insured elsewhere show different risk profiles, likely due to prior health screenings.
3.  **Preventive Care:** Frequency of checkups is a powerful proxy for health consciousness, inversely correlating with extreme costs.

---

## 🚀 Deployment & Usage

### **Local Execution**
Use the `predict_insurance_cost` function to score new applicants in real-time:
```python
# Example usage
new_prediction = predict_insurance_cost(new_df, best_xgb, scaler, X_train.columns)
```

### **API Access**
The project includes a **Flask API** (`app.py`). This allows other software systems (like a mobile app or web portal) to request predictions via standard web protocols.

**To Start the Server:**
`python app.py`

**To Test the API:**
`curl -X POST -H "Content-Type: application/json" -d @data.json http://localhost:5000/predict`

--- 

## 📦 Environment Requirements
To replicate this project, install the following via pip:
`pip install pandas numpy scikit-learn xgboost shap flask matplotlib seaborn joblib` 
```
