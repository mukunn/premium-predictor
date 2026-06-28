
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# --- 1. Load Model and Preprocessing Objects ---
# Load the trained XGBoost model
best_xgb = joblib.load('best_xgb_model.joblib')

# Load the fitted StandardScaler
scaler = joblib.load('scaler.joblib')

# Load the training columns to ensure consistent feature order
training_columns = joblib.load('training_columns.joblib')

# --- 2. Define Prediction Function (similar to what we tested) ---
def predict_insurance_cost_api(new_data_df, trained_model, scaler, train_columns):
    """
    Preprocesses new data and makes predictions using a trained model.
    Adapted for API input, expects a DataFrame. 
    """
    
    # 1. Drop 'applicant_id' if present
    if 'applicant_id' in new_data_df.columns:
        new_data_df = new_data_df.drop('applicant_id', axis=1)

    # 2. One-hot encode categorical variables
    # Identify categorical columns in the new data
    categorical_cols = new_data_df.select_dtypes(include=['object']).columns.tolist()
    new_data_processed = pd.get_dummies(new_data_df, columns=categorical_cols, drop_first=True)

    # 3. Align columns with training data
    # Add missing columns (if any) and fill with 0
    missing_cols = set(train_columns) - set(new_data_processed.columns)
    for c in missing_cols:
        new_data_processed[c] = 0

    # Drop columns not present in training data (if any)
    extra_cols = set(new_data_processed.columns) - set(train_columns)
    new_data_processed = new_data_processed.drop(columns=list(extra_cols))

    # Ensure the order of columns is the same as in training data
    new_data_processed = new_data_processed[train_columns]

    # 4. Scale numerical features using the *fitted* scaler
    new_data_scaled = pd.DataFrame(scaler.transform(new_data_processed),
                                   columns=new_data_processed.columns,
                                   index=new_data_processed.index)

    # 5. Make predictions
    predictions = trained_model.predict(new_data_scaled)

    return predictions

# --- 3. Initialize Flask App ---
app = Flask(__name__)

# --- 4. Define Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get JSON data from the request
            data = request.get_json(force=True)
            if not isinstance(data, list):
                data = [data] # Ensure it's a list of records
            
            new_customer_data = pd.DataFrame(data)

            # Make predictions
            predictions = predict_insurance_cost_api(new_customer_data, best_xgb, scaler, training_columns)

            # Return predictions as JSON
            return jsonify({'predictions': predictions.tolist()})

        except Exception as e:
            return jsonify({'error': str(e)}), 400

# --- 5. Run the Flask App (for local development) ---
if __name__ == '__main__':
    # In a production environment, you would typically use a production-ready WSGI server
    # like Gunicorn or uWSGI, and configure for security and scalability.
    # For local testing, you can run:
    # app.run(host='0.0.0.0', port=5000, debug=True)
    pass
