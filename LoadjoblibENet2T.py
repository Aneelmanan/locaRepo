import numpy as np
import pandas as pd
import joblib
# Example usage for 'Fc'
load_path_fc = r"D:\Education\PhD\ZZU\Research\Papers\PhD research\Elastic Net results\your_saved_model_ElasticNet.joblib"

# Load the XGBoost model
model = joblib.load(load_path_fc)

# Define target columns based on your specific use case
target_cols = ['Fc', 'Fy']

# Example user input values (replace with your actual input values)
user_input_values = [0.45,	0.25,	10,	30,	7]

# Reshape input for prediction
user_input = np.array(user_input_values).reshape(1, -1)

# Get predictions based on user input
user_predictions = model.predict(user_input)

# Display the predictions
print("\nPredictions based on user input:")
print(pd.DataFrame(user_predictions, columns=target_cols))
