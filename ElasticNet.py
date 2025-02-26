import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv("D:\Education\PhD\ZZU\Research\FRP\On Going Papers\Paper ( 1 Multi Target RCA)\All Tables and graphs/Database1CSV.csv")

# Assume 'target_cols' are your dependent variables
# Replace them with the actual names of your target columns
target_cols = ['FC']
X = data.drop(target_cols, axis=1)
y = data[target_cols]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train your model
model = MultiTaskElasticNetCV(alphas=np.logspace(-6, 6, 13), l1_ratio=0.5, cv=5)
model.fit(X_train, y_train)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, prefix):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    print(f"{prefix} R-squared: {r2}")
    print(f"{prefix} MAE: {mae}")
    print(f"{prefix} RMSE: {rmse}")
    print(f"{prefix} MAPE: {mape}%")

# Predict on the training set
train_predictions = model.predict(X_train)
# Predict on the test set
test_predictions = model.predict(X_test)

# Calculate metrics for FC in the training set
print("\nTraining Metrics for FC:")
calculate_metrics(y_train, train_predictions, "Training")

# Calculate metrics for FC in the testing set
print("\nTesting Metrics for FC:")
calculate_metrics(y_test, test_predictions, "Testing")

# Scatter plot for FC in the testing set
plt.scatter(y_test, test_predictions)
plt.title('Regression Scatter Plot for FC - Testing Set')
plt.xlabel('Actual FC')
plt.ylabel('Predicted FC')
plt.show()

import pandas as pd

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display actual vs predicted values for FC in the testing set
print("\nActual vs Predicted (FC) - Testing Set:")
actual_vs_predicted_testing = pd.DataFrame({
    'Actual FC': y_test.squeeze(), 'Predicted FC': test_predictions.squeeze()
})
actual_vs_predicted_testing.reset_index(drop=True, inplace=True)
print(actual_vs_predicted_testing)

# Display actual vs predicted values for FC in the training set
print("\nActual vs Predicted (FC) - Training Set:")
actual_vs_predicted_training = pd.DataFrame({
    'Actual FC': y_train.squeeze(), 'Predicted FC': train_predictions.squeeze()
})
actual_vs_predicted_training.reset_index(drop=True, inplace=True)
print(actual_vs_predicted_training)

# Save training and testing actual and predicted values to Excel
save_excel = input("Do you want to save the training and testing actual and predicted values in Excel? Enter 'yes' or 'no': ").lower()

if save_excel == 'yes':
    # Create a directory if it doesn't exist
    save_dir = r"D:\Education\PhD\ZZU\Research\FRP\On Going Papers\Paper (Multi Target)\Algorithms\Elastic Net results"
    os.makedirs(save_dir, exist_ok=True)

    # Save training actual and predicted values to Excel
    train_excel_path = os.path.join(save_dir, "training_actual_vs_predicted.xlsx")
    pd.DataFrame({'Actual FC': y_train.squeeze(), 'Predicted FC': train_predictions.squeeze()}).to_excel(train_excel_path, index=False)
    print(f"Training actual vs predicted values saved successfully at: {train_excel_path}")

    # Save testing actual and predicted values to Excel
    test_excel_path = os.path.join(save_dir, "testing_actual_vs_predicted.xlsx")
    pd.DataFrame({'Actual FC': y_test.squeeze(), 'Predicted FC': test_predictions.squeeze()}).to_excel(test_excel_path, index=False)
    print(f"Testing actual vs predicted values saved successfully at: {test_excel_path}")
else:
    print("Data not saved.")

# Ask the user whether to save the model
save_model = input("Do you want to save the trained model? Enter 'yes' or 'no': ").lower()

if save_model == 'yes':
    # Create a directory if it doesn't exist
    save_dir = r"D:\Education\PhD\ZZU\Research\FRP\On Going Papers\Paper ( 1 Multi Target RCA)\Algorithms\Algorithms\Elastic Net results"
    os.makedirs(save_dir, exist_ok=True)

    # Save the trained model
    joblib.dump(model, os.path.join(save_dir, "your_saved_model_ElasticNet.joblib"))
    print("Model saved successfully.")
else:
    print("Model not saved.")

    # Ask the user whether to run the code again
    run_again = input("Do you want to run the code again? Enter 'yes' or 'no': ").lower()

    if run_again == 'yes':
        exec(open(__file__).read())
    else:
        print("Code execution completed.")
