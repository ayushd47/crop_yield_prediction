import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')   # Replace with the actual path to your data file

# Assuming 'Crop_Yield' is the target variable and we are using all other numeric variables as features
X = data.select_dtypes(include=[np.number]).drop('Crop_Yield', axis=1)
y = data['Crop_Yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred_linear = linear_model.predict(X_test)

# Initialize the Random Forest Regressor
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate the models
linear_mse = mean_squared_error(y_test, y_pred_linear)
rf_mse = mean_squared_error(y_test, y_pred_rf)

linear_r2 = r2_score(y_test, y_pred_linear)
rf_r2 = r2_score(y_test, y_pred_rf)

print("Linear Regression MSE:", linear_mse)
print("Random Forest Regression MSE:", rf_mse)
print("Linear Regression R^2 Score:", linear_r2)
print("Random Forest R^2 Score:", rf_r2)

# Optional: Plotting predictions against true values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_linear, color='blue', label='Linear Regression Predictions')
plt.scatter(y_test, y_pred_rf, color='red', label='Random Forest Predictions', alpha=0.5)
plt.title('Regression Predictions vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
