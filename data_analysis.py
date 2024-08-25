import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')

# One-hot encoding for categorical data (if not already encoded)
data = pd.get_dummies(data)

# Separate the features and target variable
X = data.drop('Crop_Yield', axis=1)
y = data['Crop_Yield']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)  # Note: feature scaling not necessary for Random Forest
y_pred_rf = rf_reg.predict(X_test)

# SVM Regression
svm_reg = SVR(kernel='linear')  # You can experiment with different kernels such as 'rbf' or 'poly'
svm_reg.fit(X_train_scaled, y_train)
y_pred_svm = svm_reg.predict(X_test_scaled)

# Evaluating the models
rf_mse = mean_squared_error(y_test, y_pred_rf)
svm_mse = mean_squared_error(y_test, y_pred_svm)
print("Random Forest Regression RMSE:", np.sqrt(rf_mse))
print("SVM Regression RMSE:", np.sqrt(svm_mse))
print("Random Forest R^2 Score:", r2_score(y_test, y_pred_rf))
print("SVM R^2 Score:", r2_score(y_test, y_pred_svm))

# Plotting results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Random Forest Predictions')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_svm, color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('SVM Predictions')

plt.show()
