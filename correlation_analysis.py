import pandas as pd
import numpy as np  # Make sure numpy is imported
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')  # Replace 'path_to_your_data.xlsx' with the actual file path

# Exclude non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Plotting the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Soil Parameters')
plt.show()
