import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Make sure numpy is imported

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')  # Replace 'path_to_your_data.xlsx' with the actual file path

# Exclude non-numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[np.number])

# Calculate the correlation matrix for numeric data only
correlation_matrix = numeric_data.corr()

# Create the heatmap for the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Heatmap of the Correlation Matrix')
plt.show()
