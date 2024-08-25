import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')  # Replace 'path_to_your_data.xlsx' with the actual file path

# Histograms for all continuous variables
continuous_vars = ['pH', 'Organic_Matter', 'Nitrogen', 'Phosphorus', 'Potassium',
                   'Calcium', 'Magnesium', 'Sulfur', 'Micronutrients', 'CEC', 'EC', 
                   'Bulk_Density', 'Moisture_Content', 'Temperature', 'Crop_Yield']

for var in continuous_vars:
    plt.figure(figsize=(10, 5))
    sns.histplot(data[var], kde=True)
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

# Box plots for all continuous variables
for var in continuous_vars:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data[var])
    plt.title(f'Box Plot of {var}')
    plt.xlabel(var)
    plt.show()
