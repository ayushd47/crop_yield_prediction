import pandas as pd

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')  # Ensure the path is correct
print(data.columns)  # This will print all column names