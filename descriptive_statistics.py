import pandas as pd

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')  # Replace 'path_to_your_data.xlsx' with the actual file path

# Calculate descriptive statistics for the numerical columns
descriptive_stats = data.describe()

# Calculate mode for each column (note: mode can return multiple values if there's a tie)
mode_values = data.mode().iloc[0]  # Taking the first mode value if there are multiple

# Adding mode to the descriptive statistics DataFrame
descriptive_stats.loc['mode'] = mode_values

# Print descriptive statistics
print(descriptive_stats)
