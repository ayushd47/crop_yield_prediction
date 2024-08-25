import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')   # Replace 'path_to_your_data.xlsx' with the actual file path

# List of some interesting pairs of variables to plot
# Adjust these based on the columns in your dataset
pairs_to_plot = [
    ('pH', 'Crop_Yield'),
    ('Organic_Matter', 'Crop_Yield'),
    ('Nitrogen', 'Crop_Yield'),
    ('Moisture_Content', 'Crop_Yield')
]

# Generate scatter plots for each pair
for x_var, y_var in pairs_to_plot:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_var, y=y_var)
    plt.title(f'Scatter Plot of {y_var} vs. {x_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True)  # Adding grid for better readability
    plt.show()
