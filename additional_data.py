import pandas as pd
import numpy as np

# Read the existing data to understand the distributions
existing_data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')

# Districts in Nepal for expanded dataset
additional_districts = [
    "Bhojpur", "Dolakha", "Humla", "Kailali", "Kanchanpur",
    "Mustang", "Myagdi", "Okhaldhunga", "Ramechhap", "Salyan",
    "Sankhuwasabha", "Syangja", "Taplejung", "Udayapur", "Parbat",
    "Nuwakot", "Rasuwa", "Rautahat", "Saptari", "Sarlahi"
]

# Generate 400 more samples
np.random.seed(42)
new_samples = existing_data.sample(400, replace=True).reset_index(drop=True)
new_samples['District'] = np.random.choice(additional_districts, 400)

# Concatenate with existing data
expanded_data = pd.concat([existing_data, new_samples]).reset_index(drop=True)
expanded_data.tail()  # Display the last few rows to check the appended data
