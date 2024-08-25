import pandas as pd
import folium

# Load your data
data = pd.read_excel('mapping.xlsx')  # Make sure this path is correct

# Ensure your dataset has 'latitude' and 'longitude' columns
map_center_latitude = data['latitude'].mean()
map_center_longitude = data['longitude'].mean()

# Create a Folium map centered at the average latitude and longitude
map = folium.Map(location=[map_center_latitude, map_center_longitude], zoom_start=10)

# Add points to the map
for idx, row in data.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Crop_Yield: {row['Crop_Yield']}",  # This is the popup that shows when you click on the marker
          # This is the tooltip that shows on hover
    ).add_to(map)

# Save the map as an HTML file
map.save('map.html')
print("Map has been saved to 'map.html'")
