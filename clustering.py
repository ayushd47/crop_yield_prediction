import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')   # Replace with the actual path to your data file

# Selecting a subset of features for clustering, if necessary
# Adjust this based on your specific dataset, here assuming two features for easy visualization
features = ['Crop_Yield', 'Nitrogen']  # Replace 'Feature1', 'Feature2' with actual column names
X = data[features]

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Assuming you want to create 3 clusters
kmeans.fit(X_scaled)

# Assigning cluster labels to the original data
data['Cluster'] = kmeans.labels_

# Visualizing the clusters if the data is two-dimensional
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=data['Cluster'], palette='viridis', s=100, alpha=0.6, legend='full')
plt.title('Cluster Visualization')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.legend(title='Cluster')
plt.show()
