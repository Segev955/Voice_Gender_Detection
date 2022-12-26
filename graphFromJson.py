import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import warnings



# Load the JSON file
with open('boys_girls_audio.json', 'r') as f:
    data = json.load(f)

# Extract the data from the JSON file
X = data['males']
print(X)

# Create a PCA model with 2 components
pca = PCA(n_components=1)

# Fit the model to the data and transform the data to the new space
X_pca = pca.fit_transform(X)

# Create a scatter plot of the first and second components of the transformed data
plt.scatter(X_pca[:, 0], [0] * len(X_pca))

plt.xlabel('Component 1')
plt.ylabel('Component 2')

# Show the plot
plt.show()