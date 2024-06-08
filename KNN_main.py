
# Import the required libraries 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from KNN import *

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Plot the sample data
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
plt.show()

# Fit the KNN Cluster Model
KNN_Model = KNN_Clustering(3)
KNN_Model.fit(X)
  