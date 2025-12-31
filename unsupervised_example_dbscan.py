from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

#Generate synthetic data with varying densities
X, _ = make_blobs(n_samples=300, centers = 4, cluster_std=[0.60,0.80, 1.00, 1.20], random_state=42)

#Initialize dbscan
dbscan = DBSCAN(eps=0.5, min_samples=5)

#Fit the method
labels = dbscan.fit_predict(X)

#Plot the clusters and noise points
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()
