from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#Generate synthetic data
X,_ =make_blobs(n_samples=300,centers = 4,random_state=42)

#Initialize the KMeans model with 4 clusters
kmeans = KMeans(n_clusters=5,random_state=42)

# Fit the model to the data
kmeans.fit(X)

#Get the cluster centers
centroids = kmeans.cluster_centers_

#Assign labels to the data points
labels = kmeans.labels_

#Plot the clusters and centroids
plt.scatter(X[:,1],X[:, 1],c=labels, cmap = 'viridis')
plt.scatter(centroids[:,0], centroids[:,1], s=300, c='red', marker='x')
plt.title('K-means clustering')
plt.show()
