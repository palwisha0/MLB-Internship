import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

# 1. Elbow Method to find optimal K
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Apply K-Means with optimal K = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Visualize clusters using Petal Length (col 2) vs Petal Width (col 3)
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 2], X[:, 3], c=clusters, cmap='viridis', s=50, alpha=0.8)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 2], centroids[:, 3], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering (Petal Length vs Petal Width)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved 'elbow_method.png' and 'kmeans_clusters.png'.")