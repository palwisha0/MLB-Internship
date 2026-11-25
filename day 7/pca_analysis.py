import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA (4D -> 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Retained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Fit K-Means for comparison
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Side-by-side comparison plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: PCA with True Species
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
axes[0].set_title('PCA - True Flower Species')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].grid(True)

# Plot 2: PCA with K-Means Clusters
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
axes[1].set_title('PCA - K-Means Clusters (K=3)')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('pca_vs_kmeans.png', dpi=300)
plt.close()

print("Saved 'pca_vs_kmeans.png'.")