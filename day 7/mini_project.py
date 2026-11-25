import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

def main():
    # 1. Load and convert to DataFrame
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['true_species'] = iris.target

    print(f"Dataset loaded: {df.shape[0]} samples, {len(iris.feature_names)} features.")

    # 2. Elbow Method
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(iris.data)
        wcss.append(km.inertia_)

    # 3. K-Means Clustering (Optimal K = 3)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(iris.data)

    # 4. PCA Transformation (4D -> 2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(iris.data)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    # Evaluate alignment with true labels (unsupervised metric)
    ari_score = adjusted_rand_score(df['true_species'], df['cluster'])
    print(f"Optimal K selected: {optimal_k}")
    print(f"Variance retained by 2 PCs: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"Adjusted Rand Index (Cluster vs True Species match): {ari_score:.3f}")

    # 5. Unified Visualization (Original Features, K-Means Clusters, PCA Transformed)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Original Data (Selected Features: Petal Length vs Petal Width)
    axes[0].scatter(df['petal length (cm)'], df['petal width (cm)'],
                    c=df['true_species'], cmap='viridis', s=50)
    axes[0].set_title('Original Data (Petal Features)')
    axes[0].set_xlabel('Petal Length (cm)')
    axes[0].set_ylabel('Petal Width (cm)')
    axes[0].grid(True)

    # Panel 2: K-Means Clusters (Original Feature Space)
    axes[1].scatter(df['petal length (cm)'], df['petal width (cm)'],
                    c=df['cluster'], cmap='viridis', s=50)
    axes[1].set_title('K-Means Clusters (K=3)')
    axes[1].set_xlabel('Petal Length (cm)')
    axes[1].set_ylabel('Petal Width (cm)')
    axes[1].grid(True)

    # Panel 3: PCA-Transformed Data Colored by K-Means Clusters
    axes[2].scatter(df['PC1'], df['PC2'], c=df['cluster'], cmap='viridis', s=50)
    axes[2].set_title('PCA 2D Projection (Clusters)')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('mini_project_summary.png', dpi=300)
    plt.close()
    print("Mini project visualization saved as 'mini_project_summary.png'.")

if __name__ == "__main__":
    main()