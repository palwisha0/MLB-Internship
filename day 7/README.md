# Day 7: K-Means Clustering & Principal Component Analysis (PCA)

# Overview
This project explores **Unsupervised Learning** using the Iris dataset from Scikit-learn. It demonstrates how to group unlabeled data using **K-Means Clustering**, determine the optimal number of clusters with the **Elbow Method**, and reduce 4-dimensional feature data into 2 dimensions using **PCA** for clear visualization.


# 1. What is Clustering?
Clustering is an unsupervised machine learning technique that groups similar data points together based on their feature similarity without relying on pre-existing labels. Points within the same cluster are highly similar to one another and distinct from points in other clusters.

# 2. What is PCA?
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a smaller set of uncorrelated variables called **Principal Components**. These components capture the maximum possible variance (information) from the original dataset, making complex data easier to visualize and faster to process.

# 3. How the Best Value of $K$ Was Determined
Using the **Elbow Method**, K-Means was run for $K = 1$ through $K = 10$, recording the Within-Cluster Sum of Squares (WCSS) for each run. Plotting WCSS against $K$ revealed a sharp bend ("elbow") at **$K = 3$**, after which adding more clusters yielded minimal reduction in variance.

# Insights & Observations from Visualizations
* **Number of Clusters Formed:** 3 distinct clusters were formed, matching the 3 actual Iris flower species (*Setosa*, *Versicolor*, and *Virginica*).
* **Species Representation:** K-Means separated *Iris setosa* with 100% accuracy because it is linearly separable from the other two species. *Versicolor* and *Virginica* showed slight overlap at their boundary, resulting in a few borderline misclassifications, though overall alignment remained strong (Adjusted Rand Index ~0.73).
* **How PCA Helped in Visualization:** Because the Iris dataset has 4 features, it cannot be plotted directly on a single 2D graph without dropping features. PCA compressed all 4 features into 2 Principal Components while retaining **~97.7% of the total variance**, allowing us to inspect the entire dataset's cluster separation in a single clean 2D scatter plot.
