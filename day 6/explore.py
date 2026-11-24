

import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. Load Dataset
data = load_breast_cancer()

# 2. Convert to DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# 3. Basic Information
print("=== DATASET OVERVIEW ===")
print("Shape of dataset:", X.shape)
print("Number of Features:", X.shape[1])
print("Number of Samples:", X.shape[0])

print("\n=== FIRST 5 ROWS ===")
print(X.head())

print("\n=== INFO ===")
print(X.info())

print("\n=== DESCRIBE ===")
print(X.describe())

print("\n=== TARGET DISTRIBUTION ===")
print(y.value_counts())
print("\nTarget Names:", data.target_names)
print("0 = Malignant (Dangerous Cancer)")
print("1 = Benign (Not Dangerous)")

print("\n=== MISSING VALUES ===")
print("Total Missing Values:", X.isnull().sum().sum())
