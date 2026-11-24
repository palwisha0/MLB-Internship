# Day-8 - File 4: Final Prediction Pipeline
# Breast Cancer Prediction System - Complete Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("===== BREAST CANCER PREDICTION SYSTEM =====\n")

# ---------- STEP 1: LOAD DATASET ----------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
print("Dataset Loaded Successfully!")
print("Shape:", X.shape)
print("Target Distribution:\n", y.value_counts())

# ---------- STEP 2: TRAIN TEST SPLIT & SCALING ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nData Split Done: 80% Train, 20% Test")

# ---------- STEP 3: BASELINE MODEL ----------
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train_scaled, y_train)
y_pred_baseline = baseline_model.predict(X_test_scaled)

print("\n===== BASELINE MODEL =====")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Precision:", precision_score(y_test, y_pred_baseline))
print("Recall:", recall_score(y_test, y_pred_baseline))
print("F1-Score:", f1_score(y_test, y_pred_baseline))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_baseline))

# ---------- STEP 4: HYPERPARAMETER TUNING ----------
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000]
}

grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print("\n===== TUNING RESULTS =====")
print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

best_model = grid.best_estimator_

# ---------- STEP 5: TUNED MODEL EVALUATION ----------
y_pred_tuned = best_model.predict(X_test_scaled)

print("\n===== TUNED MODEL =====")
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Precision:", precision_score(y_test, y_pred_tuned))
print("Recall:", recall_score(y_test, y_pred_tuned))
print("F1-Score:", f1_score(y_test, y_pred_tuned))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned))

# ---------- STEP 6: COMPARISON ----------
print("\n===== COMPARISON =====")
print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Tuned Accuracy:    {accuracy_score(y_test, y_pred_tuned):.4f}")

# ---------- STEP 7: SAMPLE PREDICTIONS ----------
results = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred_tuned[:10]
})
results['Actual_Label'] = results['Actual'].map({0: 'Malignant', 1: 'Benign'})
results['Predicted_Label'] = results['Predicted'].map({0: 'Malignant', 1: 'Benign'})
results['Correct?'] = results['Actual'] == results['Predicted']
print("\n===== SAMPLE PREDICTIONS (First 10) =====")
print(results)

# ---------- STEP 8: CONFUSION MATRIX HEATMAP ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm1 = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Baseline Model')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm2 = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Tuned Model')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png')
plt.show()
print("\nConfusion Matrix Image Saved as confusion_matrix_comparison.png")