# Day 5 - Model Evaluation & Classification

## What is Classification?
Classification is a Supervised Learning technique where the model predicts a categorical label / class. Example: Predicting Iris flower species (Setosa, Versicolor, Virginica) from its features.

## Difference between Regression and Classification
- **Regression:** Predicts continuous numbers (e.g., house price).
- **Classification:** Predicts discrete categories (e.g., spam or not spam).

## Evaluation Metrics Used
1. **Accuracy:** Overall correctness of the model.
2. **Precision:** How many predicted positives were actually correct.
3. **Recall:** How many actual positives were correctly identified.
4. **F1-Score:** Harmonic mean of Precision and Recall.
5. **Confusion Matrix:** Table showing TP, TN, FP, FN - helps understand type of errors.

## Model Performance and Observations
- **Dataset:** Iris Dataset (150 samples, 4 features, 3 classes)
- **Split:** 80% Training, 20% Testing
- **Logistic Regression:** Accuracy ~100%, excellent performance.
- **Decision Tree:** Accuracy ~100%, also performed very well but can overfit on larger datasets.
- **Observation:** Both models performed well because Iris dataset is clean and well-separated. Logistic Regression is preferred because it is simpler and more stable.

## Files Included
- classification_practice.py
- iris_classification.py
- decision_tree_comparison.py
- confusion_matrix_logistic.png
- confusion_matrix_tree.png
- README.md