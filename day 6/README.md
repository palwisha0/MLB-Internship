
# What I Learned About Model Evaluation
Model Evaluation means checking how well a model performs on new data.
I learned about Train vs Test Performance, Overfitting, Underfitting, Cross Validation, and Learning Curves.
I also learned that Accuracy is not always enough. For cancer prediction, Recall is very important because missing one cancer patient can be dangerous.

# What is Hyperparameter Tuning and Why It Matters
Hyperparameter Tuning means finding the best settings for a model before training.
Hyperparameters are settings like C in Logistic Regression.
I used GridSearchCV which tries all combinations and selects the best one.
Tuning is important because the same model can improve from 92% to 97% just by changing its settings.

# Best Parameters Found by GridSearchCV
 C: 1 (or 10, depending on run)
  solver: liblinear
 penalty: l2
 max_iter: 1000

# Comparison Between Baseline and Tuned Models
 Baseline Model: Simple Logistic Regression without tuning
 Tuned Model: Logistic Regression after GridSearchCV
 Tuned model gave slightly higher Accuracy and better Recall and F1-Score.
 Tuned model reduced False Negatives, which is very important for cancer detection.

# Key Observations
1. Scaling improved the model performance.
2. Tuning improved Recall, which means fewer cancer patients were missed.
3. The model is reliable because Train and Test scores are close, so there is no major Overfitting.