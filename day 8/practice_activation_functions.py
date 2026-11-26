# Practice 3 - Experiment with Activation Functions

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np

def build_model(activation):
    model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation=activation), # Change activation here
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 1. ReLU Model
print("=== ReLU Model ===")
model_relu = build_model('relu')
model_relu.summary()

# 2. Sigmoid Model
print("\n=== Sigmoid Model ===")
model_sigmoid = build_model('sigmoid')
model_sigmoid.summary()

# 3. Tanh Model
print("\n=== Tanh Model ===")
model_tanh = build_model('tanh')
model_tanh.summary()

# Observation:
# ReLU: Best for hidden layers, fast learning
# Sigmoid: Output 0-1, slow, vanishing gradient problem
# Tanh: Output -1 to 1, better than sigmoid