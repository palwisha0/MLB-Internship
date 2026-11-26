# Practice 2 - Build Simple ANN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Build Model: Input -> Hidden -> Output
model = Sequential([
    Flatten(input_shape=(28, 28)),       # Input Layer: 28x28 image ko 784 me convert
    Dense(128, activation='relu'),       # Hidden Layer: 128 neurons
    Dense(10, activation='softmax')       # Output Layer: 10 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Explanation:
# Flatten: 2D image ko 1D bana deta hai
# Dense(128): Hidden layer, 128 neurons, ReLU activation
# Dense(10): Output layer, 10 outputs, Softmax for probability