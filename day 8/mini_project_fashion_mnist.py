# Mini Project - Build Your First ANN on Fashion MNIST

from tensorflow.keras.datasets import fashion_mnist  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Flatten  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

# Class Names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 1. Load Dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
print("Train Labels:", y_train[:10])

# 2. Explore - Show 1 sample image
plt.imshow(X_train[0], cmap='gray')
plt.title(class_names[y_train[0]])
plt.savefig("sample_image.png")
plt.show()

# 3. Normalize - 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4. Build ANN
model = Sequential([
    Flatten(input_shape=(28, 28)),      # Input Layer
    Dense(128, activation='relu'),      # Hidden Layer
    Dense(10, activation='softmax')     # Output Layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 5. Train Model
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test))

# 6. Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Testing Accuracy: {test_acc*100:.2f}%")

# 7. Predictions on 5 Test Images
predictions = model.predict(X_test[:5])
for i in range(5):
    pred = np.argmax(predictions[i])
    print(f"Image {i+1} - Predicted: {class_names[pred]} | Actual: {class_names[y_test[i]]}")

# 8. BONUS - Plot Accuracy Curves
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.savefig("accuracy_graph.png")
plt.show()

# 9. BONUS - Show Sample Predictions with Images
plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test[i], cmap='gray')
    pred = np.argmax(predictions[i])
    plt.title(f"P:{class_names[pred]}\nA:{class_names[y_test[i]]}", fontsize=8)
    plt.axis('off')
plt.savefig("predictions.png")
plt.show()