import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load Data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Accuracy Graph
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.savefig('accuracy_graph.png')
plt.show()

# --- Loss Graph ---
import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_graph.png')
plt.show()
print("Loss Graph saved!")

# Confusion Matrix
pred = model.predict(test_images)
pred_labels = np.argmax(pred, axis=1)
cm = confusion_matrix(test_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.show()

# 10 Correct + 10 Wrong
correct = np.where(pred_labels == test_labels)[0][:10]
wrong = np.where(pred_labels != test_labels)[0][:10]
plt.figure(figsize=(12,4))
for i, idx in enumerate(correct):
    plt.subplot(2,10,i+1)
    plt.imshow(test_images[idx].reshape(28,28), cmap='gray')
    plt.title(f"Pred: {class_names[pred_labels[idx]]}", fontsize=8)
    plt.axis('off')
for i, idx in enumerate(wrong):
    plt.subplot(2,10,10+i+1)
    plt.imshow(test_images[idx].reshape(28,28), cmap='gray')
    plt.title(f"Pred: {class_names[pred_labels[idx]]}\nTrue: {class_names[test_labels[idx]]}", fontsize=8)
    plt.axis('off')
plt.savefig('prediction_images.png')
plt.show()
# --- Confusion Matrix ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Model se prediction lena
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix banana
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix - Fashion MNIST')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion Matrix saved!")