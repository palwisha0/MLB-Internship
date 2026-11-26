# Day-8 - Deep Learning & Artificial Neural Networks

## What is Deep Learning?
Deep Learning is a subset of Machine Learning that uses Artificial Neural Networks 
with multiple layers to learn from large amounts of data like images, text and voice.

## Machine Learning vs Deep Learning
| Machine Learning | Deep Learning |
| Works well on small data | Works well on large data |
| Features are manually selected | Features are automatically learned |
| Needs less computing power | Needs GPU / High computing power |

## What is a Perceptron?
Perceptron is the simplest neural network with a single neuron. 
It takes inputs, multiplies with weights, adds bias and passes through 
an activation function to make a decision.

Formula: Output = Activation( Input * Weight + Bias )

## Activation Functions Explored
- **ReLU:** Used in Hidden Layer. Output 0 if negative, else same value.
- **Sigmoid:** Output between 0-1. Used for Binary Classification.
- **Tanh:** Output between -1 to 1. Better than Sigmoid.
- **Softmax:** Used in Output Layer. Gives probability for multi-class.

## Model Used
- Dataset: Fashion MNIST (70,000 images of clothes, 28x28 pixels)
- Architecture: Flatten -> Dense(128, ReLU) -> Dense(10, Softmax)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 10

## Results
- Training Accuracy: ~89%
- Testing Accuracy: ~87%

