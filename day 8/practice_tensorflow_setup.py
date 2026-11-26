# Practice 1 - Install & Verify TensorFlow

import tensorflow as tf
from tensorflow import keras

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", keras.__version__)

# Check GPU
print("GPU Available:", tf.config.list_physical_devices('GPU'))

print("TensorFlow Installed Successfully!")