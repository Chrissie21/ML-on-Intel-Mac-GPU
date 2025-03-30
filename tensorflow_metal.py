import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Check if GPU is used
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Create a simple model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Dummy data (MNIST-like)
import numpy as np
X_train = np.random.rand(1000, 784)
y_train = np.random.randint(0, 10, 1000)

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32)
