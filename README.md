# 🚀 Running Machine Learning on an Intel Mac GPU (AMD Radeon Pro 4GB)

## 🛠️ Overview
This guide walks you through setting up **TensorFlow** and **PyTorch** to run machine learning on an **Intel Mac with an AMD Radeon Pro GPU** using **Metal Performance Shaders (MPS)**.

Since macOS does not support CUDA (NVIDIA GPUs), we use **Apple's Metal API** for GPU acceleration. While AMD Radeon GPUs aren't as powerful as NVIDIA's, ML benefits from parallel processing, so it's worth trying!

---

## 🔥 Step 1: Install TensorFlow with Metal (Recommended)
### ✅ 1.1 Create a Conda Environment
```bash
conda create --name ml-metal python=3.9 -y
conda activate ml-metal
```

### ✅ 1.2 Install TensorFlow with Metal Acceleration
```bash
pip install tensorflow-macos tensorflow-metal
```

### ✅ 1.3 Verify TensorFlow Uses Metal GPU
Run the following Python script:
```python
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```
If you see at least `Num GPUs Available: 1`, your Mac is using Metal for TensorFlow!

---

## 🔥 Step 2: Install PyTorch with Metal (Experimental)
### ✅ 2.1 Install PyTorch
```bash
pip install torch torchvision torchaudio
```

### ✅ 2.2 Run a Quick PyTorch Test
```python
import torch

device = torch.device("mps")  # Use Metal GPU
print("Using device:", device)

# Test a simple tensor operation
x = torch.rand((3, 3)).to(device)
print(x)
```
If no errors appear, PyTorch is using Metal!

---

## 🔥 Step 3: Running a Model on Your AMD GPU
### ✅ 3.1 Run a Small Model with TensorFlow-Metal
```python
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
```
If TensorFlow is using **Metal**, you'll see logs mentioning **Metal GPU execution**.

---

### ✅ 3.2 Run a PyTorch Model with Metal
```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("mps")  # Metal GPU

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleNN().to(device)

# Generate random input data
X = torch.rand(1000, 784).to(device)
y = torch.randint(0, 10, (1000,)).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
If no errors appear, **your Mac's GPU is working with PyTorch!** 🎉

---

## 🔥 Step 4: Monitor GPU Usage
To check if your **AMD Radeon Pro GPU** is actually being used:
```bash
ps aux | grep python
```
Or open **Activity Monitor → GPU History** to see the GPU load.

---

## 💡 Conclusion
✅ **YES!** You can run ML on an **AMD Radeon Pro (4GB) GPU** using **Metal (MPS)**.
- For **best performance**, use **TensorFlow-Metal**.
- **PyTorch Metal support is still experimental**.
- **Reduce batch sizes** if memory is insufficient.
- If your model is too large, **consider training on CPU or a cloud GPU**.

🔹 *Need help setting up? Feel free to reach out!* 🚀

