import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# load MNIST --> 70000 images of handwritten digits
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

#normalize pixel values to [0, 1]
X = X / 255.0

# train.test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train: {y_train.shape}, Test: {y_test.shape}")